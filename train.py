import os, sys
import warnings

warnings.filterwarnings("ignore")

import argparse
import math, random
import torch
import time

from config import PARAMS_CONFIG
from data import get_train_val_test_data
from models import TransformerSeq
from trainer import train_iteration, full_eval
import datetime
import wandb
import os
from utils import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    create_exp_dir,
    Logger,
)


def launch(
    env_params,
    model_params,
    optim_params,
    data_params,
    trainer_params,
    wandb_params,
):
    # Initialize wandb only on the master process (rank 0)
    distributed = env_params["distributed"]
    is_master = not distributed or env_params.get("local_rank", 0) == 0
    
    wandb_flag = wandb_params.get("wandb_flag", False)
    wandb_checkpoints = wandb_params.get("wandb_checkpoints", False)
    if wandb_flag and is_master:
        run_id = wandb_params.get("run_id", None)

        wandb.init(project=wandb_params["project_name"], id = run_id, resume = "allow")
        wandb.run.name = wandb_params["run_name"] if wandb_params["run_name"] else wandb.run.name
        wandb.config.update(data_params)
        wandb.config.update(model_params)
        wandb.config.update(optim_params)
        wandb.config.update(trainer_params)
    
    # global val
    best_val_loss = None
    
    # ENVIRONMENT (device, distributed, etc.)
    set_up_env(env_params)
    device = env_params["device"]
    world_size = env_params.get("world_size", 1)
    resume = trainer_params["resume"]

    if is_master:
        print("data_params:\t", data_params)
        print("model_params:\t", model_params)
        print("optim_params:\t", optim_params)
        print("trainer_params:\t", trainer_params)

    # DATA
    train_data, val_data, test_data = get_train_val_test_data(
        data_params=data_params,
        env_params=env_params,
        batch_size=trainer_params["batch_size"],
        device=device,
    )


    # MODEL
    model = TransformerSeq(
        vocab_size=data_params["vocab_size"],
        world_size = world_size,
        **model_params,
    )
    if is_master:
        print(model)
    
    if distributed:
        local_rank = env_params["local_rank"]
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    # OPTIMIZER AND SCHEDULER
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params
    )

    # create logger - only on master process
    if is_master:
        logger = Logger()
        fold_name = trainer_params["checkpoint_path"].split("/")[-1].split(".")[0]
        folder_path = "/".join(trainer_params["checkpoint_path"].split("/")[:-1])
        logging = create_exp_dir(f"{folder_path}/experiments/{fold_name}")
        
        # log parameters
        logging(f"Training Parameters:\n {trainer_params}")
        logging(f"Models Parameters:\n {model_params}")
        
        # logging time
        current_time = datetime.datetime.now()
        logging(str(current_time))

        original_model = model.module
        
        # log model
        logging(str(original_model))
        logging(f"Total of Parameters: {sum(p.numel() for p in original_model.parameters())}")
        logging(
            f"Total of Trainable Parameters: {sum(p.numel() for p in original_model.parameters() if p.requires_grad)}"
        )
    else:
        logger = None
        logging = lambda x: None  # Dummy logging function for non-master processes
    
    # resume training from last checkpoint if exists
    iter_init = load_checkpoint(
        trainer_params["checkpoint_path"],
        model,
        optimizer,
        scheduler,
        logger,
        distributed,
        resume,
        wandb_params
    )
    
    # calculate time
    start_time = time.time()
    
    # eval model
    if trainer_params.get("full_eval_mode", False):
        # evaluate the model on test data
        with torch.no_grad():
            loss_val = full_eval(
                model,
                optimizer,
                scheduler,
                val_data,
                trainer_params["block_size"],
                model_params["hidden_size"],
            )
            loss_test = full_eval(
                model,
                optimizer,
                scheduler,
                test_data,
                trainer_params["block_size"],
                model_params["hidden_size"],
            )
            if distributed:
                # collect results into rank0
                stats = torch.tensor([loss_val, loss_test]).to(device)
                torch.distributed.reduce(stats, 0)
                if is_master:
                    loss_val = stats[0] / env_params["world_size"]
                    loss_test = stats[1] / env_params["world_size"]
                else:
                    return

            if is_master:
                if ("enwik8" in data_params["data_path"]) or (
                    "text8" in data_params["data_path"]
                ):
                    logging("Val: {:.3f} BPC".format(loss_val / math.log(2)))
                    logging("Test: {:.3f} BPC".format(loss_test / math.log(2)))
                else:
                    logging("Val: {:.3f} PPL".format(math.exp(loss_val)))
                    logging("Test: {:.3f} PPL".format(math.exp(loss_test)))
        return

    # position of current batch
    data_pos = [0] * 2
    
    # initialize caches for train and valid
    hid_cache = [
        [
            torch.zeros(
                train_data.size(0),
                model.module.layers[layer_i].attn.attn.get_cache_size() if model.module.layers[layer_i].use_attn else 0,
                model_params["hidden_size"],
            ).to(device)
            for layer_i in range(len(model.module.layers))
        ]
        for _ in range(2)
    ]

    nb_batches_per_iter = trainer_params["nb_batches_per_iter"]
    num_epochs = trainer_params.get("epochs", 5)
    for iter_no in range(iter_init, num_epochs):
        # time storing
        t_sta = time.time()
        loss_train, data_pos[0], hid_cache[0] = train_iteration(
            model,
            model_params["load_balance"],
            optimizer,
            scheduler,
            train_data,
            nb_batches_per_iter,
            trainer_params["block_size"],
            False,
            data_pos[0],
            hid_cache[0],
            trainer_params["batch_split"],
            trainer_params["checkpoint_path"],
        )
        elapsed = 1000 * (time.time() - t_sta) / nb_batches_per_iter
        with torch.no_grad():
            loss_val, data_pos[1], hid_cache[1] = train_iteration(
                model,
                model_params["load_balance"],
                optimizer,
                scheduler,
                val_data,
                nb_batches_per_iter,
                trainer_params["block_size"],
                True,
                data_pos[1],
                hid_cache[1],
                trainer_params["batch_split"],
                trainer_params["checkpoint_path"],
            )

        if distributed:
            # collect results into rank0
            stats = torch.tensor([loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0)
            if is_master:
                loss_train = stats[0] / env_params["world_size"]
                loss_val = stats[1] / env_params["world_size"]
            else:
                continue
                
        if is_master:
            logging(f"=================== EPOCHS {iter_no} ======================")
            if ("enwik8" in data_params["data_path"]) or (
                "text8" in data_params["data_path"]
            ):
                msg_result = "Epochs: {} | loss_train: {:.3f} ~ {:.3f} BPC | loss_val: {:.3f} ~ {:.3f} BPC | elapsed: {:.1f}".format(
                    iter_no,
                    loss_train,
                    float(loss_train / math.log(2)),
                    loss_val,
                    float(loss_val / math.log(2)),
                    elapsed,
                )
            else:
                msg_result = "Epochs: {} | loss_train: {:.3f} ~ {:.3f} PPL | loss_val: {:.3f} ~ {:.3f} PPL | elapsed: {:.1f}".format(
                    iter_no,
                    loss_train,
                    float(math.exp(loss_train)),
                    loss_val,
                    float(math.exp(loss_val)),
                    elapsed,
                )
            logging(msg_result)
            
            # Log to wandb only from the master process
            if wandb_flag:
                wandb.log({
                    'train_ppl': float(math.exp(loss_train)), 
                    'Epoch': iter_no, 
                    'valid_ppl': float(math.exp(loss_val)),
                    'train_loss': loss_train,
                    'valid_loss': loss_val,
                    'elapsed_ms': elapsed
                })
            
            logger.log_iter(iter_no, nb_batches_per_iter, loss_train, loss_val, elapsed, model)
            
            # Save the model if the validation loss is the best we've seen so far.
            if (best_val_loss is None) or loss_val < best_val_loss:
                best_val_loss = loss_val
                save_checkpoint(
                    trainer_params["checkpoint_path"],
                    iter_no,
                    model,
                    optimizer,
                    scheduler,
                    wandb_flag,
                    wandb_checkpoints,
                )
    
    if is_master:
        end_time = time.time()
        logging(f"Training time total: {(end_time - start_time)/3600} h")


if __name__ == "__main__":
    launch(**get_params(params_config=PARAMS_CONFIG))