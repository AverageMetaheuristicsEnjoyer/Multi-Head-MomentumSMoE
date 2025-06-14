import os, sys
import argparse
import math, random
import functools
import os, shutil
import torch
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
    get_state_dict,
    set_state_dict
)
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
import tqdm
from gates import CustomNaiveGate_Balance_SMoE, MHMoEGate
import wandb


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, "a+") as f_log:
            f_log.write(s + "\n")


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print("Debug Mode : no experiment dir created")
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    print("Experiment dir : {}".format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, "scripts")
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, "log.txt"))


def freeze_gate_weight(model):
    """Freeze the router/gate weights in the model"""
    print("* Freezing Router Weights")
    for name, p in model.named_parameters():
        if "gate.gate" in name:
            print("Freeze: ", name)
            p.requires_grad = False


def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:  # e.g., 'model_params'
        for param_flag, param_config in params_config[params_category].items():
            # e.g., param_flag = '--block-sz'
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)


def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)
    return {
        params_category: {
            param_config["dest"]: namespace.__getattribute__(param_config["dest"])
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }


##############################################################################
# ENVIRONMENT
##############################################################################


def _torch_distributed_init_process_group(local_rank):
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print("my rank={} local_rank={}".format(rank, local_rank))
    torch.cuda.set_device(local_rank)
    return {
        "rank": rank,
        "world_size": world_size,
    }


def set_up_env(env_params):
    assert torch.cuda.is_available()
    if env_params["distributed"]:
        env_params.update(
            _torch_distributed_init_process_group(local_rank=env_params["local_rank"])
        )
    env_params["device"] = torch.device("cuda")


##############################################################################
# OPTIMIZER AND SCHEDULER
##############################################################################


def _get_grad_requiring_params(model):
    nb_parameters = 0
    grad_requiring_params = []
    for param in model.parameters():
        if param.requires_grad:
            nb_parameters += param.numel()
            grad_requiring_params.append(param)
    print("nb_parameters={:.2f}M".format(nb_parameters / 1e6))
    return grad_requiring_params


def _get_optimizer(model, optim, lr, momentum=0.0, grad_clip=0.0):
    if optim == "sgd":
        return torch.optim.SGD(
            _get_grad_requiring_params(model), lr=lr, momentum=momentum
        )
    elif optim == "adam":
        return torch.optim.Adam(
            _get_grad_requiring_params(model),
            lr=lr,
        )
    else:
        raise RuntimeError("wrong type of optimizer - must be 'sgd' or 'adam'")


def _get_scheduler(optimizer, lr_warmup):
    if lr_warmup > 0:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / lr_warmup)
        )
    return None


def get_optimizer_and_scheduler(model, optim_params):
    # Handling missing parameters with defaults
    momentum = optim_params.get("momentum", 0.0)
    grad_clip = optim_params.get("grad_clip", 0.0)
    
    optimizer = _get_optimizer(
        model=model,
        optim=optim_params["optim"],
        lr=optim_params["lr"],
        momentum=momentum,
        grad_clip=grad_clip,
    )
    scheduler = _get_scheduler(optimizer=optimizer, lr_warmup=optim_params.get("lr_warmup", 0))
    return optimizer, scheduler


##############################################################################
# CHECKPOINT
##############################################################################

class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

def _load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger, distributed, sharded):
    print("loading from a checkpoint at {}".format(checkpoint_path))
    if distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        checkpoint_state = torch.load(
            checkpoint_path,
            map_location = "cpu",
            mmap = True,
            weights_only=True,
        )
    else:
        checkpoint_state = torch.load(checkpoint_path, weights_only=True)
    iter_init = checkpoint_state["nb_batches_per_iter"] + 1  # next iteration
    if sharded:
        set_model_state_dict(
            model = model,
            model_state_dict = checkpoint_state["app"]["model"],
            options = StateDictOptions(
                full_state_dict = True,
                broadcast_from_rank0 = True,
                strict = True,
            ),
        )
        set_optimizer_state_dict(
            model = model,
            optimizers = optimizer,
            optim_state_dict = checkpoint_state["app"]["optim"],
            options = StateDictOptions(
                full_state_dict = True,
                broadcast_from_rank0 = True,
                strict = True,
            )
        )
        if scheduler is not None and "scheduler_iter" in checkpoint_state:
            # we only need the step count
            scheduler.step(checkpoint_state["scheduler_iter"])
        return iter_init
    
    model.load_state_dict(checkpoint_state["app"]["model"])
    optimizer.load_state_dict(checkpoint_state["app"]["optim"])
    if scheduler is not None and "scheduler_iter" in checkpoint_state:
        # we only need the step count
        scheduler.step(checkpoint_state["scheduler_iter"])
    return iter_init


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, logger, distributed, sharded, resume, wandb_params):
    if resume:
        run_id = wandb_params.get("run_id", None)
        wandb.init(project=wandb_params["project_name"], id = run_id, resume = "allow")
        wandb_flag = wandb_params.get("wandb_flag", False)
        if os.path.exists(checkpoint_path):
            return _load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                distributed=distributed,
                sharded = sharded,
            )
        elif wandb_flag:
            print("Local checkpoint not found, attempting to download from wandb")
            artifact = wandb.use_artifact(f"{wandb.run.name}:latest", type = "model")
            
            artifact_dir = artifact.download(root = os.path.dirname(checkpoint_path))
            checkpoint_path = os.path.join(artifact_dir, os.path.basename(checkpoint_path))
            return _load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                distributed=distributed,
                sharded = sharded,
            )
        else:
            print("Failed to load checkpoint")
    return 0


def save_checkpoint(
    checkpoint_path, nb_batches_per_iter, model, optimizer, scheduler, sharded, wandb_flag, wandb_save_every
):
    if checkpoint_path:
        if sharded:        
            state_dict = {"app": AppState(model, optimizer)}
            dcp.save(state_dict, checkpoint_id = os.path.dirname(checkpoint_path))

            # ugly workaround
            dcp_to_torch_save(os.path.dirname(checkpoint_path), checkpoint_path)
            checkpoint_state = torch.load(checkpoint_path, weights_only=True)

            checkpoint_state["nb_batches_per_iter"] = nb_batches_per_iter
        
        else:
            state_dict = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
            }
            checkpoint_state = {
                "nb_batches_per_iter": nb_batches_per_iter,
                "app": state_dict,
            }

        if scheduler is not None:
            checkpoint_state["scheduler_iter"] = scheduler.last_epoch
        torch.save(checkpoint_state, checkpoint_path)

        if wandb_flag and (wandb_save_every > 0 and nb_batches_per_iter % wandb_save_every == 0):
            model_artifact = wandb.Artifact(
                name = wandb.run.name,
                type = "model"
            )

            model_artifact.add_file(checkpoint_path)
            wandb.log_artifact(model_artifact)


##############################################################################
# LOGGER
##############################################################################


class Logger:
    def __init__(self):
        self._state_dict = dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def _log(self, title, value):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)

    def log_iter(
        self, iter_no, nb_batches_per_iter, loss_train, loss_val, elapsed, model
    ):
        step = (iter_no + 1) * nb_batches_per_iter
        train_bpc = float(loss_train / math.log(2))
        val_bpc = float(loss_val / math.log(2))
        msg = "steps: {}".format(step)
        msg += "\ttrain: {:.3f}bpc\tval: {:.3f}bpc".format(train_bpc, val_bpc)
        msg += "\tms/batch: {:.1f}".format(elapsed)
        self._log(title="step", value=step)
        self._log(title="train_bpc", value=train_bpc)
        self._log(title="val_bpc", value=val_bpc)

        # Log MoE-specific information if available
        moe_stats = []
        for name, m in model.named_modules():
            if isinstance(m, CustomNaiveGate_Balance_SMoE) or isinstance(m, MHMoEGate):
                if hasattr(m, 'loss') and m.loss is not None:
                    moe_stats.append(m.loss.item())
        
        if moe_stats:
            avg_moe_loss = sum(moe_stats) / len(moe_stats)
            self._log("moe_balance_loss", avg_moe_loss)
            msg += "\tmoe_balance_loss: {:.5f}".format(avg_moe_loss)

        print(msg)