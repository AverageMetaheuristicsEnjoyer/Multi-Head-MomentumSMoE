import os

def get_env_var(var_name, default_value):
    return os.environ.get(var_name, default_value)

PARAMS_CONFIG = {
    "env_params": {
        "--distributed": {
            "action": "store_true",
            "default": False,
            "help": "Enable distributed training.",
            "dest": "distributed",
        },
        "--local-rank": {
            "type": int,
            "default": int(get_env_var('LOCAL_RANK', 0)),
            "help": "Local rank for distributed training.",
            "dest": "local_rank",
        },
    },
    "data_params": {
        "--data": {
            "type": str,
            "required": True,
            "help": "Path to the dataset directory (must contain train.txt, valid.txt, test.txt).",
            "dest": "data_path",
        },
    },
    "model_params": {
        "--architecture": {
            "type": str,
            "required": True,
            "help": "String defining layer types (e.g., 'sm' for attention + MoE). 's'=attention, 'm'=MomentumMoE, 'a'=AdamMoE, 'f'=FeedForward.",
            "dest": "architecture",
        },
        "--hidden-size": {
            "type": int,
            "default": 256,
            "help": "Hidden size (model dimension).",
            "dest": "hidden_size",
        },
        "--inner-hidden-size": {
            "type": int,
            "default": 256,
            "help": "Inner hidden size of MoE/FF layers.",
            "dest": "inner_hidden_size",
        },
         "--num-layers": {
            "type": int,
            "default": 8,
            "help": "Number of Transformer layers.",
            "dest": "num_layers",
        },
        "--num-heads": {
            "type": int,
            "default": 8,
            "help": "Number of self-attention heads.",
            "dest": "num_heads",
        },
        "--mhmoe-num-heads": {
             "type": int,
             "default": 1, # Default to 1 (no multi-head MoE)
             "help": "Number of heads for splitting/merging in MH-MoE layers.",
             "dest": "mhmoe_num_heads",
        },
        "--mhmoe-beta": {
            "type": float,
            "default": 1.0,
            "help": "Scaling factor (beta) for the inner hidden size of MH-MoE experts to control parameter count.",
            "dest": "mhmoe_beta",
        },
        "--attn-span": {
            "type": int,
            "default": 256,
            "help": "Length of the attention span.",
            "dest": "attn_span",
        },
        "--dropout": {
            "type": float,
            "default": 0.2,
            "help": "Dropout rate.",
            "dest": "dropout",
        },
        "--num-experts": {
            "type": int,
            "default": 16,
            "help": "Number of experts per MoE layer.",
            "dest": "num_experts",
        },
        "--moe-top-k": {
            "type": int,
            "default": 2,
            "help": "Number of experts to route to per token/sub-token.",
            "dest": "moe_top_k",
        },
        "--gate-name": {
            "type": str,
            "default": "smoe",
            "help": "Type of MoE gate: 'smoe' or 'mhmoe'.",
            "dest": "gate_name",
        },
         "--load-balance": {
            "type": float,
            "default": 0.01,
            "help": "Coefficient for MoE load balancing loss (0 to disable).",
            "dest": "load_balance",
        },
        "--mu": { # MomentumSMoE mu
            "type": float,
            "default": 0.9,
            "help": "Momentum coefficient (mu) for MoE layers.",
            "dest": "mu",
        },
        "--gamma1": { # MomentumSMoE gamma used in Adam
            "type": float,
            "default": 1.0,
            "help": "Adam decay parameter.",
            "dest": "gamma1",
        },
        "--gamma2": { # MomentumSMoE gamma used in heavy-ball and Adam
            "type": float,
            "default": 1.0,
            "help": "Step size (gamma) for MomentumSMoE update.",
            "dest": "gamma2",
        },
        "--beta1": { # MomentumSMoE Adam beta1
            "type": float,
            "default": 0.9,
            "help": "Adam optimizer beta1.",
            "dest": "beta1",
        },
        "--beta2": { # Adam beta2
            "type": float,
            "default": 0.999,
            "help": "Adam optimizer beta2.",
            "dest": "beta2",
        },
        "--alpha": {
            "type": float,
            "default": 5.0,
            "help": "Mixing coefficient for AdEMAMix (weight of slow momentum).",
            "dest": "alpha",
        },
        "--beta3": {
            "type": float,
            "default": 0.9999,
            "help": "Slow momentum decay factor for AdEMAMix.",
            "dest": "beta3",
        },
        "--t-warmup": {
            "type": int,
            "default": 0,
            "help": "Warmup period for AdEMAMix schedulers.",
            "dest": "t_warmup",
        },
    },
    "optim_params": {
        "--lr": {
            "type": float,
            "default": 0.001,
            "help": "Learning rate.",
            "dest": "lr"
        },
        "--optim": {
            "type": str,
            "default": "adam",
            "help": "Optimizer type: 'sgd' or 'adam'.",
            "dest": "optim",
        },
        "--lr-warmup": {
            "type": int,
            "default": 3000,
            "help": "Number of linear learning rate warmup steps (0 to disable).",
            "dest": "lr_warmup",
        },
    },
    "trainer_params": {
         "--epochs": {
            "type": int,
            "default": 10,
            "help": "Number of training epochs.",
            "dest": "epochs",
        },
        "--batch-size": {
            "type": int,
            "default": 128,
            "help": "Batch size per GPU.",
            "dest": "batch_size",
        },
        "--block-sz": {
            "type": int,
            "default": 256,
            "help": "Sequence block size.",
            "dest": "block_size",
        },
        "--batch-split": {
            "type": int,
            "default": 1,
            "help": "Split batches into smaller chunks to fit in memory.",
            "dest": "batch_split",
        },
        "--nbatches": {
            "type": int,
            "default": 1000,
            "help": "Number of batches per training epoch.",
            "dest": "nb_batches_per_iter",
        },
        "--output-dir": {
             "type": str,
             "default": "output/",
             "help": "Directory to save logs and outputs.",
             "dest": "output_dir",
        },
        "--checkpoint": { # REMOVED LEADING SPACE
            "type": str,
            "default": None,
            "help": "Path to save/load model checkpoint. If None, uses output_dir.",
            "dest": "checkpoint_path",
        },
        "--resume": {
            "action": "store_true",
            "default": False,
            "help": "Resume training from the checkpoint if it exists.",
            "dest": "resume",
        },
        "--full-eval-mode": {
            "action": "store_true",
            "default": False,
            "help": "Run in full evaluation mode.",
            "dest": "full_eval_mode",
        },
    },
    "wandb_params": {
        "--use-wandb": {
            "action": "store_true",
            "default": False,
            "help": "Enable logging to Weights & Biases.",
            "dest": "wandb_flag",
        },
        "--wandb-key": { # REMOVED LEADING SPACE
             "type": str,
             "default": None,
             "help": "WandB API key.",
             "dest": "wandb_key",
        },
        "--project-name": {
            "type": str,
            "default": None,
            "help": "WandB project name.",
            "dest": "project_name",
        },
        "--run-name": {
            "type": str,
            "default": None, # If None, WandB generates one
            "help": "WandB run name.",
            "dest": "run_name",
        },
        "--run-id": {
            "type": str,
            "default": None,
            "help": "WandB run id for resuming existing run",
            "dest": "run_id",
        },
        "--wandb-checkpoints": {
            "action": "store_true",
            "default": False,
            "help": "Enable checkpoint saving to wandb",
            "dest": "wandb_checkpoints",
        }
    },
}