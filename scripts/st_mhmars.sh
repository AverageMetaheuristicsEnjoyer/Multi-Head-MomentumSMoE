#!/bin/bash

mkdir -p /home/user/MoE_experiments/output/st_mhmars/
args="
--data ../data/wikitext-103/ \
--architecture srsrsrsrsrsr \
--gate-name mhmoe \
--num-layers 6 \
--hidden-size 352 \
--inner-hidden-size 352 \
--num-heads 8 \
--mhmoe-num-heads 2 \
--mhmoe-beta 1.875 \
--num-experts 16 \
--moe-top-k 2 \
--block-sz 512 \
--attn-span 1024 \
--dropout 0.1 \
--load-balance 0.01 \
--optim adam \
--lr 0.0007 \
--eta-min 7e-6 \
--lr-warmup 4000 \
--clip 1.0 \
--cosine-decay \
--epochs 40 \
--batch-size 48 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /home/user/MoE_experiments/output/st_mhmars/model.pt \
--gamma1 1.0 \
--gamma2 1.0 \
--project-name arch-refinement \
--run-name st_mhmars \
--wandb-save-every 10 \
"

echo "Training..."
CUDA_VISIBLE_DEVICES="5" torchrun --standalone --nproc_per_node=1 train.py $args --use-wandb

echo "Evaluating..."
CUDA_VISIBLE_DEVICES="5" torchrun --standalone --nproc_per_node=1 train.py $args --resume --full-eval-mode
