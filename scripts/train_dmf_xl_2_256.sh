#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate dmf

DATA_DIR="PATH/TO/IMAGENET256/"
OUTPUT_DIR="PATH/TO/OUTPUT/"
WANDB_DIR="PATH/TO/WANDB_LOG/"
WANDB_PROJ="WANDB_PROJECT_NAME"
PRETRAINED_CKPT="PATH/TO/PRETRAINED/CKPT"
EXP_NAME="dmf-xl-2-256"

accelerate launch \
  --num_machines 1 \
  --multi_gpu \
  --num_processes 8 \
  train_dmf.py \
  --torch-compile \
  --project-name "${WANDB_PROJ}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --wandb-dir "${WANDB_DIR}" \
  --exp-name "${EXP_NAME}" \
  --pretrained "${PRETRAINED_CKPT}" \
  --seed 0 \
  --mixed-precision bf16 \
  --model DMFT-XL/2 \
  --attn-func fa3 \
  --resolution 256 \
  --dmf-depth 20 \
  --g-type mg \
  --omega 0.6 \
  --g-min 0.0 \
  --g-max 0.7 \
  --P-mean 0.0 \
  --P-mean-t 0.4 \
  --P-mean-r -1.2 \
  --max-train-steps 400000
