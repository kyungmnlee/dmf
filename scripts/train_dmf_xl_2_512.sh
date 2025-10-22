#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate dmf

MASTER_ADDR="MASTER_IP_ADDRESS"
MASTER_PORT="MASTER_PORT_NUMBER"
MACHINE_RANK=0  # Set to 0 for the first machine, 1 for the second machine

DATA_DIR="PATH/TO/IMAGENET512/"
OUTPUT_DIR="PATH/TO/OUTPUT/"
WANDB_DIR="PATH/TO/WANDB_LOG/"
WANDB_PROJ="WANDB_PROJECT_NAME"
PRETRAINED_CKPT="PATH/TO/PRETRAINED/CKPT"
EXP_NAME="dmf-xl-2-512"


accelerate launch \
  --num_machines 2 \
  --machine_rank "${MACHINE_RANK}" \
  --num_processes 16 \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
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
  --resolution 512 \
  --dmf-depth 20 \
  --g-type mg \
  --omega 0.6 \
  --g-min 0.0 \
  --g-max 0.8 \
  --P-mean 0.0 \
  --P-mean-t 0.4 \
  --P-mean-r -1.2 \
  --max-train-steps 800000 \
  --qk-norm
