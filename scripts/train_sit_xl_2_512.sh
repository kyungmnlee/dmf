#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate dmf

DATA_DIR="PATH/TO/IMAGENET512/"
OUTPUT_DIR="PATH/TO/OUTPUT/"
WANDB_DIR="PATH/TO/WANDB_LOG/"
WANDB_PROJ="WANDB_PROJECT_NAME"
EXP_NAME="sit-xl-2-512-repa"

accelerate launch \
  --num_machines 1 \
  --multi_gpu \
  --num_processes 8 \
  train_fm.py \
  --torch-compile \
  --mixed-precision "bf16" \
  --project-name "${WANDB_PROJ}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --wandb-dir "${WANDB_DIR}" \
  --exp-name "${EXP_NAME}" \
  --model "SiT-XL/2" \
  --attn-func "fa3" \
  --resolution 512 \
  --seed 0 \
  --P-mean 0.0 \
  --max-train-steps 2000000 \
  --proj-coeff 0.5  # proj-coeff > 0 to enable REPA, otherwise set proj-coeff = 0 for standard training
