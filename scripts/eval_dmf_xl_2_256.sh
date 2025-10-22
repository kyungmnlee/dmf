#!/bin/sh
set -e
# Activate the conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate dmf

CKPT_DIR="./ckpt/dmf_xl_2_256.pt"
SAMPLE_DIR="./samples/"

for ns in 1 2 4; do
  torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    --master-port 29502 \
    generate.py \
    --ckpt "${CKPT_DIR}" \
    --sample-dir "${SAMPLE_DIR}" \
    --num-fid-samples 50000 \
    --per-proc-batch-size 32 \
    --model "DMFT-XL/2" \
    --dmf-depth 20 \
    --resolution 256 \
    --mode "euler" \
    --num-steps "${ns}" \
    --global-seed 3
done

cd preprocessing/
for ns in 1 2 4; do
  python calculate_metrics.py calc \
  --images="${SAMPLE_DIR}/DMFT-XL-2-seed-3-mode-euler-steps-${ns}-shift-1.0-cfg-1.0-gmin-0.0-gmax-1.0" \
  --ref="./ckpt/imgnet256.pkl" 
done