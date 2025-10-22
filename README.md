<h1 align="center"> Decoupled MeanFlow: Turning Flow Models into Flow Maps for Accelerated Sampling
</h1>

<!-- [![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.06940-b31b1b.svg)]()&nbsp; -->


<div align="center">
  <a href="https://kyungmnlee.github.io" target="_blank">Kyungmin&nbsp;Lee</a>&ensp; <b>&middot;</b> &ensp;
  <a href="https://sihyun.me/" target="_blank">Sihyun&nbsp;Yu</a>&ensp; <b>&middot;</b> &ensp;
  <a href="https://alinlab.kaist.ac.kr/shin.html" target="_blank">Jinwoo&nbsp;Shin</a>&ensp; <b>&middot;</b> &ensp;
  <br>
  KAIST &emsp;<br>
</div>
<!-- <h3 align="center">[<a href="https://sihyun.me/REPA">project page</a>]&emsp;[<a href="http://arxiv.org/abs/2410.06940">arXiv</a>]</h3>
<br> -->

<b>Summary</b>: Decoupled MeanFlow is a converts flow transformers into flow maps for accelerated sampling. Through fine-tuning the flow maps to predict the average velocity, we obtain state-of-the-art 1-step FID=2.16 and FID=2.12 on ImageNet 256 and 512 dataset, respectively. By increasing number of steps, we achieve FID=1.51 and FID=1.68 on ImageNet 256 and 512 dataset, matching the performance of flow models while significantly accelerating the inference. 

### 0. Download Pretrained models
We provide our DMF models, base SiT-XL/2+REPA models, and dataset statistics (for evaluation) in our Huggingface repository. One can download the model checkpoints and dataset statistics through following:
```bash
pip install huggingface_hub
hf download kyungmnlee/DMF --local-dir ckpt
```


### 1. Environment setup

When using Hopper GPUs (e.g., H100, H200, H800), we recommend to use CUDA==12.8 for compatibility with <a href="https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release">Flash Attention v3</a>. If using Ampere GPUs (e.g., A100), install <a href="https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features">Flash-Attention v2</a>.

```bash
conda create -n dmf python=3.12 -y
conda activate dmf
pip install torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt
```
To install Flash Attention v3, we recommend to install from source, following the official github repository of Flash Attention v3.

### 2. Dataset

#### Dataset download

Currently, we provide experiments for [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). You can place the data that you want and can specifiy it via `--data-dir` arguments in training scripts. Please refer to our [preprocessing guide](https://github.com/sihyun-yu/REPA/tree/master/preprocessing).

### 3. Training
We recommend fine-tuning from flow models, where we also provide model weights and training code below. Here, we provide a skeleton to launch the training. Please see [scripts/train_dmf_xl_2_256.sh](scripts/train_dmf_xl_2_256.sh) and [scripts/train_dmf_xl_2_512.sh](scripts/train_dmf_xl_2_512.sh) for details.
```bash
accelerate launch train_dmf.py \
  --model="DMFT-XL/2" \
  --attn-func "fa3" \
  --dmf-depth 20 \
  --g-type "mg" \
  --omega 0.6 \
  --g-min 0.0 \
  --g-max 0.7 \
  --P-mean 0.0 \
  --P-mean-t 0.4 \
  --P-mean-r -1.2
```

Then this script will automatically create the folder in `exps` to save logs and checkpoints. You can adjust the following options:

- `--models`: `[DMFT-B/2, DMFT-L/2, DMFT-XL/2]`
- `--attn-func`: `["fa3", "fa2", "base"]`. `fa3` for Flash-Attention v3 with JVP support, `fa2` for Flash-Attention v2 with JVP support, `base` for naive attention.
- `--g-type`: `["default", "mg", "distill"]`. `mg` stands for training with model guidance, `default` for non-cfg, and `distill` for cfg velocity from teacher flow model. Default choice is `mg`
- `--dmf-depth`: depth for DMF model (usually take 2/3 of total transformer depth)
- `--omega`: model guidance scale
- `--g-min`, `--g-max`: guidance interval for `mg`
- `--P-mean`, `--P-mean-t`, `--P-mean-r`: time proposal distributions
- `--qk-norm`: For 512 experiments, we find that adding qk norm during DMF training helps stabilizing the training (which is due to instable JVP computation).

### 4. Evaluation

One can generate images through the following code, see [scripts/eval_dmf_xl_2_256.sh](scripts/eval_dmf_xl_2_256.sh) and [scripts/eval_dmf_xl_2_512.sh](scripts/eval_dmf_xl_2_512.sh) for details:

```bash
torchrun --nnodes=1 --nproc_per_node=8 generate.py \
  --model "DMFT-XL/2" \
  --num-fid-samples 50000 \
  --ckpt YOUR_CHECKPOINT_PATH \
  --dmf-depth=20 \
  --per-proc-batch-size=64 \
  --mode "euler" \
  --num-steps 4 \
  --shift 1.0
```
- `--mode`: `["euler", "restart"]`
- `--shift`: time distribution shifting (use `shift=1.5` for ImageNet 512)

After generating images, one can compute FID and FD-DINOv2 using following code below. We provide dataset statistics `imgnet256.pkl` and `imgnet512.pkl` in our huggingface repository.
```bash
python calculate_metrics.py calc \
  --images "SAMPLE_DIR" \
  --ref="./ckpt/imgnet256.pkl"
```

### 5. Flow model training
We follow the setup in [REPA](https://github.com/sihyun-yu/REPA), where we refactorize the code for simplicity. The training scripts can be found in 
[scripts/train_sit_xl_2_256.sh](scripts/train_sit_xl_2_256.sh) and 
[scripts/train_sit_xl_2_512.sh](scripts/train_sit_xl_2_512.sh). We provide pretrained checkpoints in our huggingface repository

### Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to carry out sanity-check experiments in the near future.

## Acknowledgement

This code is mainly built upon [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), [edm2](https://github.com/NVlabs/edm2).

<!-- ## BibTeX

```bibtex
``` -->