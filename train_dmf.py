import argparse
import copy
import logging
import os
import json
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.dmft import DMFT_models
from models.sit import SiT_models
from loss import DMFLoss
from samplers import euler_sampler

from dataset import VAELabelDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid


torch.set_num_threads(4)
torch.set_num_interop_threads(1)
logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    """Sample from latent VAE"""
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float = 0.9999,
    only_trainable: bool = True,   # skip params with requires_grad=False
    update_buffers: bool = True    # keep buffers (e.g., running stats) in sync
):
    """
    Exponential moving average update.
    - Parameters with requires_grad=False are skipped when only_trainable=True.
    - Buffers are copied directly (no decay) if update_buffers=True.
    """
    # Map names consistently (DDP may prepend "module.")
    def _strip(name: str) -> str:
        for pre in ("module.", "orig_mod.", "_orig_mod."):
            if name.startswith(pre):
                return name[len(pre):]
        return name

    # Build quick-lookup sets & dicts
    trainable = {_strip(n) for n, p in model.named_parameters() if p.requires_grad}
    ema_params = { _strip(n): p for n, p in ema_model.named_parameters() }
    model_params = { _strip(n): p for n, p in model.named_parameters() }

    # EMA on parameters
    for name, p_model in model_params.items():
        if only_trainable and name not in trainable:
            continue
        p_ema = ema_params.get(name, None)
        if p_ema is None:
            continue
        # Guard: only EMA float tensors
        if p_ema.is_floating_point():
            p_ema.mul_(decay).add_(p_model.data, alpha=1.0 - decay)
        else:
            p_ema.copy_(p_model.data)

    # Keep buffers in sync (no EMA decay)
    if update_buffers:
        ema_buffers = { _strip(n): b for n, b in ema_model.named_buffers() }
        for name, b_model in model.named_buffers():
            name = _strip(name)
            b_ema = ema_buffers.get(name, None)
            if b_ema is not None:
                b_ema.copy_(b_model)


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    block_kwargs = {"attn_func": args.attn_func, "qk_norm": args.qk_norm}
    model = DMFT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg_embedding=(args.cfg_prob > 0.0),
        dmf_depth=args.dmf_depth,
        use_logvar=True,
        **block_kwargs
    )
    if args.pretrained:
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained model from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        msg = model.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            logger.info(msg)

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    if args.g_type == "distill":
        model_t = SiT_models["SiT-XL/2"](
            input_size=latent_size,
            num_classes=args.num_classes,
            use_cfg_embedding=True,
            **block_kwargs
        )
        msg = model_t.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            logger.info(msg)
        model_t.requires_grad_(False)
        model_t = model_t.to(device)
        model_t.eval()
    else:
        model_t = None
        
    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.0] * 4).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_config = dict(
        P_mean=args.P_mean,
        P_std=args.P_std,
        P_mean_t=args.P_mean_t,
        P_std_t=args.P_std_t,
        P_mean_r=args.P_mean_r,
        P_std_r=args.P_std_r,
        cfg_prob=args.cfg_prob,
        omega=args.omega,
        g_min=args.g_min,
        g_max=args.g_max,
        path_type=args.path_type,
        g_type=args.g_type,
    )
    loss_fn = DMFLoss(**loss_config)

    if accelerator.is_main_process:
        logger.info(f"Training Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Setup optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        fused=True,
    )

    # Setup data:
    train_dataset = VAELabelDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0.0, only_trainable=False, update_buffers=True)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False
        )
        msg = model.load_state_dict(ckpt['model'])
        if accelerator.is_main_process:
            logger.info(msg)
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        if accelerator.is_main_process:
            logger.info(f"Resume from {global_step}")
    
    compiled_model = torch.compile(
        model,
        backend=getattr(args, "compile_backend", "inductor"),
        mode=getattr(args, "compile_mode", "default"),
        fullgraph=getattr(args, "compile_fullgraph", False),
        dynamic=getattr(args, "compile_dynamic", True),
    )
    if model_t is not None:
        model_t = torch.compile(
            model_t,
            backend=getattr(args, "compile_backend", "inductor"),
            mode=getattr(args, "compile_mode", "default"),
            fullgraph=getattr(args, "compile_fullgraph", False),
            dynamic=getattr(args, "compile_dynamic", True),
        )

    compiled_model, optimizer, train_dataloader = accelerator.prepare(
        compiled_model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name=args.project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}", "dir": f"{args.wandb_dir}"}},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Create sampling noise:
    sample_batch_size = 64 // accelerator.num_processes
    ys = torch.randint(0, 1000, (sample_batch_size,), device=device)
    x1 = torch.randn((sample_batch_size, 4, latent_size, latent_size), device=device)
    sampling_kwargs = dict(
        num_steps=4,
        cfg_scale=1.0,
        g_min=0.0,
        g_max=1.0,
        shift=1.0,
        path_type=args.path_type,
    )

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dataloader:
            x = x.squeeze(dim=1).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
            with accelerator.autocast():
                loss, loss_dict = loss_fn(compiled_model, model, x, y, model_t=model_t)
            loss = loss.mean()

            ## optimization
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                update_ema(ema, model, decay=args.ema_decay, only_trainable=True, update_buffers=True)

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 100 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                with torch.no_grad():
                    with accelerator.autocast():
                        samples = euler_sampler(ema, x1, ys, **sampling_kwargs).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                accelerator.log({
                    "samples": wandb.Image(array2grid(out_samples)),
                })
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            for key, value in loss_dict.items():
                logs[key] = accelerator.gather(value).mean().detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="./exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--project-name", type=str, required=True)
    parser.add_argument("--wandb-dir", type=str, default="./wandb/")

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--dmf-depth", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    
    # torch compile
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile-backend", type=str, default="inductor")
    parser.add_argument("--compile-fullgraph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=True)

    # Fine-tuning
    parser.add_argument("--pretrained", type=str, default=None)

    #training loss
    parser.add_argument("--g-type", type=str, default="default", choices=["default", "mg", "distill"])
    parser.add_argument("--P-mean", type=float, default=0.0, help="Mean of the P distribution.")
    parser.add_argument("--P-std", type=float, default=1.0, help="Standard deviation of the P distribution.")
    parser.add_argument("--P-mean-t", type=float, default=0.4, help="Mean of the P_t distribution.")
    parser.add_argument("--P-std-t", type=float, default=1.0, help="Standard deviation of the P_t distribution.")
    parser.add_argument("--P-mean-r", type=float, default=-1.2, help="Mean of the P_r distribution.")
    parser.add_argument("--P-std-r", type=float, default=1.0, help="Standard deviation of the P_r distribution.")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--attn-func", type=str, default="base", choices=["base", "torch_sdpa", "fa2", "fa3"])
    parser.add_argument("--omega", type=float, default=0.5, help="Scale for the mean flow loss.")
    parser.add_argument("--g-min", type=float, default=0.0, help="Minimum value for the cfg index.")
    parser.add_argument("--g-max", type=float, default=1.0, help="Maximum value for the cfg index.")

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    ## adam-beta2 is set to 0.95 in the original DiT paper
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="Exponential moving average decay.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
