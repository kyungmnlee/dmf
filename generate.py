import os
import math
import argparse
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.distributed as dist

from diffusers.models import AutoencoderKL
from models.dmft import DMFT_models
from models.sit import SiT_models
from samplers import euler_sampler, restart_sampler, ctm_sampler, fm_euler_sampler, fm_sde_sampler


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}. Must be one of ['fp16', 'bf16', 'fp32'].")

    # Load model:
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    latent_size = args.resolution // 8
    block_kwargs = {"attn_func": "torch_sdpa", "qk_norm": args.qk_norm}
    if args.model in DMFT_models:
        model = DMFT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            use_cfg_embedding=True,
            dmf_depth=args.dmf_depth,
            **block_kwargs
        )
    elif args.model in SiT_models:
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            use_cfg_embedding=True,
            learn_sigma=args.learn_sigma,
            **block_kwargs
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")
    # load checkpoint:
    state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"Model loaded with message: {msg}")
    model = model.to(device=device, dtype=dtype)
    model.eval()  # important!


    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    folder_name = (
        f"{model_string_name}-seed-{args.global_seed}-"
        f"mode-{args.mode}-steps-{args.num_steps}-shift-{args.shift}-"
        f"cfg-{args.cfg_scale}-gmin-{args.g_min}-gmax-{args.g_max}"
    )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device, dtype=dtype)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            g_min=args.g_min,
            g_max=args.g_max,
            shift=args.shift,
            heun=args.heun,
            gamma=args.gamma,
            coeff_type=args.coeff_type,
            path_type=args.path_type,
        )
        with torch.no_grad():
            if args.mode == "euler":
                if args.model in DMFT_models:
                    samples = euler_sampler(**sampling_kwargs).to(torch.float32)
                elif args.model in SiT_models:
                    samples = fm_euler_sampler(**sampling_kwargs).to(torch.float32)
                else:
                    raise NotImplementedError()
            elif args.mode == "restart":
                samples = restart_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ctm":
                samples = ctm_sampler(**sampling_kwargs, euler_steps=args.euler_steps).to(torch.float32)
            elif args.mode == "sde":
                assert args.model in SiT_models, "SDE sampler is only implemented for SiT models."
                samples = sit_sde_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            samples = samples.to(dtype=torch.float32)  # Convert to float32 for VAE decoding
            latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor([0.] * 4).view(1, 4, 1, 1).to(device)
            samples = vae.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
            ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        if args.create_npz:
            create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)
    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a model checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="./samples")
    # model
    parser.add_argument("--model", type=str, default="DMFT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dmf-depth", type=int, default=0)
    parser.add_argument("--create-npz", action=argparse.BooleanOptionalAction, default=False)

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="euler")
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--g-min", type=float, default=0.)
    parser.add_argument("--g-max", type=float, default=1.)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--coeff-type", type=str, default="sigma", choices=["sigma", "elbo"])
    parser.add_argument("--gamma", type=float, default=0.0, help="Only used for SDE samplers.")
    args = parser.parse_args()
    main(args)
