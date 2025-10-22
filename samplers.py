import math
from typing import Callable, Optional, List, Union
import numpy as np

import torch


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
        t: [batch_dim,], time vector
        x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError
    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var
    return score


def compute_diffusion(t_cur, coeff_type="sigma", path_type="linear", gamma=1.0):
    """Compute diffusion coefficient based on time t_cur"""
    if path_type == "linear":
        if coeff_type == "sigma":
            return 2 * gamma * t_cur
        elif coeff_type == "elbo":
            return 2 * gamma * t_cur / (1 - t_cur)
    elif path_type == "cosine":
        if coeff_type == "sigma":
            return 2 * gamma * torch.sin(t_cur * np.pi / 2)
        elif coeff_type == "elbo":
            return np.pi * gamma * torch.tan(t_cur * np.pi / 2)
    else:
        raise NotImplementedError(f"Path type {path_type} not implemented.")


def get_tsteps(
    num_steps: int, t_start: float = 1.0, t_end: float = 0.0, shift: float = 1.0
):
    t_steps = torch.linspace(t_start, t_end, num_steps + 1, dtype=torch.float64)
    if shift == 1.0:
        return t_steps
    return shift * t_steps / (1 + (shift - 1.0) * t_steps)


@torch.no_grad()
def euler_sampler(
    model,
    latents,
    y,
    num_steps: int   = 4,
    cfg_scale: float = 1.0,
    g_min:     float = 0.0,
    g_max:     float = 1.0,
    shift:     float = 1.0,
    **kwargs
):
    _dtype, _device = latents.dtype, latents.device
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=_device)
    t_steps = get_tsteps(num_steps=num_steps, shift=shift)
    x_nxt = latents.to(torch.float64)
    for i, (t_cur, t_nxt) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, y_cur = x_nxt, y
        model_input = x_cur
        if cfg_scale > 1.0 and t_cur <= g_max and t_cur >= g_min:
            model_input = torch.cat([model_input] * 2, dim=0)
            y_cur = torch.cat([y_cur, y_null], dim=0)
        model_input = model_input.to(dtype=_dtype)
        t_input = t_cur * torch.ones(model_input.shape[0]).to(device=_device, dtype=_dtype)
        r_input = t_nxt * torch.ones(model_input.shape[0]).to(device=_device, dtype=_dtype)
        d_cur = model(model_input, t_input, r_input, y_cur).to(torch.float64)
        if cfg_scale > 1. and t_cur <= g_max and t_cur >= g_min:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
        x_nxt = x_cur + (t_nxt - t_cur) * d_cur
    return x_nxt


@torch.no_grad()
def restart_sampler(
    model,
    latents,
    y,
    num_steps:  int   = 4,
    cfg_scale:  float = 1.0,
    g_min:      float = 0.0,
    g_max:      float = 1.0,
    shift:      float = 1.0,
    path_type:  str   = "linear",
    **kwargs
):
    _dtype, _device = latents.dtype, latents.device
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=_device)
    t_steps = get_tsteps(num_steps=num_steps, shift=shift)
    x_nxt = latents.to(torch.float64)
    for i, (t_cur, t_nxt) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, y_cur = x_nxt, y
        model_input = x_cur
        if cfg_scale > 1.0 and t_cur <= g_max and t_cur >= g_min:
            model_input = torch.cat([model_input] * 2, dim=0)
            y_cur = torch.cat([y_cur, y_null], dim=0)
        model_input = model_input.to(dtype=_dtype)
        t_input = t_cur * torch.ones(model_input.shape[0]).to(device=_device, dtype=_dtype)
        r_input = torch.zeros(model_input.shape[0]).to(device=_device, dtype=_dtype)
        d_cur = model(model_input, t_input, r_input, y_cur).to(torch.float64)
        if cfg_scale > 1. and t_cur <= g_max and t_cur >= g_min:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
        x_nxt = x_cur - t_cur * d_cur
        if i < num_steps - 1:
            noise = torch.randn_like(x_nxt).to(device=_device)
            if path_type == "linear":
                x_nxt = (1 - t_nxt) * x_nxt + t_nxt * noise
            elif path_type == "cosine":
                x_nxt = torch.cos(t_nxt * np.pi / 2) * x_nxt + torch.sin(t_nxt * np.pi / 2) * noise
            else:
                raise NotImplementedError(f"Path type {path_type} not implemented.")
    return x_nxt


@torch.no_grad()
def ctm_sampler(
    model,
    latents,
    y,
    num_steps:  int   = 4,
    cfg_scale:  float = 1.0,
    g_min:      float = 0.0,
    g_max:      float = 1.0,
    shift:      float = 1.0,
    gamma:      float = 0.9,
    **kwargs
):
    _dtype, _device = latents.dtype, latents.device
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=_device)
    t_steps = get_tsteps(num_steps=num_steps, shift=shift)
    x_nxt = latents.to(torch.float64)
    for i, (t_cur, t_nxt) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, y_cur = x_nxt, y
        model_input = x_cur
        if cfg_scale > 1.0 and t_cur <= g_max and t_cur >= g_min:
            model_input = torch.cat([model_input] * 2, dim=0)
            y_cur = torch.cat([y_cur, y_null], dim=0)
        model_input = model_input.to(dtype=_dtype)
        t_input = t_cur * torch.ones(model_input.shape[0]).to(device=_device, dtype=_dtype)
        if i < num_steps - 1:
            t_nxt_gamma = t_nxt * np.sqrt(1 - gamma**2)
            alpha_shift = (1 - t_nxt) / (1 - t_nxt_gamma)
            sigma_shift = 1 - alpha_shift
        else:
            t_nxt_gamma = t_nxt
        r_input = t_nxt_gamma * torch.ones(model_input.shape[0]).to(device=_device, dtype=_dtype)

        d_cur = model(model_input, t_input, r_input, y_cur).to(torch.float64)
        if cfg_scale > 1. and t_cur <= g_max and t_cur >= g_min:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
        x_nxt = x_cur + (t_nxt_gamma - t_cur) * d_cur
        if i < num_steps - 1:
            x_nxt = alpha_shift * x_nxt + sigma_shift * torch.randn_like(x_nxt).to(device=_device)

    return x_nxt


"""Euler sampler for flow models"""
@torch.no_grad()
def fm_euler_sampler(
    model,
    latents,
    y,
    num_steps: int   = 4,
    cfg_scale: float = 1.0,
    g_min:     float = 0.0,
    g_max:     float = 1.0,
    shift:     float = 1.0,
    heun:      bool = False,
    **kwargs
):
    _dtype, _device = latents.dtype, latents.device
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=_device)
    t_steps = get_tsteps(num_steps=num_steps, shift=shift)
    x_nxt = latents.to(torch.float64)
    for i, (t_cur, t_nxt) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, y_cur = x_nxt, y
        model_input = x_cur
        if cfg_scale > 1.0 and t_cur <= g_max and t_cur >= g_min:
            model_input = torch.cat([model_input] * 2, dim=0)
            y_cur = torch.cat([y_cur, y_null], dim=0)
        model_input = model_input.to(dtype=_dtype)
        t_input = t_cur * torch.ones(model_input.shape[0]).to(device=_device, dtype=_dtype)
        d_cur = model(model_input, t_input, y_cur).to(torch.float64)
        if cfg_scale > 1. and t_cur <= g_max and t_cur >= g_min:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            x_nxt = x_cur + (t_nxt - t_cur) * d_cur
        if heun and (i < num_steps - 1):
            model_input = x_nxt
            y_cur = y
            if cfg_scale > 1.0 and t_cur <= g_max and t_cur >= g_min:
                model_input = torch.cat([x_nxt] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            t_input = t_nxt * torch.ones(model_input.shape[0]).to(device=_device, dtype=_dtype)
            d_prime = model(model_input.to(dtype=_dtype), t_input, y_cur).to(torch.float64)
            if cfg_scale > 1.0 and t_cur <= g_max and t_cur >= g_min:
                d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                x_nxt = x_cur + (t_nxt - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_nxt


"""Euler-Maruyama sampler for flow models"""
@torch.no_grad()
def fm_sde_sampler(
    model,
    latents,
    y,
    num_steps: int = 250,
    cfg_scale: float = 1.0,
    g_min: float = 0.0,
    g_max: float = 1.0,
    shift: float = 1.0,
    path_type: str = "linear",
    coeff_type: str = "sigma",
    gamma: float = 1.0,
    **kwargs
):
    _dtype, _device = latents.dtype, latents.device
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=_device)
    t_steps = get_tsteps(num_steps=num_steps - 1, t_start=1.0, t_end=1e-4, shift=shift)
    t_steps = torch.cat([t_steps, torch.tensor([0.0], device=t_steps.device, dtype=torch.float64)], dim=0)
    x_nxt = latents.to(torch.float64)
    for i, (t_cur, t_nxt) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, y_cur = x_nxt, y
        model_input = x_cur
        if cfg_scale > 1.0 and t_cur <= g_max and t_cur >= g_min:
            model_input = torch.cat([model_input] * 2, dim=0)
            y_cur = torch.cat([y_cur, y_null], dim=0)
        model_input = model_input.to(dtype=_dtype)
        t_input = t_cur * torch.ones(model_input.shape[0]).to(device=_device, dtype=_dtype)
        w_t = compute_diffusion(t_cur, coeff_type=coeff_type, path_type=path_type, gamma=gamma)
        d_eps = torch.sqrt(t_cur - t_nxt) * torch.randn_like(x_cur).to(device=_device, dtype=_dtype)
        v_cur = model(model_input, t_input, y_cur).to(torch.float32)
        s_cur = get_score_from_velocity(v_cur, model_input, t_input, path_type=path_type)
        d_cur = v_cur - 0.5 * w_t * s_cur
        if cfg_scale > 1. and t_cur <= g_max and t_cur >= g_min:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
        x_nxt = x_cur + (t_nxt - t_cur) * d_cur
        if i < num_steps - 1:
            x_nxt = x_nxt + torch.sqrt(w_t) * d_eps
    
    return x_nxt
