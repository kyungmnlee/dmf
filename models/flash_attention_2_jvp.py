"""
Custom implementation of Flash Attention v2 with JVP support.
Based on the official implementation:
https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py

Basically, we use same forward and backward from flash attention v2, but use custom
triton kernel for JVP computation. See `kernel_utils/triton_utils.py` for details.
"""

import pytest

from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

import flash_attn_2_cuda 

from models.triton_utils import _attn_fwd_dual, _attn_fwd_dual_split, split_head_dim


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


@torch.library.custom_op("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_attn_2_cuda.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,
    )
    return out, softmax_lse, S_dmask, rng_state

@torch.library.register_fake("flash_attn::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size, seqlen_q, num_heads, head_size = q.shape
    seqlen_k = k.shape[1]
    out = torch.empty_like(q)
    softmax_lse = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        p = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128)), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

    return out, softmax_lse, p, rng_state


@torch.library.custom_op("flash_attn::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_2_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    return softmax_d


@torch.library.register_fake("flash_attn::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    batch_size, seqlen_q, num_heads, _ = q.shape
    softmax_d = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128)), device=q.device, dtype=torch.float32)

    return softmax_d


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(q, k, v, tq, tk, tv, softmax_scale):
        ## q, k, v: (B, L, H, D)
        if softmax_scale is None:
            softmax_scale = (q.shape[-1]) ** (-0.5)
            
        enable_jvp = (tq is not None) and (tk is not None) and (tv is not None)
        if enable_jvp:
            block_q, block_kv, num_stages, num_warps = 128, 128, 2, 8  # can be tuned
            b, l_q, h, d = q.shape
            l_kv = k.shape[1]
            tile_dim, tail_dim = split_head_dim(d)
            has_tail = True if tail_dim > 0 else False
            o, to = torch.empty_like(q), torch.empty_like(tq)
            grid = (l_q // block_q, b * h, 1)
            m = torch.empty((b, h, l_q), device=q.device, dtype=torch.float32)
            if has_tail:
                _attn_fwd_dual_split[grid](
                    q, k, v,  #
                    tq, tk, tv,  #
                    m, o, to,  #
                    softmax_scale,  #
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                    tq.stride(0), tq.stride(1), tq.stride(2), tq.stride(3),  #
                    tk.stride(0), tk.stride(1), tk.stride(2), tk.stride(3),  #
                    tv.stride(0), tv.stride(1), tv.stride(2), tv.stride(3),  #
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                    to.stride(0), to.stride(1), to.stride(2), to.stride(3),
                    h, l_q, l_kv, d, tile_dim, tail_dim,  #
                    block_q=block_q, block_kv=block_kv,  #
                    num_stages=num_stages, num_warps=num_warps,  #
                )
            else:
                _attn_fwd_dual[grid](
                    q, k, v,  #
                    tq, tk, tv,  #
                    m, o, to,  #
                    softmax_scale,  #
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                    tq.stride(0), tq.stride(1), tq.stride(2), tq.stride(3),
                    tk.stride(0), tk.stride(1), tk.stride(2), tk.stride(3),
                    tv.stride(0), tv.stride(1), tv.stride(2), tv.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                    to.stride(0), to.stride(1), to.stride(2), to.stride(3),  #
                    h, l_q, l_kv, d,  #
                    block_q=block_q, block_kv=block_kv,  #
                    num_stages=num_stages, num_warps=num_warps,  #
                )
        else:
            tq, tk, tv, to = None, None, None, None
            o, m, *rest = _flash_attn_forward(
                q, k, v,
                0.0,  # no dropout
                softmax_scale,
                causal=False,  # non causal
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
        return o, m, to

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        (
            q, k, v, tq, tk, tv, softmax_scale
        ) = inputs
        (o, m, to) = outputs
        if softmax_scale is None:
            softmax_scale = (q.shape[-1]) ** (-0.5)
        ctx.save_for_backward(q, k, v, o, m)
        if tq is not None and tk is not None and tv is not None:
            ctx.save_for_forward(to)
        ctx.softmax_scale = softmax_scale

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, o, m = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q, k, v, o, m, #
            dq, dk, dv,  #
            0.0,  # dropout = 0.0
            ctx.softmax_scale,
            False,  # causal
            -1,  # window_size_left
            -1,  # window_size_right
            0.0, # softcap
            None,  #
            False,  # deterministic
        )
        return dq, dk, dv, None, None, None, None
    
    @staticmethod
    def jvp(ctx, *args):
        t_saved = getattr(ctx, "saved_forward", ())
        to = t_saved[0] if len(t_saved) > 0 else None
        return to, None, None
        

def _is_dual_mode(*tensors):
    """Helper function to check if any tensor is in dual mode."""
    for tensor in tensors:
        _, tangent = fwAD.unpack_dual(tensor)
        if tangent is not None:
            return True
    return False


def flash_attn_func(q, k, v, softmax_scale=None):
    dual = _is_dual_mode(q, k, v)
    if dual:
        tensors = [q, k, v]
        primals = []
        tangents = []
        for tensor in tensors:
            primal, tangent = fwAD.unpack_dual(tensor)
            primals.append(primal)
            if tangent is None:
                tangent = torch.zeros_like(primal)
            tangents.append(tangent)
        q_primal, k_primal, v_primal = primals
        q_tangent, k_tangent, v_tangent = tangents
        out, _, t_out = FlashAttnFunc.apply(
            q_primal, k_primal, v_primal,  # primal
            q_tangent, k_tangent, v_tangent,  # tangent
            softmax_scale
        )
        out_dual = fwAD.make_dual(out, t_out)
        return out_dual
    else:
        out, _, _ = FlashAttnFunc.apply(
            q, k, v, None, None, None, softmax_scale
        )
        return out


def raw_attn_func(q, k, v):
    """A simple attention implementation for testing purposes.
        q: (b, l, h, d)
        k: (b, l, h, d)
        v: (b, l, h, d)
        Returns: (b, l, h, d)
    """
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    attn_probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_probs, v)
    out = out.permute(0, 2, 1, 3)
    return out


@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("l", [256, 1024, 2048])
@pytest.mark.parametrize("h", [16, 24, 32])
@pytest.mark.parametrize("d", [64, 72, 96, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_jvp(b, l, h, d, dtype):
    rtol = 0.0
    atol = 1.2e-3
    torch.cuda.empty_cache()
    q, k, v, dq, dk, dv, do = (
        torch.empty(
            (b, l, h, d), dtype=dtype, device="cuda"
        ).normal_(mean=0.0, std=0.5).requires_grad_()
        for _ in range(7)
    )
    u_ref, du_ref = torch.func.jvp(
        raw_attn_func, (q, k, v), (dq, dk, dv)
    )
    u_tri, du_tri = torch.func.jvp(
        flash_attn_func, (q, k, v), (dq, dk, dv)
    )
    
    u_ref.backward(do)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    
    u_tri.backward(do)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    
    # compare
    torch.testing.assert_close(u_ref, u_tri, atol=atol, rtol=rtol)
    torch.testing.assert_close(du_ref, du_tri, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--disable-warnings"])

