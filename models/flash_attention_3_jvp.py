"""
Custom implementation of Flash Attention v3 with JVP support.
Based on the official implementation:
https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py

Basically, we use same forward and backward from flash attention v3, but use custom
triton kernel for JVP computation. See `kernel_utils/triton_utils.py` for details.
"""

import pytest

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

import flash_attn_3._C  # Flash Attention 3 CUDA
flash_attn_3_cuda = torch.ops.flash_attn_3

from models.triton_utils import _attn_fwd_dual, _attn_fwd_dual_split, split_head_dim


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
    ):
    q, k, k_new, v_new = [maybe_contiguous(x) for x in (q, k, k_new, v_new)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new)
    ]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table, kv_batch_idx, leftpad_k = [
        maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    seqlens_rotary = maybe_contiguous(seqlens_rotary)
    out, softmax_lse, *rest = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return out, softmax_lse, *rest


def _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
):
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, *rest = flash_attn_3_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        sm_margin,
    )
    return dq, dk, dv, softmax_d


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    # @torch.amp.custom_fwd(device_type="cuda")
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
                q, k, v,  #
                None, None,  # k_new, v_new
                None,  # qv
                None,  # out
                None, None, None,   # cu_seqlens_q/k/k_new
                None, None,   # seqused_q/k
                None, None,   # max_seqlen_q/k
                None, None, None,   # page_table, kv_batch_idx, leftpad_k,
                None, None, None,  # rotary_cos/sin, seqlens_rotary
                None, None, None,
                softmax_scale,
                causal=False,
                window_size=(-1, -1),
                attention_chunk=0,
                softcap=0.0,
                num_splits=1,
                pack_gqa=None,
                sm_margin=0,
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
    # @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout, *args):
        q, k, v, o, m = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q, k, v, o, m,  #
            None, None, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            None, None, # max_seqlen_q, max_seqlen_k,
            dq, dk, dv,  #
            ctx.softmax_scale,
            False,  # causal = False
            (-1, -1),  # window_size = (-1, -1)
            0.0,  # softcap = 0.0
            False,  # deterministic = False
            0,  # sm_margin = 0
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
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

