"""triton implementation of flash attention forward with jvp support
Source:
    - Official Triton implementation: (https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
    - JVP support for FlashAttention
        - From Ryu1845 (https://gist.github.com/Ryu1845/904a9b2c1fd58ca0b5d14119a5452eea)
        - From Birch-san (https://gist.github.com/Birch-san/c51234fe006cf1ffc680063abb4f572f)
Note:
    - This implementation only supports non-causal attention.
    - Supports jvp computation using torch.func.jvp
    - For 128 > head_dim > 64, split the head dim into two parts (tile and tail) for efficiency.
        - E.g. DiT-XL/2 uses head_dim = 72 -> split into 64 + 16 (last 8 is not computed)
"""

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

import triton
import triton.language as tl


def split_head_dim(head_dim: int):
    """
    Split head_dim into (main, tail) tile sizes for block loading.
    The main part is the largest power-of-two <= head_dim.
    The tail is the next power-of-two (up to 64) that covers the remainder.

    Examples:
        72 -> (64, 16)
        96 -> (64, 32)
        80 -> (64, 16)
        128 -> (128, 0)
    """
    # Largest power of two <= head_dim
    main = 1 << (head_dim.bit_length() - 1)
    if main > 64:
        main = 64  # we usually cap main tile to 64 for flash-attn kernels

    remainder = head_dim - main
    if remainder <= 0:
        return main, 0
    elif remainder <= 16:
        return main, 16
    elif remainder <= 32:
        return main, 32
    else:
        return 128, 0


# inner kernel for attention forward with jvp
@triton.jit
def _attn_fwd_dual_inner(
    o_block, to_block,  #
    l_i, m_i, mu_i,  #
    q_block, tq_block,  #
    k_block_ptr, tk_block_ptr, v_block_ptr, tv_block_ptr,  #
    softmax_scale,  #
    seq_len_kv: tl.constexpr,  #
    block_q: tl.constexpr,
    block_kv: tl.constexpr,
):
    k_block_ptr = tl.advance(k_block_ptr, (0, 0))
    v_block_ptr = tl.advance(v_block_ptr, (0, 0))
    k_curr = tl.load(k_block_ptr)
    v_curr = tl.load(v_block_ptr)
    tk_block_ptr = tl.advance(tk_block_ptr, (0, 0))
    tv_block_ptr = tl.advance(tv_block_ptr, (0, 0))
    tk_curr = tl.load(tk_block_ptr)
    tv_curr = tl.load(tv_block_ptr)

    # loop over k, v and update accumulator
    for start_kv in range(0, seq_len_kv, block_kv):
        start_kv = tl.multiple_of(start_kv, block_kv)
        # load K, V, TK, TV
        k_ptr_next = k_block_ptr; v_ptr_next = v_block_ptr
        k_next = k_curr; v_next = v_curr
        tk_ptr_next = tk_block_ptr; tv_ptr_next = tv_block_ptr
        tk_next = tk_curr; tv_next = tv_curr

        # prefetch next block
        has_next = start_kv + block_kv < seq_len_kv
        if has_next:
            k_ptr_next = tl.advance(k_block_ptr, (0, block_kv))
            v_ptr_next = tl.advance(v_block_ptr, (block_kv, 0))
            k_next = tl.load(k_ptr_next)
            v_next = tl.load(v_ptr_next)
            tk_ptr_next = tl.advance(tk_block_ptr, (0, block_kv))
            tv_ptr_next = tl.advance(tv_block_ptr, (block_kv, 0))
            tk_next = tl.load(tk_ptr_next)
            tv_next = tl.load(tv_ptr_next)
        # compute scores
        qk_block = softmax_scale * tl.dot(q_block, k_curr)
        m_ij = tl.maximum(m_i, tl.max(qk_block, 1))
        qk_block = qk_block - m_ij[:, None]
        p_block = tl.math.exp(qk_block)
        l_ij = tl.sum(p_block, 1)
        alpha = tl.math.exp(m_i - m_ij)
        tqk_block = softmax_scale * (
            tl.dot(tq_block, k_curr) + tl.dot(q_block, tk_curr)
        )
        p_tqk_block = p_block * tqk_block
        # compute online stats
        l_new = l_i * alpha + l_ij
        inv_l_new = 1.0 / l_new
        scale_prev = (alpha * l_i) * inv_l_new
        # compute output
        s_v = tl.dot(p_block.to(dtype=v_curr.dtype), v_curr)
        o_block_new = scale_prev[:, None] * o_block + (s_v * inv_l_new[:, None])
        # compute jvp output
        s_gv = tl.dot(p_tqk_block.to(dtype=v_curr.dtype), v_curr)
        s_tv = tl.dot(p_block.to(dtype=tv_curr.dtype), tv_curr)
        s_mu = tl.sum(p_tqk_block, 1)
        mu_new = scale_prev * mu_i + s_mu * inv_l_new
        to_block = (
            scale_prev[:, None] * to_block
            + (scale_prev * mu_i)[:, None] * (o_block - o_block_new)
            + (s_gv + s_tv - s_mu[:, None] * o_block_new) * inv_l_new[:, None]
        )
        # update stats
        mu_i = mu_new; l_i = l_new; m_i = m_ij
        o_block = o_block_new
        # move to next block
        k_curr = k_next; v_curr = v_next
        k_block_ptr = k_ptr_next; v_block_ptr = v_ptr_next
        tk_curr = tk_next; tv_curr = tv_next
        tk_block_ptr = tk_ptr_next; tv_block_ptr = tv_ptr_next
    return o_block, to_block, l_i, m_i, mu_i


# inner kernel for attention forward with jvp (split head dim)
@triton.jit
def _attn_fwd_dual_inner_split(
    o0_block, o1_block,  #
    to0_block, to1_block,  #
    l_i, m_i, mu_i,  #
    q0_block, q1_block,  #
    tq0_block, tq1_block,  #
    k0_block_ptr, k1_block_ptr,  #
    tk0_block_ptr, tk1_block_ptr,  #
    v0_block_ptr, v1_block_ptr,  #
    tv0_block_ptr, tv1_block_ptr,  #
    softmax_scale,  #
    seq_len_kv: tl.constexpr,  #
    block_q: tl.constexpr,
    block_kv: tl.constexpr,
):
    k0_block_ptr = tl.advance(k0_block_ptr, (0, 0))
    k1_block_ptr = tl.advance(k1_block_ptr, (0, 0))
    v0_block_ptr = tl.advance(v0_block_ptr, (0, 0))
    v1_block_ptr = tl.advance(v1_block_ptr, (0, 0))
    k0_curr = tl.load(k0_block_ptr)
    k1_curr = tl.load(k1_block_ptr, boundary_check=(0,), padding_option="zero")
    v0_curr = tl.load(v0_block_ptr)
    v1_curr = tl.load(v1_block_ptr, boundary_check=(1,), padding_option="zero")
    tk0_block_ptr = tl.advance(tk0_block_ptr, (0, 0))
    tv0_block_ptr = tl.advance(tv0_block_ptr, (0, 0))
    tk1_block_ptr = tl.advance(tk1_block_ptr, (0, 0))
    tv1_block_ptr = tl.advance(tv1_block_ptr, (0, 0))
    tk0_curr = tl.load(tk0_block_ptr)
    tk1_curr = tl.load(tk1_block_ptr, boundary_check=(0,), padding_option="zero")
    tv0_curr = tl.load(tv0_block_ptr)
    tv1_curr = tl.load(tv1_block_ptr, boundary_check=(1,), padding_option="zero")

    # loop over k, v and update accumulator
    for start_kv in range(0, seq_len_kv, block_kv):
        start_kv = tl.multiple_of(start_kv, block_kv)
        # load k, v, tk, tv
        k0_ptr_next = k0_block_ptr; v0_ptr_next = v0_block_ptr
        k1_ptr_next = k1_block_ptr; v1_ptr_next = v1_block_ptr
        k0_next = k0_curr; v0_next = v0_curr
        k1_next = k1_curr; v1_next = v1_curr
        tk0_ptr_next = tk0_block_ptr; tv0_ptr_next = tv0_block_ptr
        tk1_ptr_next = tk1_block_ptr; tv1_ptr_next = tv1_block_ptr
        tk0_next = tk0_curr; tv0_next = tv0_curr
        tk1_next = tk1_curr; tv1_next = tv1_curr
        # prefetch next block
        has_next = start_kv + block_kv < seq_len_kv
        if has_next:
            k0_ptr_next = tl.advance(k0_ptr_next, (0, block_kv))
            v0_ptr_next = tl.advance(v0_ptr_next, (block_kv, 0))
            k1_ptr_next = tl.advance(k1_ptr_next, (0, block_kv))
            v1_ptr_next = tl.advance(v1_ptr_next, (block_kv, 0))
            k0_next = tl.load(k0_ptr_next)
            v0_next = tl.load(v0_ptr_next)
            k1_next = tl.load(k1_ptr_next, boundary_check=(0,), padding_option="zero")
            v1_next = tl.load(v1_ptr_next, boundary_check=(1,), padding_option="zero")
            tk0_ptr_next = tl.advance(tk0_ptr_next, (0, block_kv))
            tv0_ptr_next = tl.advance(tv0_ptr_next, (block_kv, 0))
            tk1_ptr_next = tl.advance(tk1_ptr_next, (0, block_kv))
            tv1_ptr_next = tl.advance(tv1_ptr_next, (block_kv, 0))
            tk0_next = tl.load(tk0_ptr_next)
            tv0_next = tl.load(tv0_ptr_next)
            tk1_next = tl.load(tk1_ptr_next, boundary_check=(0,), padding_option="zero")
            tv1_next = tl.load(tv1_ptr_next, boundary_check=(1,), padding_option="zero")
        # compute scores
        qk_block = softmax_scale * (
            tl.dot(q0_block, k0_curr) + tl.dot(q1_block, k1_curr)
        )
        m_ij = tl.maximum(m_i, tl.max(qk_block, 1))
        qk_block = qk_block - m_ij[:, None]
        p_block = tl.math.exp(qk_block)
        tqk_block = softmax_scale * (
            tl.dot(tq0_block, k0_curr) + tl.dot(q0_block, tk0_curr) +
            tl.dot(tq1_block, k1_curr) + tl.dot(q1_block, tk1_curr)
        )
        p_tqk_block = p_block * tqk_block
        # compute online stats
        l_ij = tl.sum(p_block, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_new = l_i * alpha + l_ij
        inv_l_new = 1.0 / l_new
        scale_prev = (alpha * l_i) * inv_l_new
        # compute output
        s_v0 = tl.dot(p_block.to(dtype=v0_curr.dtype), v0_curr)
        o0_block_new = scale_prev[:, None] * o0_block + (s_v0 * inv_l_new[:, None])
        s_v1 = tl.dot(p_block.to(dtype=v1_curr.dtype), v1_curr)
        o1_block_new = scale_prev[:, None] * o1_block + (s_v1 * inv_l_new[:, None])
        # compute jvp output
        s_gv0 = tl.dot(p_tqk_block.to(dtype=v0_curr.dtype), v0_curr)
        s_gv1 = tl.dot(p_tqk_block.to(dtype=v1_curr.dtype), v1_curr)
        s_tv0 = tl.dot(p_block.to(dtype=tv0_curr.dtype), tv0_curr)
        s_tv1 = tl.dot(p_block.to(dtype=tv1_curr.dtype), tv1_curr)
        s_mu = tl.sum(p_tqk_block, 1)
        mu_new = scale_prev * mu_i + s_mu * inv_l_new
        to0_block = (
            scale_prev[:, None] * to0_block
            + (scale_prev * mu_i)[:, None] * (o0_block - o0_block_new)
            + (s_gv0 + s_tv0 - s_mu[:, None] * o0_block_new) * inv_l_new[:, None]
        )
        to1_block = (
            scale_prev[:, None] * to1_block
            + (scale_prev * mu_i)[:, None] * (o1_block - o1_block_new)
            + (s_gv1 + s_tv1 - s_mu[:, None] * o1_block_new) * inv_l_new[:, None]
        )
        # update stats
        mu_i = mu_new
        o0_block = o0_block_new
        o1_block = o1_block_new
        l_i = l_new
        m_i = m_ij
        # move to next block
        k0_curr = k0_next; v0_curr = v0_next
        k1_curr = k1_next; v1_curr = v1_next
        k0_block_ptr = k0_ptr_next; v0_block_ptr = v0_ptr_next
        k1_block_ptr = k1_ptr_next; v1_block_ptr = v1_ptr_next
        tk0_curr = tk0_next; tv0_curr = tv0_next
        tk1_curr = tk1_next; tv1_curr = tv1_next
        tk0_block_ptr = tk0_ptr_next; tv0_block_ptr = tv0_ptr_next
        tk1_block_ptr = tk1_ptr_next; tv1_block_ptr = tv1_ptr_next
    return (
        o0_block, o1_block,
        to0_block, to1_block,
        l_i, m_i, mu_i
    )


# attention forward with jvp
@triton.jit
def _attn_fwd_dual(
    q, k, v,  #  query, key, value
    tq, tk, tv,  # tangent query, key, value
    m, o, to,  #  logsumexp, output, tangent output
    softmax_scale,  #  scaling factor for softmax 
    stride_qb, stride_ql, stride_qh, stride_qd,  #
    stride_kb, stride_kl, stride_kh, stride_kd,  #
    stride_vb, stride_vl, stride_vh, stride_vd,  #
    stride_tqb, stride_tql, stride_tqh, stride_tqd,  #
    stride_tkb, stride_tkl, stride_tkh, stride_tkd,  #
    stride_tvb, stride_tvl, stride_tvh, stride_tvd,  #
    stride_ob, stride_ol, stride_oh, stride_od,  #
    stride_tob, stride_tol, stride_toh, stride_tod,  #
    num_heads: tl.constexpr,  #  number of heads
    seq_len_q: tl.constexpr,  #  sequence length of query
    seq_len_kv: tl.constexpr,  #  sequence length of key/value
    head_dim: tl.constexpr,  #  dimension of each head
    block_q: tl.constexpr,  #  block size for query
    block_kv: tl.constexpr,  #  block size for key/value
):
    # This indicate which block in the sequence length to process
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bh_batch = pid_bh // num_heads
    bh_head = pid_bh % num_heads
    offs_q = pid_q * block_q + tl.arange(0, block_q)

    q += bh_batch * stride_qb + bh_head * stride_qh
    k += bh_batch * stride_kb + bh_head * stride_kh
    v += bh_batch * stride_vb + bh_head * stride_vh
    o += bh_batch * stride_ob + bh_head * stride_oh
    q_block_ptr = tl.make_block_ptr(
        base=q, shape=(seq_len_q, head_dim), strides=(stride_ql, stride_qd),
        offsets=(pid_q * block_q, 0), block_shape=(block_q, head_dim), order=(1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k, shape=(head_dim, seq_len_kv), strides=(stride_kd, stride_kl),
        offsets=(0, 0), block_shape=(head_dim, block_kv), order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v, shape=(seq_len_kv, head_dim), strides=(stride_vl, stride_vd),
        offsets=(0, 0), block_shape=(block_kv, head_dim), order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o, shape=(seq_len_q, head_dim), strides=(stride_ol, stride_od),
        offsets=(pid_q * block_q, 0), block_shape=(block_q, head_dim), order=(1, 0)
    )
    # initialize blocks
    q_block = tl.load(q_block_ptr)
    m_i = tl.zeros([block_q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_q], dtype=tl.float32)
    o_block = tl.zeros([block_q, head_dim], dtype=tl.float32)

    tq += bh_batch * stride_tqb + bh_head * stride_tqh
    tk += bh_batch * stride_tkb + bh_head * stride_tkh
    tv += bh_batch * stride_tvb + bh_head * stride_tvh
    to += bh_batch * stride_tob + bh_head * stride_toh
    tq_block_ptr = tl.make_block_ptr(
        base=tq, shape=(seq_len_q, head_dim), strides=(stride_tql, stride_tqd),
        offsets=(pid_q * block_q, 0), block_shape=(block_q, head_dim), order=(1, 0)
    )
    tk_block_ptr = tl.make_block_ptr(
        base=tk, shape=(head_dim, seq_len_kv), strides=(stride_tkd, stride_tkl),
        offsets=(0, 0), block_shape=(head_dim, block_kv), order=(0, 1)
    )
    tv_block_ptr = tl.make_block_ptr(
        base=tv, shape=(seq_len_kv, head_dim), strides=(stride_tvl, stride_tvd),
        offsets=(0, 0), block_shape=(block_kv, head_dim), order=(1, 0)
    )
    to_block_ptr = tl.make_block_ptr(
        base=to, shape=(seq_len_q, head_dim), strides=(stride_tol, stride_tod),
        offsets=(pid_q * block_q, 0), block_shape=(block_q, head_dim), order=(1, 0)
    )
    tq_block = tl.load(tq_block_ptr)
    mu_i = tl.zeros([block_q], dtype=tl.float32)  # running sum of the JVP
    to_block = tl.zeros([block_q, head_dim], dtype=tl.float32)

    o_block, to_block, l_i, m_i, mu_i = _attn_fwd_dual_inner(
        o_block, to_block,  #
        l_i, m_i, mu_i,  #
        q_block, tq_block,  #
        k_block_ptr, tk_block_ptr, v_block_ptr, tv_block_ptr,  #
        softmax_scale,  #
        seq_len_kv,  #
        block_q,  #
        block_kv  #
    )
    # epilogue
    m_i += tl.math.log(l_i)
    m_ptrs = m + pid_bh * seq_len_q + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(o_block_ptr, o_block.to(o.type.element_ty))
    tl.store(to_block_ptr, to_block.to(to.type.element_ty))


# attention forward with jvp (split head dim)
@triton.jit
def _attn_fwd_dual_split(
    q, k, v,  #
    tq, tk, tv,  #
    m, o, to,  #
    softmax_scale,  #
    stride_qb, stride_ql, stride_qh, stride_qd,  #
    stride_kb, stride_kl, stride_kh, stride_kd,  #
    stride_vb, stride_vl, stride_vh, stride_vd,  #
    stride_tqb, stride_tql, stride_tqh, stride_tqd,  #
    stride_tkb, stride_tkl, stride_tkh, stride_tkd,  #
    stride_tvb, stride_tvl, stride_tvh, stride_tvd,  #
    stride_ob, stride_ol, stride_oh, stride_od,  #
    stride_tob, stride_tol, stride_toh, stride_tod,  #
    num_heads: tl.constexpr,  #  number of heads
    seq_len_q: tl.constexpr,  #  sequence length of query
    seq_len_kv: tl.constexpr,  #  sequence length of key/value
    head_dim: tl.constexpr,  #  dimension of each head
    tile_dim: tl.constexpr,  #  dimension of the main tile
    tail_dim: tl.constexpr,  #  dimension of the tail tile
    block_q: tl.constexpr,  #  block size for query
    block_kv: tl.constexpr,  #  block size for key/value

):
    # This indicate which block in the sequence length to process
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bh_batch = pid_bh // num_heads
    bh_head = pid_bh % num_heads
    offs_q = pid_q * block_q + tl.arange(0, block_q)  # [BQ]

    q += bh_batch * stride_qb + bh_head * stride_qh
    k += bh_batch * stride_kb + bh_head * stride_kh
    v += bh_batch * stride_vb + bh_head * stride_vh
    o += bh_batch * stride_ob + bh_head * stride_oh

    q_block_ptr = tl.make_block_ptr(
        base=q, shape=(seq_len_q, head_dim), strides=(stride_ql, stride_qd),
        offsets=(pid_q * block_q, 0), block_shape=(block_q, tile_dim), order=(1, 0)
    )
    q1_block_ptr = tl.make_block_ptr(
        base=q, shape=(seq_len_q, head_dim), strides=(stride_ql, stride_qd),
        offsets=(pid_q * block_q, tile_dim), block_shape=(block_q, tail_dim), order=(1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k, shape=(head_dim, seq_len_kv), strides=(stride_kd, stride_kl),
        offsets=(0, 0), block_shape=(tile_dim, block_kv), order=(0, 1),
    )
    k1_block_ptr = tl.make_block_ptr(
        base=k, shape=(head_dim, seq_len_kv), strides=(stride_kd, stride_kl),
        offsets=(tile_dim, 0), block_shape=(tail_dim, block_kv), order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v, shape=(seq_len_kv, head_dim), strides=(stride_vl, stride_vd),
        offsets=(0, 0), block_shape=(block_kv, tile_dim), order=(1, 0),
    )
    v1_block_ptr = tl.make_block_ptr(
        base=v, shape=(seq_len_kv, head_dim), strides=(stride_vl, stride_vd),
        offsets=(0, tile_dim), block_shape=(block_kv, tail_dim), order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o, shape=(seq_len_q, head_dim), strides=(stride_ol, stride_od),
        offsets=(pid_q * block_q, 0), block_shape=(block_q, tile_dim), order=(1, 0)
    )
    o1_block_ptr = tl.make_block_ptr(
        base=o, shape=(seq_len_q, head_dim), strides=(stride_ol, stride_od),
        offsets=(pid_q * block_q, tile_dim), block_shape=(block_q, tail_dim), order=(1, 0)
    )

    q_block = tl.load(q_block_ptr)
    q1_block = tl.load(q1_block_ptr, boundary_check=(1,), padding_option="zero")
    m_i = tl.zeros([block_q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_q], dtype=tl.float32)
    o_block = tl.zeros([block_q, tile_dim], dtype=tl.float32)
    o1_block = tl.zeros([block_q, tail_dim], dtype=tl.float32)

    tq += bh_batch * stride_tqb + bh_head * stride_tqh
    tk += bh_batch * stride_tkb + bh_head * stride_tkh
    tv += bh_batch * stride_tvb + bh_head * stride_tvh
    to += bh_batch * stride_tob + bh_head * stride_toh

    tq_block_ptr = tl.make_block_ptr(
        base=tq, shape=(seq_len_q, head_dim), strides=(stride_tql, stride_tqd),
        offsets=(pid_q * block_q, 0), block_shape=(block_q, tile_dim), order=(1, 0)
    )
    tq1_block_ptr = tl.make_block_ptr(
        base=tq, shape=(seq_len_q, head_dim), strides=(stride_tql, stride_tqd),
        offsets=(pid_q * block_q, tile_dim), block_shape=(block_q, tail_dim), order=(1, 0)
    )
    tk_block_ptr = tl.make_block_ptr(
        base=tk, shape=(head_dim, seq_len_kv), strides=(stride_tkd, stride_tkl),
        offsets=(0, 0), block_shape=(tile_dim, block_kv), order=(0, 1)
    )
    tk1_block_ptr = tl.make_block_ptr(
        base=tk, shape=(head_dim, seq_len_kv), strides=(stride_tkd, stride_tkl),
        offsets=(tile_dim, 0), block_shape=(tail_dim, block_kv), order=(0, 1)
    )
    tv_block_ptr = tl.make_block_ptr(
        base=tv, shape=(seq_len_kv, head_dim), strides=(stride_tvl, stride_tvd),
        offsets=(0, 0), block_shape=(block_kv, tile_dim), order=(1, 0)
    )
    tv1_block_ptr = tl.make_block_ptr(
        base=tv, shape=(seq_len_kv, head_dim), strides=(stride_tvl, stride_tvd),
        offsets=(0, tile_dim), block_shape=(block_kv, tail_dim), order=(1, 0)
    )
    to_block_ptr = tl.make_block_ptr(
        base=to, shape=(seq_len_q, head_dim), strides=(stride_tol, stride_tod),
        offsets=(pid_q * block_q, 0), block_shape=(block_q, tile_dim), order=(1, 0)
    )
    to1_block_ptr = tl.make_block_ptr(
        base=to, shape=(seq_len_q, head_dim), strides=(stride_tol, stride_tod),
        offsets=(pid_q * block_q, tile_dim), block_shape=(block_q, tail_dim), order=(1, 0)
    )
    tq_block = tl.load(tq_block_ptr)
    tq1_block = tl.load(tq1_block_ptr, boundary_check=(1,), padding_option="zero")
    to_block = tl.zeros([block_q, tile_dim], dtype=tl.float32)
    to1_block = tl.zeros([block_q, tail_dim], dtype=tl.float32)
    mu_i = tl.zeros([block_q], dtype=tl.float32)

    (
        o_block, o1_block,
        to_block, to1_block,
        l_i, m_i, mu_i
    ) = _attn_fwd_dual_inner_split(
        o_block, o1_block,
        to_block, to1_block,
        l_i, m_i, mu_i,
        q_block, q1_block,
        tq_block, tq1_block,
        k_block_ptr, k1_block_ptr,
        tk_block_ptr, tk1_block_ptr,
        v_block_ptr, v1_block_ptr,
        tv_block_ptr, tv1_block_ptr,
        softmax_scale,
        seq_len_kv,
        block_q,
        block_kv,
    )
    # epilogue
    m_i += tl.math.log(l_i)
    m_ptrs = m + pid_bh * seq_len_q + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(o_block_ptr, o_block.to(o.type.element_ty))
    tl.store(o1_block_ptr, o1_block.to(o.type.element_ty), boundary_check=(1,))
    tl.store(to_block_ptr, to_block.to(to.type.element_ty))
    tl.store(to1_block_ptr, to1_block.to(to.type.element_ty), boundary_check=(1,))
