"""Building blocks of the Transformer (Part 3): RoPE, causal self-attention, SwiGLU feed-forward."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def naive_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d_k)) V with a causal mask, written out by hand.

    q, k, v: (..., seq, d). The mask sets every "future" position (j > i) to -inf *before* the
    softmax, so those positions get exactly zero weight.
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., seq_q, seq_k)
    seq_q, seq_k = scores.shape[-2:]
    future = torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class RoPE(nn.Module):
    """Rotary position embedding.

    Every consecutive pair of features (x_{2k}, x_{2k+1}) at position i is rotated by the angle
    ``i / theta ** (2k / d_head)``. cos / sin are pre-computed once into non-persistent buffers
    (so they are not saved in checkpoints and are rebuilt from the config).
    """

    def __init__(self, theta: float, d_head: int, max_len: int):
        super().__init__()
        if d_head % 2:
            raise ValueError("d_head must be even for RoPE")
        k = torch.arange(d_head // 2, dtype=torch.float64)
        inv_freq = theta ** (-2.0 * k / d_head)  # (d_head/2,)
        angles = torch.arange(max_len, dtype=torch.float64)[:, None] * inv_freq[None, :]  # (max_len, d_head/2)
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)
        self.d_head, self.max_len, self.theta = d_head, max_len, theta

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """x: (..., seq, d_head). ``positions``: optional (seq,) integer positions (default 0..seq-1)."""
        seq = x.shape[-2]
        if positions is None:
            cos, sin = self.cos[:seq], self.sin[:seq]
        else:
            cos, sin = self.cos[positions], self.sin[positions]
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)  # (seq, d_head/2), broadcast over leading dims
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)  # (..., seq, d/2, 2)
        return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE on queries and keys only.

    Input / output shape: (..., seq, d_model); any number of leading batch dimensions is allowed.
    ``past`` (optional) is a (k, v) pair from earlier steps, enabling a KV cache at inference time.
    """

    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model, self.num_heads, self.d_head = d_model, num_heads, d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope  # shared between layers; None = no position information

    def forward(self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False):
        lead = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])  # (B, seq, d_model) so SDPA only ever sees 4-D inputs
        seq = x.shape[-2]
        h = self.num_heads
        q = rearrange(self.q_proj(x), "b s (h d) -> b h s d", h=h)
        k = rearrange(self.k_proj(x), "b s (h d) -> b h s d", h=h)
        v = rearrange(self.v_proj(x), "b s (h d) -> b h s d", h=h)
        offset = 0 if past is None else past[0].shape[-2]
        if self.rope is not None:
            pos = None if offset == 0 else torch.arange(offset, offset + seq, device=x.device)
            q, k = self.rope(q, pos), self.rope(k, pos)  # not applied to v
        if past is not None:
            k, v = torch.cat([past[0], k], dim=-2), torch.cat([past[1], v], dim=-2)
        if offset == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # new queries may attend to every cached key, and causally to the new ones
            mask = torch.ones(seq, offset + seq, dtype=torch.bool, device=x.device).tril(diagonal=offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = self.o_proj(rearrange(out, "b h s d -> b s (h d)"))
        out = out.reshape(*lead, seq, self.d_model)
        return (out, (k, v)) if use_cache else out


class FeedForward(nn.Module):
    """SwiGLU: W2( SiLU(W1 x) * W3 x )."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Plain SiLU MLP, W2 SiLU(W1 x) (used by the optional 'SiLU instead of SwiGLU' experiment)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))
