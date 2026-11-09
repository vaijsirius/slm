"""Tests for RoPE, attention and the feed-forward layer (Part 3)."""
import math

import pytest
import torch
import torch.nn.functional as F

from minilm.model import CausalSelfAttention, FeedForward, RoPE, SiLUFeedForward, naive_causal_attention

torch.manual_seed(0)


# --------------------------------------------------------------------- hand-written attention
@pytest.mark.parametrize("shape", [(5, 8), (3, 7, 16), (2, 4, 9, 32), (2, 3, 4, 5, 8)])
def test_naive_attention_matches_sdpa(shape):
    q, k, v = (torch.randn(*shape) for _ in range(3))
    mine = naive_causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert mine.shape == q.shape
    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max()


def test_naive_attention_is_causal_and_test_would_catch_non_causal():
    q, k, v = (torch.randn(2, 6, 8) for _ in range(3))
    base = naive_causal_attention(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, 4:], v2[:, 4:] = torch.randn(2, 2, 8), torch.randn(2, 2, 8)  # change the future (t >= 4)
    changed = naive_causal_attention(q, k2, v2)
    assert torch.equal(base[:, :4], changed[:, :4])  # positions 0..3 untouched
    assert not torch.allclose(base[:, 4:], changed[:, 4:])  # but later positions do change
    # the non-causal reference would fail the same check -> the check is meaningful
    nc = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    nc2 = F.scaled_dot_product_attention(q, k2, v2, is_causal=False)
    assert not torch.allclose(nc[:, :4], nc2[:, :4])


def test_first_position_attends_only_to_itself():
    q, k, v = (torch.randn(1, 5, 8) for _ in range(3))
    assert torch.allclose(naive_causal_attention(q, k, v)[:, 0], v[:, 0], atol=1e-6)


# --------------------------------------------------------------------- RoPE
def slow_rope(x, theta):
    """Reference: build the full block-diagonal rotation matrix for every position."""
    seq, d = x.shape[-2:]
    out = torch.zeros_like(x, dtype=torch.float64)
    for i in range(seq):
        R = torch.zeros(d, d, dtype=torch.float64)
        for k in range(d // 2):
            ang = i / theta ** (2 * k / d)
            c, s = math.cos(ang), math.sin(ang)
            R[2 * k, 2 * k], R[2 * k, 2 * k + 1] = c, -s
            R[2 * k + 1, 2 * k], R[2 * k + 1, 2 * k + 1] = s, c
        out[..., i, :] = x[..., i, :].double() @ R.T
    return out


@pytest.mark.parametrize("theta,d,seq", [(10_000.0, 8, 6), (10_000.0, 32, 20), (500.0, 4, 11)])
def test_rope_matches_rotation_matrix(theta, d, seq):
    rope = RoPE(theta, d, max_len=32)
    x = torch.randn(2, 3, seq, d)
    assert torch.allclose(rope(x).double(), slow_rope(x, theta), atol=1e-5)


def test_rope_position_zero_is_identity_and_norm_preserved():
    rope = RoPE(10_000.0, 32, 64)
    x = torch.randn(4, 10, 32)
    out = rope(x)
    assert torch.allclose(out[:, 0], x[:, 0])
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)  # rotations preserve length


def test_rope_explicit_positions_equal_slice():
    rope = RoPE(10_000.0, 16, 64)
    x = torch.randn(2, 8, 16)
    full = rope(torch.cat([torch.zeros(2, 5, 16), x], dim=1))[:, 5:]
    assert torch.allclose(rope(x, positions=torch.arange(5, 13)), full, atol=1e-6)


def test_rope_buffers_not_in_state_dict():
    rope = RoPE(10_000.0, 16, 64)
    assert len(rope.state_dict()) == 0
    assert rope.cos.shape == rope.sin.shape == (64, 8)


def test_rope_relative_position_property():
    """<R_i q, R_j k> depends only on i - j."""
    rope = RoPE(10_000.0, 32, 128)
    q, k = torch.randn(32), torch.randn(32)

    def dot(i, j):
        qi = rope(q.expand(1, i + 1, 32).contiguous())[0, i]
        kj = rope(k.expand(1, j + 1, 32).contiguous())[0, j]
        return (qi @ kj).item()

    base = dot(7, 3)
    for shift in (0, 5, 40, 100):
        assert dot(7 + shift, 3 + shift) == pytest.approx(base, abs=1e-3)
    assert dot(7, 4) != pytest.approx(base, abs=1e-3)  # a different distance does change it


# --------------------------------------------------------------------- shapes
@pytest.mark.parametrize("lead", [(), (2,), (2, 3), (2, 1, 3)])
def test_every_piece_preserves_shape_with_extra_batch_dims(lead):
    d_model, heads, d_ff, seq = 64, 4, 96, 10
    x = torch.randn(*lead, seq, d_model)
    rope = RoPE(10_000.0, d_model // heads, 32)
    assert rope(torch.randn(*lead, seq, d_model // heads)).shape == (*lead, seq, d_model // heads)
    assert CausalSelfAttention(d_model, heads, rope)(x).shape == x.shape
    assert FeedForward(d_model, d_ff)(x).shape == x.shape
    assert SiLUFeedForward(d_model, d_ff)(x).shape == x.shape
    assert torch.nn.RMSNorm(d_model, eps=1e-5)(x).shape == x.shape


# --------------------------------------------------------------------- attention wiring
def test_attention_matches_manual_computation_from_weights():
    """RoPE goes on q and k only, heads are split/merged correctly, projections have no bias."""
    d_model, heads, seq = 32, 4, 9
    rope = RoPE(10_000.0, d_model // heads, 16)
    attn = CausalSelfAttention(d_model, heads, rope)
    x = torch.randn(2, seq, d_model)
    dh = d_model // heads

    def split(t):  # (b, s, d_model) -> (b, h, s, dh)
        return t.view(2, seq, heads, dh).transpose(1, 2)

    q, k, v = split(x @ attn.q_proj.weight.T), split(x @ attn.k_proj.weight.T), split(x @ attn.v_proj.weight.T)
    q, k = rope(q), rope(k)
    ref = naive_causal_attention(q, k, v).transpose(1, 2).reshape(2, seq, d_model) @ attn.o_proj.weight.T
    assert torch.allclose(attn(x), ref, atol=1e-5)
    assert all(m.bias is None for m in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj))


def test_attention_cannot_see_the_future():
    attn = CausalSelfAttention(32, 4, RoPE(10_000.0, 8, 16))
    x = torch.randn(2, 10, 32)
    y = attn(x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 32)
    y2 = attn(x2)
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y[:, 6:], y2[:, 6:])


def test_attention_kv_cache_matches_full_forward():
    attn = CausalSelfAttention(32, 4, RoPE(10_000.0, 8, 16))
    x = torch.randn(1, 8, 32)
    full = attn(x)
    out, past = attn(x[:, :5], use_cache=True)
    step_outs = [out]
    for t in range(5, 8):
        o, past = attn(x[:, t:t + 1], past, use_cache=True)
        step_outs.append(o)
    assert torch.allclose(torch.cat(step_outs, dim=1), full, atol=1e-5)


def test_feed_forward_is_swiglu():
    ff = FeedForward(16, 24)
    x = torch.randn(3, 16)
    ref = (F.silu(x @ ff.w1.weight.T) * (x @ ff.w3.weight.T)) @ ff.w2.weight.T
    assert torch.allclose(ff(x), ref, atol=1e-6)
    assert ff.w1.weight.shape == (24, 16) and ff.w2.weight.shape == (16, 24) and ff.w3.weight.shape == (24, 16)


def test_rmsnorm_settings_used_in_model_blocks():
    from minilm.model import TransformerBlock
    blk = TransformerBlock(32, 4, 48)
    assert isinstance(blk.ln1, torch.nn.RMSNorm) and blk.ln1.eps == 1e-5
