"""Tests for the full model (Part 4)."""
import math
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from minilm.model import ModelConfig, TransformerBlock, TransformerLM

CFG = dict(vocab_size=10_000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344,
           rope_theta=10_000.0)
SMALL = dict(vocab_size=100, context_length=32, d_model=32, num_layers=2, num_heads=4, d_ff=64, rope_theta=10_000.0)


@pytest.fixture(scope="module")
def big_model():
    torch.manual_seed(0)
    return TransformerLM(**CFG).eval()


def test_logits_shape(big_model):
    toks = torch.randint(0, 10_000, (2, 17))
    with torch.no_grad():
        assert big_model(toks).shape == (2, 17, 10_000)


def future_dependence(model, t=7, seq=20):
    """Max |difference| of the logits at positions <= t when only tokens after t are changed."""
    torch.manual_seed(1)
    a = torch.randint(0, model.config.vocab_size, (2, seq))
    b = a.clone()
    b[:, t + 1:] = torch.randint(0, model.config.vocab_size, (2, seq - t - 1))
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert not torch.equal(la[:, t + 1:], lb[:, t + 1:])  # later positions *do* react (test is sensitive)
    return (la[:, : t + 1] - lb[:, : t + 1]).abs().max().item()


def test_model_cannot_see_the_future():
    model = TransformerLM(**SMALL).eval()
    assert future_dependence(model) == 0.0  # exactly the same, not just close


def test_future_test_would_catch_a_leaky_mask():
    """Negative control: make attention non-causal and check the future-dependence test fails."""
    model = TransformerLM(**SMALL).eval()
    real_sdpa = F.scaled_dot_product_attention

    def leaky(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        return real_sdpa(q, k, v, attn_mask=None, is_causal=False)

    with mock.patch("minilm.model.layers.F.scaled_dot_product_attention", leaky):
        assert future_dependence(model) > 1e-4


def test_changing_the_past_changes_the_present():
    model = TransformerLM(**SMALL).eval()
    a = torch.randint(0, 100, (1, 12))
    b = a.clone()
    b[:, 2] = (b[:, 2] + 1) % 100
    with torch.no_grad():
        assert not torch.allclose(model(a)[:, 8], model(b)[:, 8])


def test_initial_loss_is_ln_vocab(big_model):
    """At initialisation the logits are small and nearly input-independent, so the model predicts an
    almost uniform distribution over the V = 10,000 tokens. The cross entropy of the uniform
    distribution is -log(1/V) = ln(V) = 9.21 for *any* target, so random tokens must give ~9.2.
    (Default PyTorch init gives logits with std ~0.6, which adds ~0.5*0.6^2 = 0.2 on top.)"""
    torch.manual_seed(0)
    toks = torch.randint(0, 10_000, (8, 65))
    with torch.no_grad():
        logits = big_model(toks[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 10_000), toks[:, 1:].reshape(-1)).item()
    assert abs(loss - math.log(10_000)) < 0.5, loss
    assert loss > math.log(10_000) - 0.05  # cannot beat uniform by chance at init


def test_parameter_count_by_hand(big_model):
    V, d, L, dff = 10_000, 512, 4, 1344
    emb = V * d
    attn = 4 * d * d
    ffn = 3 * d * dff
    norms = 2 * d
    per_layer = attn + ffn + norms
    expected = emb + L * per_layer + d + V * d  # + final norm + output projection
    assert expected == 22_696_448
    assert big_model.num_parameters() == expected == sum(p.numel() for p in big_model.parameters())
    assert L * (attn + ffn) == 12_451_840  # ~12.5M in the transformer layers (+ 4096 norm weights)
    assert sum(p.numel() for p in big_model.layers.parameters()) == L * per_layer


def test_block_is_prenorm_residual():
    blk = TransformerBlock(32, 4, 64).eval()
    x = torch.randn(2, 6, 32)
    z = x + blk.attn(blk.ln1(x))
    y = z + blk.ffn(blk.ln2(z))
    assert torch.allclose(blk(x), y, atol=1e-6)


def test_layers_are_in_a_module_list(big_model):
    assert isinstance(big_model.layers, torch.nn.ModuleList) and len(big_model.layers) == 4


def test_kv_cache_logits_match_full_forward():
    torch.manual_seed(0)
    model = TransformerLM(**SMALL).eval()
    toks = torch.randint(0, 100, (1, 20))
    with torch.no_grad():
        full = model(toks)
        logits, past = model(toks[:, :12], use_cache=True)
        outs = [logits]
        for t in range(12, 20):
            lg, past = model(toks[:, t:t + 1], past, use_cache=True)
            outs.append(lg)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-4)


@pytest.mark.parametrize("kw", [dict(norm="post"), dict(norm="none"), dict(use_rope=False), dict(ffn="silu"),
                                dict(tie_weights=True)])
def test_experiment_variants_run(kw):
    model = TransformerLM(**SMALL, **kw)
    toks = torch.randint(0, 100, (2, 10))
    assert model(toks).shape == (2, 10, 100)
    if kw.get("tie_weights"):
        assert model.lm_head.weight is model.token_embeddings.weight


def test_config_json_round_trip(tmp_path):
    cfg = ModelConfig(**CFG)
    p = tmp_path / "config.json"
    cfg.to_json(str(p))
    import json
    assert set(json.load(open(p))) == set(ModelConfig.CORE)
    assert ModelConfig.from_json(str(p)) == cfg
    assert TransformerLM.from_config(cfg).num_parameters() == 22_696_448
