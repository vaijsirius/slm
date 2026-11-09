"""Tests for generation (Part 7) and the packaged inference script (Part 8)."""
import collections
import os
import subprocess
import sys

import pytest
import torch

from minilm.generate import generate, generate_ids, sample_next
from minilm.model import ModelConfig, TransformerLM
from minilm.tokenizer import Tokenizer

EOT = "<|endoftext|>"
CFG = dict(vocab_size=257 + 0, context_length=32, d_model=32, num_layers=2, num_heads=4, d_ff=64, rope_theta=10_000.0)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [], [EOT])  # byte-level tokenizer: ids 0..255 + <|endoftext|> = 256
    model = TransformerLM(**CFG).eval()
    return model, tok


def test_greedy_is_deterministic(setup):
    model, tok = setup
    a = generate(model, tok, "Once upon a time", 40, temperature=0.0)
    b = generate(model, tok, "Once upon a time", 40, temperature=0.0)
    assert a == b and a.startswith("Once upon a time") and len(a) > len("Once upon a time")


def test_greedy_picks_argmax(setup):
    logits = torch.tensor([0.1, 3.0, 2.9, -1.0])
    assert sample_next(logits, temperature=0.0) == 1
    # tiny temperature / tiny top-p collapse onto the argmax as well
    assert sample_next(logits, temperature=1e-4) == 1
    assert all(sample_next(logits, 1.0, top_p=0.01, generator=torch.Generator().manual_seed(s)) == 1 for s in range(20))


def test_same_seed_same_sample_different_seed_differs(setup):
    model, tok = setup
    g = lambda s: generate(model, tok, "Hi", 30, 1.0, 0.95, torch.Generator().manual_seed(s))  # noqa: E731
    assert g(1) == g(1)
    assert len({g(s) for s in range(6)}) > 1


def test_top_p_keeps_smallest_set_reaching_p():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()
    counts = collections.Counter(
        sample_next(logits, 1.0, 0.7, torch.Generator().manual_seed(i)) for i in range(3000))
    assert set(counts) == {0, 1}  # 0.5 + 0.3 = 0.8 >= 0.7 -> keep exactly two tokens
    assert counts[0] / 3000 == pytest.approx(0.5 / 0.8, abs=0.04)  # rescaled to sum to one
    counts = collections.Counter(
        sample_next(logits, 1.0, 0.5, torch.Generator().manual_seed(i)) for i in range(200))
    assert set(counts) == {0}  # the top token alone already reaches p = 0.5
    counts = collections.Counter(
        sample_next(logits, 1.0, 1.0, torch.Generator().manual_seed(i)) for i in range(4000))
    assert set(counts) == {0, 1, 2, 3}  # p = 1 disables the filter


def test_temperature_sharpens_and_flattens():
    logits = torch.tensor([2.0, 1.0, 0.0])

    def top_freq(T):
        c = collections.Counter(sample_next(logits, T, 1.0, torch.Generator().manual_seed(i)) for i in range(2000))
        return c[0] / 2000

    assert top_freq(0.3) > top_freq(1.0) > top_freq(3.0)


class _Fake:
    """Stand-in model that always predicts one fixed token."""

    context_length = 8

    def __init__(self, token):
        self.token = token
        self.calls = []

    def parameters(self):
        return iter([torch.zeros(1)])

    training = False

    def eval(self):
        return self

    def __call__(self, x, past=None, use_cache=False):
        self.calls.append(x.shape[1])
        logits = torch.full((1, x.shape[1], 257), -10.0)
        logits[..., self.token] = 10.0
        return logits


def test_stops_at_end_of_text(setup):
    _, tok = setup
    assert generate_ids(_Fake(256), tok, "abc", 50, 0.0, use_cache=False) == []
    out = generate_ids(_Fake(65), tok, "abc", 5, 0.0, use_cache=False)
    assert out == [65] * 5  # no EOT -> runs to max_new_tokens


def test_context_is_truncated_to_last_256_style_window(setup):
    _, tok = setup
    fake = _Fake(65)
    generate_ids(fake, tok, "x" * 30, 12, 0.0, use_cache=False)  # context_length = 8
    assert max(fake.calls) == 8  # never feeds more than context_length tokens


def test_long_generation_beyond_context_runs_and_cache_matches_no_cache(setup):
    model, tok = setup
    a = generate_ids(model, tok, "Once", 70, 0.0, use_cache=True)  # 70 > context_length = 32
    b = generate_ids(model, tok, "Once", 70, 0.0, use_cache=False)
    assert len(a) == len(b) == 70
    assert a == b


def test_empty_prompt_works(setup):
    model, tok = setup
    assert len(generate_ids(model, tok, "", 5, 0.0)) <= 5


def test_inference_script_end_to_end_is_seeded(tmp_path, setup):
    """Build a tiny release folder and run inference.py twice in a subprocess."""
    model, tok = setup
    from minilm.tokenizer import save_vocab_and_merges
    rel = tmp_path / "release"
    rel.mkdir()
    torch.save(model.state_dict(), rel / "model.pt")
    ModelConfig(**CFG).to_json(str(rel / "config.json"))
    save_vocab_and_merges({i: b for i, b in tok.vocab.items()}, [], str(rel / "vocab.json"), str(rel / "merges.txt"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(root, "inference.py"), "--model-dir", str(rel), "--prompt", "Hi",
           "--max-new-tokens", "20", "--temperature", "0.9", "--top-p", "0.9", "--seed", "7", "--device", "cpu"]
    run = lambda: subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=root,  # noqa: E731
                                 check=True).stdout
    out1, out2 = run(), run()
    assert out1.split("\nGenerated")[0] == out2.split("\nGenerated")[0]
    last = out1.strip().splitlines()[-1]
    assert last.startswith("Generated 20 tokens") and "tokens/s" in last
