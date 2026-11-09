"""Tests for the training set-up (Part 5) and train.py (Part 6)."""
import copy
import csv
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from minilm.model import TransformerLM
from minilm.train import (build_optimizer, build_param_groups, build_scheduler, cross_entropy_loss,
                          fixed_eval_batches, get_batch, load_checkpoint, lr_at_step, make_autocast,
                          save_checkpoint, train_step)

TINY = dict(vocab_size=50, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64, rope_theta=10_000.0)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return TransformerLM(**TINY)


# --------------------------------------------------------------------------- loss
def test_cross_entropy_matches_manual_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11)
    targets = torch.randint(0, 11, (3, 5))
    manual = -torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
    assert torch.allclose(cross_entropy_loss(logits, targets), manual, atol=1e-6)


def test_uniform_logits_give_ln_vocab():
    assert cross_entropy_loss(torch.zeros(2, 4, 100), torch.randint(0, 100, (2, 4))).item() == pytest.approx(math.log(100))


# --------------------------------------------------------------------------- optimizer
def test_param_groups_cover_every_parameter_exactly_once():
    m = tiny_model()
    groups = build_param_groups(m, 0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    assert set(ids) == {id(p) for p in m.parameters()}
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0]


def test_decay_group_is_matrices_and_nodecay_is_norm_weights():
    m = tiny_model()
    decay, no_decay = build_param_groups(m, 0.1)
    assert all(p.ndim == 2 for p in decay["params"])
    norm_weights = {id(mod.weight) for mod in m.modules() if isinstance(mod, torch.nn.RMSNorm)}
    assert {id(p) for p in no_decay["params"]} == norm_weights
    assert len(norm_weights) == 2 * TINY["num_layers"] + 1


def test_tied_weights_are_not_duplicated():
    torch.manual_seed(0)
    m = TransformerLM(**TINY, tie_weights=True)
    groups = build_param_groups(m, 0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))


def test_adamw_hyperparameters():
    opt = build_optimizer(tiny_model(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["betas"] == (0.9, 0.95)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]


def test_weight_decay_only_shrinks_decay_group():
    m = tiny_model()
    opt = build_optimizer(m, lr=0.1, weight_decay=0.5)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)  # zero gradient: only decoupled decay can move weights
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt.step()
    for n, p in m.named_parameters():
        if p.ndim >= 2:
            assert torch.allclose(p, before[n] * (1 - 0.1 * 0.5), atol=1e-6), n
        else:
            assert torch.equal(p, before[n]), n


# --------------------------------------------------------------------------- schedule
A_MAX, A_MIN, TW, TC = 1e-3, 1e-4, 10, 110


def test_lr_schedule_hand_values():
    f = lambda t: lr_at_step(t, A_MAX, A_MIN, TW, TC)  # noqa: E731
    assert f(0) == 0.0
    assert f(5) == pytest.approx(0.5 * A_MAX)  # t / Tw * max
    assert f(10) == pytest.approx(A_MAX)  # cosine starts at the peak
    assert f(60) == pytest.approx((A_MAX + A_MIN) / 2)  # half way through the cosine: cos(pi/2) = 0
    assert f(35) == pytest.approx(A_MIN + 0.5 * (1 + math.cos(math.pi * 0.25)) * (A_MAX - A_MIN))
    assert f(110) == pytest.approx(A_MIN)
    assert f(500) == A_MIN  # constant after Tc


def test_lr_schedule_is_monotone_after_warmup():
    vals = [lr_at_step(t, A_MAX, A_MIN, TW, TC) for t in range(TW, TC + 5)]
    assert all(a >= b - 1e-15 for a, b in zip(vals, vals[1:]))
    assert max(vals) == pytest.approx(A_MAX) and min(vals) == pytest.approx(A_MIN)


def test_scheduler_follows_the_formula():
    m = tiny_model()
    opt = build_optimizer(m, lr=A_MAX)
    sch = build_scheduler(opt, A_MAX, A_MIN, TW, TC)
    for t in range(130):
        assert opt.param_groups[0]["lr"] == pytest.approx(lr_at_step(t, A_MAX, A_MIN, TW, TC), abs=1e-12)
        assert opt.param_groups[1]["lr"] == pytest.approx(lr_at_step(t, A_MAX, A_MIN, TW, TC), abs=1e-12)
        opt.step()
        sch.step()


# --------------------------------------------------------------------------- gradient clipping
def test_clipped_gradient_norm_is_at_most_one():
    m = tiny_model()
    x = torch.randint(0, 50, (4, 16))
    (100.0 * cross_entropy_loss(m(x), x)).backward()  # make the gradient large
    pre = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    post = torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters()))
    assert pre > 1.0
    assert post <= 1.0 + 1e-5


def test_train_step_unscales_before_clipping():
    """With a loss scale of 1024 the *reported* norm must be the true one. If the gradients were
    clipped while still scaled, the norm would be ~1024x too big (and everything would be clipped)."""
    x = torch.randint(0, 50, (4, 16))
    y = torch.randint(0, 50, (4, 16))

    def one_step(scale):
        m = tiny_model(3)
        opt = build_optimizer(m, 1e-3)
        scaler = torch.amp.GradScaler("cpu", init_scale=scale, enabled=scale is not None) if scale else \
            torch.amp.GradScaler("cpu", enabled=False)
        loss, gn = train_step(m, opt, None, scaler, x, y, grad_clip=1e9)
        return gn, copy.deepcopy(m.state_dict())

    gn_plain, sd_plain = one_step(None)
    gn_scaled, sd_scaled = one_step(1024.0)
    assert gn_scaled == pytest.approx(gn_plain, rel=1e-3)
    for k in sd_plain:  # and the update itself is identical
        assert torch.allclose(sd_plain[k], sd_scaled[k], atol=1e-6), k


# --------------------------------------------------------------------------- mixed precision
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_autocast_loss_close_to_fp32(precision):
    m = tiny_model().eval()
    x = torch.randint(0, 50, (4, 16))
    y = torch.randint(0, 50, (4, 16))
    with torch.no_grad():
        ref = cross_entropy_loss(m(x), y).item()
        try:
            with make_autocast(precision, "cpu")():
                low = cross_entropy_loss(m(x).float(), y).item()
        except RuntimeError:
            pytest.skip(f"{precision} autocast not supported on this CPU")
    assert abs(low - ref) < 0.05


# --------------------------------------------------------------------------- batches
def test_get_batch_targets_are_inputs_shifted_by_one():
    data = np.arange(1000, dtype=np.uint16)  # data[i] == i, so the shift is easy to see
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = get_batch(data, 8, 32, rng=rng)
        assert x.shape == y.shape == (8, 32) and x.dtype == torch.int64
        assert torch.equal(y, x + 1)
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert (x[:, 1:] - x[:, :-1] == 1).all()  # contiguous slices


def test_get_batch_never_runs_past_the_end_and_can_reach_the_last_token():
    n, L = 20, 8
    data = np.arange(n, dtype=np.uint16)
    rng = np.random.default_rng(0)
    seen_last = False
    for _ in range(300):
        x, y = get_batch(data, 16, L, rng=rng)
        assert y.max() <= n - 1 and x.min() >= 0
        assert (y[:, -1] <= n - 1).all()
        seen_last |= bool((y.max() == n - 1).item())
    assert seen_last, "the final token should be reachable as a target"
    x, y = get_batch(np.arange(L + 1, dtype=np.uint16), 4, L)  # only one valid start
    assert x[0].tolist() == list(range(L)) and y[0].tolist() == list(range(1, L + 1))
    with pytest.raises(ValueError):
        get_batch(np.arange(L, dtype=np.uint16), 1, L)


def test_get_batch_works_on_memmap(tmp_path):
    arr = np.arange(500, dtype=np.uint16)
    np.save(tmp_path / "d.npy", arr)
    data = np.load(tmp_path / "d.npy", mmap_mode="r")
    x, y = get_batch(data, 4, 16, rng=np.random.default_rng(0))
    assert torch.equal(y, x + 1)


def test_fixed_eval_batches_are_identical_every_time():
    data = np.random.default_rng(0).integers(0, 50, 5000).astype(np.uint16)
    a = fixed_eval_batches(data, 3, 4, 16)
    b = fixed_eval_batches(data, 3, 4, 16)
    for (xa, ya), (xb, yb) in zip(a, b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert not torch.equal(a[0][0], a[1][0])  # but the batches differ from each other


# --------------------------------------------------------------------------- checkpoints
def make_batches(n, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 50, 4000).astype(np.uint16)
    return [get_batch(data, 4, 16, rng=rng) for _ in range(n)]


def setup(seed=0, scaler_enabled=False):
    m = tiny_model(seed)
    opt = build_optimizer(m, lr=1e-2)
    sch = build_scheduler(opt, 1e-2, 1e-3, 3, 30)
    scaler = torch.amp.GradScaler("cpu", enabled=scaler_enabled)
    return m, opt, sch, scaler


def run(m, opt, sch, scaler, batches):
    return [train_step(m, opt, sch, scaler, x, y)[0] for x, y in batches]


def test_checkpoint_resume_reproduces_losses(tmp_path):
    batches = make_batches(10)
    m, opt, sch, scaler = setup()
    run(m, opt, sch, scaler, batches[:5])
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, m, opt, sch, scaler, step=5)
    original = run(m, opt, sch, scaler, batches[5:])

    m2, opt2, sch2, scaler2 = setup(seed=123)  # different init: everything must come from the checkpoint
    ck = load_checkpoint(path, m2, opt2, sch2, scaler2)
    assert ck["step"] == 5
    resumed = run(m2, opt2, sch2, scaler2, batches[5:])
    assert max(abs(a - b) for a, b in zip(original, resumed)) < 1e-3
    assert max(abs(a - b) for a, b in zip(original, resumed)) < 1e-5  # in fact bit-for-bit close


def test_resume_test_is_sensitive_to_missing_optimizer_state(tmp_path):
    """Negative control: if we only restored the weights the losses would drift apart."""
    batches = make_batches(10)
    m, opt, sch, scaler = setup()
    run(m, opt, sch, scaler, batches[:5])
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, m, opt, sch, scaler, step=5)
    original = run(m, opt, sch, scaler, batches[5:])
    m2, opt2, sch2, scaler2 = setup(seed=123)
    m2.load_state_dict(torch.load(path, weights_only=False)["model"])  # weights only
    wrong = run(m2, opt2, sch2, scaler2, batches[5:])
    assert max(abs(a - b) for a, b in zip(original, wrong)) > 1e-3


def test_checkpoint_has_all_states(tmp_path):
    m, opt, sch, scaler = setup(scaler_enabled=False)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, m, opt, sch, scaler, step=7)
    ck = torch.load(path, weights_only=False)
    assert set(ck) >= {"model", "optimizer", "scheduler", "scaler", "step"} and ck["step"] == 7


# --------------------------------------------------------------------------- single-batch check
def test_overfits_a_single_batch():
    m = tiny_model()
    opt = build_optimizer(m, lr=3e-3, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    x, y = make_batches(1)[0]
    losses = [train_step(m, opt, None, scaler, x, y)[0] for _ in range(300)]
    assert losses[0] > 3.5  # ~ln(50) = 3.9 at the start
    assert losses[-1] < 0.1
    assert all(math.isfinite(l) for l in losses)


# --------------------------------------------------------------------------- train.py end to end
def _tiny_args(tmp_path, out, extra=()):
    rng = np.random.default_rng(0)
    for name in ("train", "val"):
        p = tmp_path / f"{name}.npy"
        if not p.exists():
            np.save(p, rng.integers(0, 50, 3000).astype(np.uint16))
    return ["--train-data", str(tmp_path / "train.npy"), "--val-data", str(tmp_path / "val.npy"),
            "--out-dir", str(tmp_path / out), "--vocab-size", "50", "--context-length", "16", "--d-model", "32",
            "--num-layers", "2", "--num-heads", "4", "--d-ff", "64", "--batch-size", "4", "--max-steps", "20",
            "--lr", "3e-3", "--eval-interval", "10", "--eval-batches", "2", "--final-eval-batches", "3",
            "--eval-batch-size", "4", "--log-interval", "5", "--ckpt-interval", "10", "--device", "cpu", *extra]


def _read_log(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_train_py_runs_and_resume_matches_uninterrupted_run(tmp_path):
    import train as train_script
    train_script.main(_tiny_args(tmp_path, "full"))
    train_script.main(_tiny_args(tmp_path, "part", ["--stop-after", "10"]))
    train_script.main(_tiny_args(tmp_path, "part", ["--resume"]))
    full, part = _read_log(tmp_path / "full" / "log.csv"), _read_log(tmp_path / "part" / "log.csv")
    assert [r["step"] for r in full] == [r["step"] for r in part]
    for a, b in zip(full, part):
        assert float(a["train_loss"]) == pytest.approx(float(b["train_loss"]), abs=1e-4), (a, b)
        assert a["lr"] == b["lr"]
        if a["val_loss"]:
            assert float(a["val_loss"]) == pytest.approx(float(b["val_loss"]), abs=1e-4)
    assert set(full[0]) == {"step", "train_loss", "val_loss", "lr", "grad_norm", "elapsed_s", "tokens_seen"}
    assert (tmp_path / "full" / "summary.json").exists()
    # last step uses exactly alpha_min = 0.1 * lr
    assert float(full[-1]["lr"]) == pytest.approx(3e-4, rel=1e-3)


def test_train_py_overfit_single_batch_mode(tmp_path):
    import train as train_script
    train_script.main(_tiny_args(tmp_path, "overfit", ["--overfit-single-batch", "--max-steps", "300",
                                                        "--lr", "3e-3", "--log-interval", "50"]))
    rows = _read_log(tmp_path / "overfit" / "log.csv")
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"]) * 0.2
