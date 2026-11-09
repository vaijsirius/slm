"""Training pieces (Part 5): loss, optimizer groups, LR schedule, batches, checkpoints."""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------- loss
def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (batch, seq, vocab), targets (batch, seq) -> mean cross entropy over batch*seq tokens."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


# ----------------------------------------------------------------------------------- optimizer
def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Two groups that together hold every parameter exactly once.

    * decay group    : weight *matrices* (ndim >= 2: attention/FFN projections, embedding, output layer)
    * no-decay group : the RMSNorm weight vectors (ndim == 1)
    """
    decay, no_decay, seen = [], [], set()
    for p in model.parameters():  # parameters() de-duplicates shared (tied) tensors
        if id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model: nn.Module, lr: float, betas=(0.9, 0.95), weight_decay: float = 0.1,
                    eps: float = 1e-8) -> torch.optim.AdamW:
    return torch.optim.AdamW(build_param_groups(model, weight_decay), lr=lr, betas=betas, eps=eps)


# ------------------------------------------------------------------------------- LR schedule
def lr_at_step(t: int, alpha_max: float, alpha_min: float, warmup_steps: int, cosine_end: int) -> float:
    """Linear warm-up to alpha_max, cosine decay to alpha_min at step ``cosine_end``, then constant."""
    if t < warmup_steps:
        return t / warmup_steps * alpha_max
    if t <= cosine_end:
        if cosine_end == warmup_steps:
            return alpha_min
        progress = (t - warmup_steps) / (cosine_end - warmup_steps)
        return alpha_min + 0.5 * (1 + math.cos(math.pi * progress)) * (alpha_max - alpha_min)
    return alpha_min


def build_scheduler(optimizer, alpha_max: float, alpha_min: float, warmup_steps: int, cosine_end: int):
    """LambdaLR whose multiplier is lr_at_step(t) / alpha_max (the optimizer's base lr must be alpha_max)."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda t: lr_at_step(t, alpha_max, alpha_min, warmup_steps, cosine_end) / alpha_max)


# ------------------------------------------------------------------------------------ batches
def get_batch(data, batch_size: int, context_length: int, device="cpu",
              rng: np.random.Generator | None = None, starts: np.ndarray | None = None):
    """Random (inputs, targets): inputs = data[i : i+L], targets = data[i+1 : i+L+1] (shifted by one).

    Start positions are drawn so that i + L + 1 <= len(data): no slice ever runs off the end.
    ``starts`` lets the caller pass explicit start indices (used for the fixed validation batches).
    """
    n = len(data)
    if n < context_length + 1:
        raise ValueError("data shorter than context_length + 1")
    if starts is None:
        rng = rng if rng is not None else np.random.default_rng()
        starts = rng.integers(0, n - context_length, size=batch_size)  # high is exclusive
    x = np.stack([np.asarray(data[i: i + context_length]) for i in starts]).astype(np.int64)
    y = np.stack([np.asarray(data[i + 1: i + context_length + 1]) for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def fixed_eval_batches(data, num_batches: int, batch_size: int, context_length: int, device="cpu", seed: int = 1234):
    """The *same* validation batches every time (fixed seed) so val losses are comparable."""
    rng = np.random.default_rng(seed)
    return [get_batch(data, batch_size, context_length, device, rng=rng) for _ in range(num_batches)]


@torch.no_grad()
def evaluate(model: nn.Module, batches, autocast_ctx=None) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for x, y in batches:
        if autocast_ctx is not None:
            with autocast_ctx():
                logits = model(x)
        else:
            logits = model(x)
        total += cross_entropy_loss(logits.float(), y).item()
    if was_training:
        model.train()
    return total / len(batches)


# ------------------------------------------------------------------------------------ one step
def make_autocast(precision: str, device_type: str):
    """None for fp32, else a zero-arg factory for ``torch.autocast`` (fp16 or bf16)."""
    if precision == "fp32":
        return None
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return lambda: torch.autocast(device_type, dtype=dtype)


def train_step(model, optimizer, scheduler, scaler, x, y, grad_clip: float | None = 1.0, autocast_ctx=None):
    """One optimisation step; returns (loss, grad_norm_before_clipping).

    Order matters: backward on the *scaled* loss -> ``scaler.unscale_`` -> clip -> step. Clipping the
    still-scaled gradients would compare a (scale x too big) norm against the threshold.
    """
    optimizer.zero_grad(set_to_none=True)
    if autocast_ctx is not None:
        with autocast_ctx():
            logits = model(x)
    else:
        logits = model(x)
    loss = cross_entropy_loss(logits.float(), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # no-op when the scaler is disabled (fp32)
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    scaler.step(optimizer)
    scaler.update()
    if scheduler is not None:
        scheduler.step()
    return loss.item(), float(grad_norm)


# --------------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, model, optimizer, scheduler, scaler, step: int, extra: dict | None = None) -> None:
    """Save model/optimizer/scheduler/scaler state + step. ``model`` may be a torch.compile wrapper."""
    raw = getattr(model, "_orig_mod", model)
    ckpt = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save cannot corrupt the last good checkpoint


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    raw = getattr(model, "_orig_mod", model)
    raw.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
