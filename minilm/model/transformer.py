"""The Transformer block and the full language model (Part 4)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn

from .layers import CausalSelfAttention, FeedForward, RoPE, SiLUFeedForward


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10_000.0
    # --- optional switches for the architecture experiments (defaults = the assignment's model)
    norm: str = "pre"        # "pre" | "post" | "none"
    use_rope: bool = True
    ffn: str = "swiglu"      # "swiglu" | "silu"
    tie_weights: bool = False

    CORE = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")

    def to_json(self, path: str, core_only: bool = True) -> None:
        d = asdict(self)
        if core_only and (self.norm, self.use_rope, self.ffn, self.tie_weights) == ("pre", True, "swiglu", False):
            d = {k: d[k] for k in self.CORE}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            d = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class TransformerBlock(nn.Module):
    """Pre-norm block:  z = x + Attn(RMSNorm(x));  y = z + FFN(RMSNorm(z))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None,
                 norm: str = "pre", ffn: str = "swiglu"):
        super().__init__()
        if norm not in ("pre", "post", "none"):
            raise ValueError(norm)
        self.norm_style = norm
        self.attn = CausalSelfAttention(d_model, num_heads, rope)
        self.ffn = FeedForward(d_model, d_ff) if ffn == "swiglu" else SiLUFeedForward(d_model, d_ff)
        if norm != "none":
            self.ln1 = nn.RMSNorm(d_model, eps=1e-5)
            self.ln2 = nn.RMSNorm(d_model, eps=1e-5)

    def forward(self, x: torch.Tensor, past=None, use_cache: bool = False):
        present = None
        if self.norm_style == "pre":
            a = self.attn(self.ln1(x), past, use_cache)
            if use_cache:
                a, present = a
            z = x + a
            y = z + self.ffn(self.ln2(z))
        elif self.norm_style == "post":  # experiment: z = Norm(x + Attn(x)); y = Norm(z + FFN(z))
            a = self.attn(x, past, use_cache)
            if use_cache:
                a, present = a
            z = self.ln1(x + a)
            y = self.ln2(z + self.ffn(z))
        else:  # no normalisation at all
            a = self.attn(x, past, use_cache)
            if use_cache:
                a, present = a
            z = x + a
            y = z + self.ffn(z)
        return (y, present) if use_cache else y


class TransformerLM(nn.Module):
    """Embedding -> N blocks -> RMSNorm -> Linear to the vocabulary. Returns (batch, seq, vocab) logits."""

    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int,
                 d_ff: int, rope_theta: float, *, norm: str = "pre", use_rope: bool = True,
                 ffn: str = "swiglu", tie_weights: bool = False):
        super().__init__()
        self.config = ModelConfig(vocab_size, context_length, d_model, num_layers, num_heads, d_ff,
                                  rope_theta, norm, use_rope, ffn, tie_weights)
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        rope = RoPE(rope_theta, d_model // num_heads, context_length) if use_rope else None
        self.rope = rope
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope, norm, ffn) for _ in range(num_layers)])
        self.ln_final = nn.RMSNorm(d_model, eps=1e-5) if norm != "none" else nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "TransformerLM":
        return cls(cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff,
                   cfg.rope_theta, norm=cfg.norm, use_rope=cfg.use_rope, ffn=cfg.ffn, tie_weights=cfg.tie_weights)

    def forward(self, tokens: torch.Tensor, past: list | None = None, use_cache: bool = False):
        """tokens: (batch, seq) of ids -> logits (batch, seq, vocab). With ``use_cache`` also returns
        the per-layer (k, v) list to pass back as ``past`` for the next step."""
        x = self.token_embeddings(tokens)
        presents = []
        for i, layer in enumerate(self.layers):
            out = layer(x, None if past is None else past[i], use_cache)
            if use_cache:
                x, p = out
                presents.append(p)
            else:
                x = out
        logits = self.lm_head(self.ln_final(x))
        return (logits, presents) if use_cache else logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
