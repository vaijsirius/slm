"""Text generation (Part 7): temperature + top-p (nucleus) sampling, optional KV cache."""
from __future__ import annotations

import torch

EOT = "<|endoftext|>"


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0,
                generator: torch.Generator | None = None) -> int:
    """Pick the next token id from a 1-D ``logits`` vector.

    temperature <= 0 means greedy (argmax). Otherwise: probs = softmax(logits / T); with top_p < 1 keep the
    smallest set of most-likely tokens whose probabilities add up to at least p, rescale them to sum to 1,
    and sample from them.
    """
    logits = logits.float()
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = (cum - sorted_p) < top_p  # a token stays if the mass *before* it is still < p
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        sorted_p = sorted_p / sorted_p.sum()
        choice = torch.multinomial(sorted_p, 1, generator=generator)
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.no_grad()
def generate_ids(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0,
                 top_p: float = 1.0, generator: torch.Generator | None = None, use_cache: bool = True,
                 eos_token: str = EOT) -> list[int]:
    """Return the newly generated token ids (the end-of-text token itself is not included)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ctx = model.context_length
    eos_id = tokenizer.special_ids.get(eos_token)
    ids = tokenizer.encode(prompt)
    if not ids:  # the model needs at least one token to condition on
        ids = [eos_id] if eos_id is not None else [0]
    new: list[int] = []
    past = None
    for _ in range(max_new_tokens):
        if use_cache and len(ids) <= ctx:
            if past is None:  # prefill with the whole prompt (last `ctx` tokens at most)
                x = torch.tensor([ids[-ctx:]], device=device)
            else:  # one new token, keys/values of everything before it come from the cache
                x = torch.tensor([[ids[-1]]], device=device)
            logits, past = model(x, past, use_cache=True)
        else:  # sliding window: feed only the last `ctx` tokens, recompute everything
            past = None
            logits = model(torch.tensor([ids[-ctx:]], device=device))
        nxt = sample_next(logits[0, -1].cpu(), temperature, top_p, generator)
        if eos_id is not None and nxt == eos_id:
            break
        ids.append(nxt)
        new.append(nxt)
    if was_training:
        model.train()
    return new


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0,
             top_p: float = 1.0, generator: torch.Generator | None = None, use_cache: bool = True) -> str:
    """Prompt + continuation as text."""
    new = generate_ids(model, tokenizer, prompt, max_new_tokens, temperature, top_p, generator, use_cache)
    return prompt + tokenizer.decode(new)
