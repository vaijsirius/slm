"""Question 11: CPU generation speed with and without a KV cache, and how much work a cache avoids.

    python scripts/benchmark_inference.py --model-dir release --new-tokens 256 --out report/inference_benchmark.json
"""
import argparse
import json
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import load_release
from minilm.generate import generate_ids


def token_work(prompt_len, new_tokens, ctx, cached):
    """Tokens pushed through the network to generate ``new_tokens`` (mirrors generate_ids)."""
    total, length = 0, prompt_len
    past = False
    for _ in range(new_tokens):
        if cached and length <= ctx:
            total += length if not past else 1
            past = True
        else:
            total += min(length, ctx)
            past = False
        length += 1
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="release")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    model, tok = load_release(a.model_dir)
    n_prompt = len(tok.encode(a.prompt))
    res = {"machine": platform.processor(), "platform": platform.platform(), "torch": torch.__version__,
           "threads": torch.get_num_threads(), "prompt_tokens": n_prompt, "new_tokens": a.new_tokens}
    for cached in (True, False):
        times = []
        for r in range(a.repeats):
            gen = torch.Generator().manual_seed(r)
            t0 = time.perf_counter()
            ids = generate_ids(model, tok, a.prompt, a.new_tokens, 0.8, 0.9, gen, use_cache=cached, eos_token=None)
            times.append(time.perf_counter() - t0)
            assert len(ids) == a.new_tokens
        best = min(times)
        res["kv_cache" if cached else "no_cache"] = {"seconds": times, "best_tokens_per_second": a.new_tokens / best}
        print(("with" if cached else "without"), "KV cache:", [f"{t:.2f}s" for t in times],
              f"-> {a.new_tokens / best:.1f} tok/s (best)", flush=True)
    ctx = model.context_length
    w_no, w_yes = token_work(n_prompt, a.new_tokens, ctx, False), token_work(n_prompt, a.new_tokens, ctx, True)
    res["token_passes_no_cache"] = w_no
    res["token_passes_with_cache"] = w_yes
    res["fraction_of_token_work_repeated"] = 1 - w_yes / w_no
    res["measured_speedup"] = res["no_cache"]["best_tokens_per_second"] and res["kv_cache"]["best_tokens_per_second"] / res["no_cache"]["best_tokens_per_second"]
    print(json.dumps({k: v for k, v in res.items() if k not in ("no_cache", "kv_cache")}, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
