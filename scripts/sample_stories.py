"""Generate sample stories (>= 256 tokens, prompt "Once upon a time") with two sampling settings (Part 7).

    python scripts/sample_stories.py --model-dir release --out report/stories.md
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import load_release
from minilm.generate import generate_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="release")
    ap.add_argument("--out", default="report/stories.md")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--min-tokens", type=int, default=256)
    ap.add_argument("--settings", default="0.7:0.9,1.0:0.95", help="comma separated temperature:top_p")
    ap.add_argument("--per-setting", type=int, default=None, help="stories per setting (default 2 and 1)")
    ap.add_argument("--max-seeds", type=int, default=150)
    a = ap.parse_args()
    model, tok = load_release(a.model_dir)
    settings = [tuple(float(x) for x in s.split(":")) for s in a.settings.split(",")]
    counts = [2, 1] if a.per_setting is None else [a.per_setting] * len(settings)
    lines = [f"# Sample stories (prompt: \"{a.prompt}\")\n"]
    for (T, p), n_stories in zip(settings, counts):
        for k in range(n_stories):
            seed = 0
            while True:  # keep the first seed that gives a story of at least min-tokens that ends naturally or hits 320
                gen = torch.Generator().manual_seed(1000 * k + seed)
                ids = generate_ids(model, tok, a.prompt, 320, T, p, gen)
                if len(ids) >= a.min_tokens or seed >= a.max_seeds:
                    break
                seed += 1
            n_prompt = len(tok.encode(a.prompt))
            lines += [f"## temperature={T}, top_p={p}, seed={1000 * k + seed}  ({n_prompt + len(ids)} tokens incl. prompt)\n",
                      a.prompt + tok.decode(ids), ""]
            print(f"T={T} p={p} seed={1000 * k + seed}: {len(ids)} new tokens", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
