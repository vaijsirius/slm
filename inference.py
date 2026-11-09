"""Generate a story from the packaged model in ``release/`` - CPU is fine, no training code needed.

    python inference.py --model-dir release --prompt "Once upon a time" \
        --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
"""
import argparse
import os
import sys
import time

import torch

from minilm.generate import generate_ids
from minilm.model import ModelConfig, TransformerLM
from minilm.tokenizer import Tokenizer


def load_release(model_dir: str, device: str = "cpu"):
    cfg = ModelConfig.from_json(os.path.join(model_dir, "config.json"))
    model = TransformerLM.from_config(cfg)
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    tok = Tokenizer.from_files(os.path.join(model_dir, "vocab.json"), os.path.join(model_dir, "merges.txt"),
                               ["<|endoftext|>"])
    return model, tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="release")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-kv-cache", action="store_true", help="recompute the whole context every step")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # stories may contain non-ASCII text; Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model, tok = load_release(args.model_dir, args.device)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)  # sampling always happens on the CPU
    t0 = time.perf_counter()
    new_ids = generate_ids(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_p, gen,
                           use_cache=not args.no_kv_cache)
    dt = time.perf_counter() - t0
    print(args.prompt + tok.decode(new_ids))
    print(f"\nGenerated {len(new_ids)} tokens in {dt:.2f}s ({len(new_ids) / dt:.1f} tokens/s)")


if __name__ == "__main__":
    main()
