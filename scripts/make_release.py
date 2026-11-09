"""Package the trained model into release/ (weights only + config + tokenizer files) - Part 8.

    python scripts/make_release.py --checkpoint runs/main/final_model.pt --tokenizer-dir artifacts/tokenizer \
        --release-dir release
"""
import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minilm.model import ModelConfig, TransformerLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="final_model.pt (a state_dict) or a last.pt checkpoint")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--release-dir", default="release")
    ap.add_argument("--args-json", default=None, help="args.json written by train.py (default: next to the checkpoint)")
    a = ap.parse_args()
    import json
    args_json = a.args_json or os.path.join(os.path.dirname(a.checkpoint), "args.json")
    with open(args_json) as f:
        ar = json.load(f)
    cfg = ModelConfig(ar["vocab_size"], ar["context_length"], ar["d_model"], ar["num_layers"], ar["num_heads"],
                      ar["d_ff"], ar["rope_theta"])
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if "model" in state:  # a full training checkpoint: keep only the weights
        state = state["model"]
    model = TransformerLM.from_config(cfg)
    model.load_state_dict(state)  # fails loudly if config and weights disagree
    os.makedirs(a.release_dir, exist_ok=True)
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, os.path.join(a.release_dir, "model.pt"))
    cfg.to_json(os.path.join(a.release_dir, "config.json"))
    for fn in ("vocab.json", "merges.txt"):
        shutil.copy(os.path.join(a.tokenizer_dir, fn), os.path.join(a.release_dir, fn))
    for fn in sorted(os.listdir(a.release_dir)):
        print(f"{fn:14s} {os.path.getsize(os.path.join(a.release_dir, fn)) / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
