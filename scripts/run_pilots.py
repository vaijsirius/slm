"""Short learning-rate pilot runs (Part 6): same setup as the main run, a few steps each.

    python scripts/run_pilots.py --train-data data/train.npy --val-data data/valid.npy --runs-dir runs \
        --lrs 3e-4 1e-3 3e-3 1e-2 3e-2 --steps 100 --batch-size 16
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    for lr in a.lrs:
        out = f"{a.runs_dir}/pilot_lr{lr:g}"
        cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--train-data", a.train_data, "--val-data", a.val_data,
               "--out-dir", out, "--lr", str(lr), "--max-steps", str(a.steps), "--batch-size", str(a.batch_size),
               "--log-interval", "1", "--eval-interval", str(max(1, a.steps // 4)), "--eval-batches", "2",
               "--final-eval-batches", "8", "--eval-batch-size", "16", "--ckpt-interval", "100000", "--seed", "0"]
        if a.device:
            cmd += ["--device", a.device]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
