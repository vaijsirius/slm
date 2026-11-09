"""Part 6 correctness checks on the real model: single-batch overfit, and stop-and-resume.

    python scripts/run_checks.py --train-data data/train.npy --val-data data/valid.npy --runs-dir runs
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(args):
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), *args]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--overfit-batch-size", type=int, default=4)
    ap.add_argument("--resume-steps", type=int, default=60)
    ap.add_argument("--resume-batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    run(["--train-data", a.train_data, "--out-dir", f"{a.runs_dir}/check_overfit", "--overfit-single-batch",
         "--batch-size", str(a.overfit_batch_size), "--max-steps", str(a.overfit_steps), "--lr", "1e-3",
         "--log-interval", "5", "--ckpt-interval", "100000", "--device", a.device])

    common = ["--train-data", a.train_data, "--val-data", a.val_data, "--batch-size", str(a.resume_batch_size),
              "--max-steps", str(a.resume_steps), "--lr", "2e-3", "--log-interval", "2", "--eval-interval", "10",
              "--eval-batches", "2", "--final-eval-batches", "2", "--eval-batch-size", "8", "--ckpt-interval", "10",
              "--device", a.device]
    run(common + ["--out-dir", f"{a.runs_dir}/check_resume_full"])
    run(common + ["--out-dir", f"{a.runs_dir}/check_resume_split", "--stop-after", str(a.resume_steps // 2)])
    run(common + ["--out-dir", f"{a.runs_dir}/check_resume_split", "--resume"])


if __name__ == "__main__":
    main()
