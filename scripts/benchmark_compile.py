"""Time training steps with and without torch.compile (Part 5). Writes a JSON with the numbers.

    python scripts/benchmark_compile.py --steps 30 --batch-size 32 --out report/compile_benchmark.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minilm.model import TransformerLM
from minilm.train import build_optimizer, build_scheduler, get_batch, train_step


def bench(compile_model: bool, steps: int, warmup: int, batch_size: int, device: str):
    torch.manual_seed(0)
    model = TransformerLM(10_000, 256, 512, 4, 16, 1344, 10_000.0).to(device)
    opt = build_optimizer(model, 1e-3)
    sch = build_scheduler(opt, 1e-3, 1e-4, 10, steps + warmup)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    run_model = torch.compile(model) if compile_model else model
    data = np.random.default_rng(0).integers(0, 10_000, 200_000).astype(np.uint16)
    rng = np.random.default_rng(0)
    t_compile = None
    t0 = time.perf_counter()
    for i in range(warmup):
        x, y = get_batch(data, batch_size, 256, device, rng=rng)
        train_step(run_model, opt, sch, scaler, x, y)
        if i == 0:
            t_compile = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(steps):
        x, y = get_batch(data, batch_size, 256, device, rng=rng)
        train_step(run_model, opt, sch, scaler, x, y)
    dt = (time.perf_counter() - t1) / steps
    return {"sec_per_step": dt, "first_step_sec": t_compile}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()
    res = {"device": args.device, "torch": torch.__version__, "threads": torch.get_num_threads(),
           "batch_size": args.batch_size, "steps_timed": args.steps}
    res["eager"] = bench(False, args.steps, args.warmup, args.batch_size, args.device)
    print("eager", res["eager"], flush=True)
    if not args.skip_compile:
        try:
            res["compiled"] = bench(True, args.steps, args.warmup, args.batch_size, args.device)
            res["speedup"] = res["eager"]["sec_per_step"] / res["compiled"]["sec_per_step"]
            print("compiled", res["compiled"], "speedup", res["speedup"], flush=True)
        except Exception as e:  # e.g. no C++ compiler on Windows CPU
            res["compile_error"] = f"{type(e).__name__}: {str(e)[:500]}"
            print("torch.compile failed:", res["compile_error"], flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
