"""Training script (Part 6).

Every model / optimizer setting is a command-line flag (or comes from ``--config file.json``, whose keys
are the flag names with underscores). Example (the main run on a GPU):

    python train.py --train-data data/train.npy --val-data data/valid.npy --out-dir runs/main \
        --lr 3e-3 --max-steps 5000 --precision fp16

Sanity-check modes:  --overfit-single-batch   (loss must go to ~0)       --resume  (continue a run)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from minilm.model import ModelConfig, TransformerLM
from minilm.train import (build_optimizer, build_scheduler, evaluate, fixed_eval_batches, get_batch,
                          load_checkpoint, lr_at_step, make_autocast, save_checkpoint, train_step)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file with default values for any of these flags")
    g = p.add_argument_group("data / output")
    g.add_argument("--train-data", required=False, help=".npy of uint16 token ids (opened with mmap)")
    g.add_argument("--val-data", required=False)
    g.add_argument("--out-dir", default="runs/run")
    g = p.add_argument_group("model")
    g.add_argument("--vocab-size", type=int, default=10_000)
    g.add_argument("--context-length", type=int, default=256)
    g.add_argument("--d-model", type=int, default=512)
    g.add_argument("--num-layers", type=int, default=4)
    g.add_argument("--num-heads", type=int, default=16)
    g.add_argument("--d-ff", type=int, default=1344)
    g.add_argument("--rope-theta", type=float, default=10_000.0)
    g.add_argument("--norm", default="pre", choices=["pre", "post", "none"])
    g.add_argument("--no-rope", action="store_true", help="experiment: remove position information")
    g.add_argument("--ffn", default="swiglu", choices=["swiglu", "silu"])
    g.add_argument("--tie-weights", action="store_true")
    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=3e-3, help="peak learning rate (alpha_max)")
    g.add_argument("--min-lr-ratio", type=float, default=0.1, help="alpha_min = ratio * lr")
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.95)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--warmup-frac", type=float, default=0.05)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--max-steps", type=int, default=5000)
    g = p.add_argument_group("evaluation / logging / checkpoints")
    g.add_argument("--eval-interval", type=int, default=250)
    g.add_argument("--eval-batches", type=int, default=8, help="fixed val batches used at every interval eval")
    g.add_argument("--final-eval-batches", type=int, default=40, help="fixed val batches for the final eval")
    g.add_argument("--eval-batch-size", type=int, default=32)
    g.add_argument("--log-interval", type=int, default=10)
    g.add_argument("--ckpt-interval", type=int, default=500)
    g.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt")
    g = p.add_argument_group("system")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--precision", default=None, choices=["fp32", "fp16", "bf16"],
                   help="default: fp16 on cuda, fp32 on cpu")
    g.add_argument("--compile", action="store_true", help="torch.compile the model")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--overfit-single-batch", action="store_true", help="train on one fixed batch repeatedly")
    g.add_argument("--stop-after", type=int, default=None,
                   help="(testing) stop after this many steps but keep the full schedule, to test --resume")
    return p


def parse_args(argv=None):
    p = get_parser()
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            p.set_defaults(**{k.replace("-", "_"): v for k, v in json.load(f).items()})
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device
    precision = args.precision or ("fp16" if device.startswith("cuda") else "fp32")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r") if args.val_data else None
    assert train_data.dtype == np.uint16

    cfg = ModelConfig(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads,
                      args.d_ff, args.rope_theta, args.norm, not args.no_rope, args.ffn, args.tie_weights)
    model = TransformerLM.from_config(cfg).to(device)
    n_params = model.num_parameters()
    optimizer = build_optimizer(model, args.lr, (args.beta1, args.beta2), args.weight_decay)
    warmup = max(1, int(round(args.warmup_frac * args.max_steps)))
    alpha_min = args.min_lr_ratio * args.lr
    cosine_end = args.max_steps - 1  # lowest lr is reached exactly at the last step
    scheduler = build_scheduler(optimizer, args.lr, alpha_min, warmup, cosine_end)
    use_scaler = precision == "fp16" and device_type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
    autocast_ctx = make_autocast(precision, device_type) if precision != "fp32" else None

    ckpt_path = os.path.join(args.out_dir, "last.pt")
    log_path = os.path.join(args.out_dir, "log.csv")
    start_step, elapsed_before = 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, map_location=device)
        start_step = ckpt["step"]
        elapsed_before = ckpt["extra"].get("elapsed", 0.0)
        if "rng_state" in ckpt["extra"]:
            rng.bit_generator.state = ckpt["extra"]["rng_state"]
        print(f"resumed from {ckpt_path} at step {start_step}", flush=True)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "elapsed_s", "tokens_seen"])
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "precision": precision, "n_params": n_params}, f, indent=2)

    run_model = torch.compile(model) if args.compile else model
    val_batches = None
    if val_data is not None:
        total = max(args.eval_batches, args.final_eval_batches)
        val_batches = fixed_eval_batches(val_data, total, args.eval_batch_size, args.context_length, device)
    fixed_batch = get_batch(train_data, args.batch_size, args.context_length, device,
                            rng=np.random.default_rng(args.seed + 1)) if args.overfit_single_batch else None

    print(f"params={n_params:,} device={device} precision={precision} steps={args.max_steps} "
          f"lr={args.lr:g} warmup={warmup} compile={args.compile}", flush=True)
    model.train()
    t_start = time.time()
    run_loss, run_n, last_val = 0.0, 0, float("nan")
    last_step = args.max_steps if args.stop_after is None else min(args.max_steps, args.stop_after)
    log_f = open(log_path, "a", newline="")
    writer = csv.writer(log_f)
    try:
        for step in range(start_step, last_step):
            lr_now = optimizer.param_groups[0]["lr"]
            x, y = fixed_batch if fixed_batch is not None else get_batch(
                train_data, args.batch_size, args.context_length, device, rng=rng)
            loss, grad_norm = train_step(run_model, optimizer, scheduler, scaler, x, y, args.grad_clip, autocast_ctx)
            if not math.isfinite(loss):
                print(f"step {step}: non-finite loss {loss}", flush=True)
            run_loss += loss
            run_n += 1
            done = step + 1
            elapsed = elapsed_before + time.time() - t_start
            is_last = done == args.max_steps
            do_eval = val_batches is not None and (done % args.eval_interval == 0 or is_last)
            if do_eval:
                n = args.final_eval_batches if is_last else args.eval_batches
                last_val = evaluate(model, val_batches[:n], autocast_ctx)
            if done % args.log_interval == 0 or do_eval or is_last or step == 0:
                tr = run_loss / run_n
                run_loss, run_n = 0.0, 0
                writer.writerow([step + 1, f"{tr:.5f}", f"{last_val:.5f}" if do_eval else "", f"{lr_now:.3e}",
                                 f"{grad_norm:.4f}", f"{elapsed:.1f}", done * args.batch_size * args.context_length])
                log_f.flush()
                print(f"step {done:5d} | train {tr:.4f} | val {last_val if do_eval else float('nan'):.4f} | "
                      f"lr {lr_now:.2e} | gnorm {grad_norm:.3f} | {elapsed:7.1f}s", flush=True)
            if done % args.ckpt_interval == 0 or is_last or done == last_step:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, done,
                                {"elapsed": elapsed, "rng_state": rng.bit_generator.state})
    finally:
        log_f.close()

    if last_step == args.max_steps:
        summary = {"steps": args.max_steps, "n_params": n_params, "final_train_loss": tr,
                   "elapsed_s": elapsed, "lr": args.lr, "precision": precision, "device": device}
        if val_batches is not None:
            summary.update(final_val_loss=last_val, final_val_ppl=math.exp(last_val),
                           val_tokens=args.final_eval_batches * args.eval_batch_size * args.context_length)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pt"))
        print(json.dumps(summary, indent=2), flush=True)
    return model


if __name__ == "__main__":
    main()
