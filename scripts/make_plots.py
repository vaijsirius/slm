"""Make every figure used in the writeup from the CSV logs written by train.py.

    python scripts/make_plots.py --runs-dir runs --out-dir report/figures
"""
import argparse
import csv
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {k: [] for k in ("step", "train", "val_step", "val", "lr", "gn", "t")}
    for r in rows:
        out["step"].append(int(r["step"]))
        out["train"].append(float(r["train_loss"]))
        out["lr"].append(float(r["lr"]))
        out["gn"].append(float(r["grad_norm"]))
        out["t"].append(float(r["elapsed_s"]))
        if r["val_loss"]:
            out["val_step"].append(int(r["step"]))
            out["val"].append(float(r["val_loss"]))
    return out


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def plot_overfit(runs, out):
    p = f"{runs}/check_overfit/log.csv"
    if not os.path.exists(p):
        return
    d = read(p)
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.9))
    ax[0].semilogy(d["step"], d["train"])
    ax[0].set(xlabel="step", ylabel="loss (log scale)", title="Single-batch overfit")
    ax[1].plot(d["step"], d["gn"])
    ax[1].set(xlabel="step", ylabel="grad norm", title="Gradient norm")
    save(fig, out, "fig_single_batch.png")


def plot_resume(runs, out):
    a, b = f"{runs}/check_resume_full/log.csv", f"{runs}/check_resume_split/log.csv"
    if not (os.path.exists(a) and os.path.exists(b)):
        return
    full, split = read(a), read(b)
    args = json.load(open(f"{runs}/check_resume_split/args.json"))
    half = args["max_steps"] // 2
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.9))
    ax[0].plot(full["step"], full["train"], lw=3, alpha=0.5, label="uninterrupted")
    ax[0].plot(split["step"], split["train"], "--", label=f"stopped at {half}, resumed")
    ax[0].axvline(half, color="gray", ls=":")
    ax[0].set(xlabel="step", ylabel="train loss", title="Resume check: train loss")
    ax[0].legend(fontsize=7)
    ax[1].plot(full["val_step"], full["val"], "o-", lw=3, alpha=0.5, label="uninterrupted")
    ax[1].plot(split["val_step"], split["val"], "x--", label="resumed")
    ax[1].axvline(half, color="gray", ls=":")
    ax[1].set(xlabel="step", ylabel="val loss", title="Resume check: val loss")
    ax[1].legend(fontsize=7)
    save(fig, out, "fig_resume.png")
    n = min(len(full["train"]), len(split["train"]))
    diff = max(abs(x - y) for x, y in zip(full["train"][:n], split["train"][:n]))
    print(f"resume: max |train-loss difference| over {n} logged points = {diff:.2e}")
    return diff


def pilot_table(runs):
    rows = []
    for d in sorted(glob.glob(f"{runs}/pilot_lr*"), key=lambda p: float(p.split("pilot_lr")[1])):
        lr = float(d.split("pilot_lr")[1])
        s = read(f"{d}/log.csv")
        summ = json.load(open(f"{d}/summary.json")) if os.path.exists(f"{d}/summary.json") else {}
        tail = [x for x in s["train"][-10:] if math.isfinite(x)]
        rows.append({"lr": lr, "first_loss": s["train"][0], "end_train_loss": sum(tail) / max(1, len(tail)),
                     "end_val_loss": summ.get("final_val_loss", float("nan")), "max_gnorm": max(s["gn"]),
                     "max_gnorm_after_warmup": max(s["gn"][10:] or [float("nan")]),
                     "any_nonfinite": any(not math.isfinite(x) for x in s["train"])})
    return rows


def plot_pilots(runs, out):
    dirs = sorted(glob.glob(f"{runs}/pilot_lr*"), key=lambda p: float(p.split("pilot_lr")[1]))
    if not dirs:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for d in dirs:
        lr = float(d.split("pilot_lr")[1])
        s = read(f"{d}/log.csv")
        ax[0].plot(s["step"], s["train"], label=f"lr={lr:g}")
        ax[1].semilogy(s["step"], s["gn"], label=f"lr={lr:g}")
    ax[0].set(xlabel="step", ylabel="train loss", title="LR pilots: training loss", ylim=(2, 12))
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="step", ylabel="grad norm (before clipping)", title="Gradient norm")
    rows = pilot_table(runs)
    ax[2].bar([f"{r['lr']:g}" for r in rows], [r["end_val_loss"] for r in rows])
    ax[2].set(xlabel="peak learning rate", ylabel="end-of-pilot val loss", title="Final val loss", ylim=(0, 10))
    save(fig, out, "fig_lr_pilots.png")
    with open(os.path.join(out, "pilot_table.json"), "w") as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(r)


def plot_main(runs, out, name="main", extra=None, fname="fig_main_run.png"):
    p = f"{runs}/{name}/log.csv"
    if not os.path.exists(p):
        return
    d = read(p)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    series = [(name, d)] + ([(extra, read(f"{runs}/{extra}/log.csv"))] if extra and os.path.exists(f"{runs}/{extra}/log.csv") else [])
    for label, s in series:
        ax[0].plot(s["step"], s["train"], label=f"{label} train", alpha=0.8)
        ax[0].plot(s["val_step"], s["val"], "o-", ms=3, label=f"{label} val")
        ax[1].plot([t / 60 for t in s["t"]], s["train"], alpha=0.8)
        ax[1].plot([s["t"][s["step"].index(v)] / 60 for v in s["val_step"]], s["val"], "o-", ms=3)
    ax[0].set(xlabel="step", ylabel="loss", title="Loss vs step", ylim=(1.5, 6))
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="wall-clock minutes (CPU)", ylabel="loss", title="Loss vs time", ylim=(1.5, 6))
    ax[2].plot(d["step"], d["lr"], color="tab:green")
    ax[2].set(xlabel="step", ylabel="learning rate", title="LR schedule (twin axis: grad norm)")
    ax2 = ax[2].twinx()
    ax2.plot(d["step"], d["gn"], color="tab:red", alpha=0.5)
    ax2.set_ylabel("grad norm", color="tab:red")
    fig.subplots_adjust(wspace=0.45)
    save(fig, out, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="report/figures")
    ap.add_argument("--main-run", default="main")
    ap.add_argument("--experiment-run", default=None)
    a = ap.parse_args()
    plot_overfit(a.runs_dir, a.out_dir)
    plot_resume(a.runs_dir, a.out_dir)
    plot_pilots(a.runs_dir, a.out_dir)
    plot_main(a.runs_dir, a.out_dir, a.main_run)
    if a.experiment_run:
        plot_main(a.runs_dir, a.out_dir, a.main_run, a.experiment_run, "fig_experiment.png")


if __name__ == "__main__":
    main()
