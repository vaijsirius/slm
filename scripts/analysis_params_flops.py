"""Questions 5-8: parameter counts, matmul FLOPs, optimizer memory.

    python scripts/analysis_params_flops.py > report/analysis/params_flops.txt
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minilm.model import TransformerLM
from minilm.train import build_optimizer, cross_entropy_loss


def params(V, d, L, d_ff, mats=3):
    parts = {
        "token embedding (V x d)": V * d,
        "attention q,k,v,o projections (4 d^2) x L": 4 * d * d * L,
        f"FFN ({mats} matrices of d x d_ff) x L": mats * d * d_ff * L,
        "RMSNorm weights (2 per layer + final)": (2 * L + 1) * d,
        "output projection (d x V)": V * d,
    }
    return parts


def matmul_flops(V, d, L, d_ff, heads, T, mats=3):
    dh = d // heads
    per_layer = {
        "q,k,v,o projections (4 x 2*T*d*d)": 4 * 2 * T * d * d,
        "scores Q K^T (heads x 2*T*dh*T)": heads * 2 * T * dh * T,
        "weights x V (heads x 2*T*T*dh)": heads * 2 * T * T * dh,
        "FFN up-projections (w1[,w3]) (2*T*d*d_ff each)": (mats - 1) * 2 * T * d * d_ff,
        "FFN down-projection w2 (2*T*d_ff*d)": 2 * T * d_ff * d,
    }
    out = {k: v * L for k, v in per_layer.items()}
    out["output projection (2*T*d*V)"] = 2 * T * d * V
    return per_layer, out


def report(name, V, d, L, d_ff, heads, T, mats=3):
    print(f"\n######## {name}: V={V}, d_model={d}, layers={L}, heads={heads}, d_ff={d_ff}, context={T}")
    pc = params(V, d, L, d_ff, mats)
    total = sum(pc.values())
    print("-- parameters")
    for k, v in pc.items():
        print(f"  {k:48s} {v:>14,d}  ({100 * v / total:5.1f}%)")
    print(f"  {'TOTAL':48s} {total:>14,d}")
    emb_out = 2 * V * d
    print(f"  embedding + output layers: {emb_out:,} = {100 * emb_out / total:.1f}% of the total "
          f"(transformer layers incl. norms: {total - emb_out:,})")
    per_layer, tot = matmul_flops(V, d, L, d_ff, heads, T, mats)
    ft = sum(tot.values())
    print("-- matmul FLOPs, one forward pass over T tokens (2mnp per m x n @ n x p), per layer then whole model")
    for k in per_layer:
        print(f"  {k:48s} per layer {per_layer[k]:>16,d}   x{L}: {tot[k]:>18,d}  ({100 * tot[k] / ft:5.1f}%)")
    k = "output projection (2*T*d*V)"
    print(f"  {k:48s} {'':>22} {tot[k]:>18,d}  ({100 * tot[k] / ft:5.1f}%)")
    print(f"  {'TOTAL forward':48s} {ft:>16,d} = {ft / 1e9:.2f} GFLOPs;  forward+backward ~ 3x = {3 * ft / 1e9:.1f} GFLOPs")
    groups = {
        "attention projections": tot["q,k,v,o projections (4 x 2*T*d*d)"],
        "attention scores+values": tot["scores Q K^T (heads x 2*T*dh*T)"] + tot["weights x V (heads x 2*T*T*dh)"],
        "feed-forward": tot["FFN up-projections (w1[,w3]) (2*T*d*d_ff each)"] + tot["FFN down-projection w2 (2*T*d_ff*d)"],
        "output projection": tot[k],
    }
    print("  share:", ", ".join(f"{g} {100 * v / ft:.1f}%" for g, v in groups.items()))
    return total, ft


# ---- Q5 / Q6: our model
V, d, L, dff, H, T = 10_000, 512, 4, 1344, 16, 256
total, ft = report("OUR MODEL", V, d, L, dff, H, T)
model = TransformerLM(V, T, d, L, H, dff, 10_000.0)
code_total = sum(p.numel() for p in model.parameters())
print(f"\ncode check: sum(p.numel()) = {code_total:,}  matches by-hand total: {code_total == total}")
for n, mod in [("embedding", model.token_embeddings), ("layers", model.layers), ("final norm", model.ln_final),
               ("lm_head", model.lm_head)]:
    print(f"  {n:12s} {sum(p.numel() for p in mod.parameters()):>12,d}")
print(f"  one block: {sum(p.numel() for p in model.layers[0].parameters()):,}")

try:  # independent check with PyTorch's FLOP counter (counts matmuls / sdpa)
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        model(torch.randint(0, V, (1, T)))
    print(f"torch FlopCounterMode forward total: {fc.get_total_flops():,}  (by hand: {ft:,})")
except Exception as e:  # pragma: no cover
    print("FlopCounterMode unavailable:", e)

# ---- Q7: GPT-2 small sized
for name, ff, mats in (("GPT-2-small-sized, SwiGLU d_ff=2048 (8/3 d)", 2048, 3),
                       ("GPT-2 small as published (GELU MLP, 2 matrices, d_ff=3072=4d; untied head)", 3072, 2)):
    report(name, 50_257, 768, 12, ff, 12, 1024, mats)

# ---- Q8: memory of params / grads / AdamW state (fp32)
print("\n######## Q8: fp32 memory for our model")
n = code_total
MB = 1e6
print(f"parameters : {n:,} x 4 B = {4 * n / MB:.1f} MB")
print(f"gradients  : {n:,} x 4 B = {4 * n / MB:.1f} MB")
print(f"AdamW m    : {4 * n / MB:.1f} MB   AdamW v: {4 * n / MB:.1f} MB   -> state = {8 * n / MB:.1f} MB")
print(f"total      : {16 * n / MB:.1f} MB ({16 * n / 2 ** 20:.1f} MiB) = 16 bytes per parameter")
opt = build_optimizer(model, 1e-3)
x = torch.randint(0, V, (2, 32))
cross_entropy_loss(model(x), x).backward()
opt.step()
state_elems = sum(t.numel() for s in opt.state.values() for t in s.values() if t.ndim > 0)
grad_elems = sum(p.grad.numel() for p in model.parameters())
print(f"measured: optimizer state tensors hold {state_elems:,} elements = {state_elems / n:.2f} x #params "
      f"({4 * state_elems / MB:.1f} MB); gradients {grad_elems:,}")
