"""Warm-up questions (Parts 1 and 5) and Questions 3 and 4 - prints the numbers used in the writeup.

    python scripts/analysis_warmups.py > report/analysis/warmups.txt
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minilm.model import RoPE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def h(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ---------------------------------------------------------------- Part 1 warm-ups
h("Part 1, warm-up 1: chr(0), repr vs print")
print("chr(0) ->", repr(chr(0)), "| ord:", ord(chr(0)), "| len:", len(chr(0)))
print("repr(chr(0)) =", repr(chr(0)), "(an escaped, visible description)")
print("print(chr(0)) writes the raw NUL byte:", end=" ")
sys.stdout.write("[" + chr(0) + "]\n")
s = "this is a test" + chr(0) + "string"
print("repr :", repr(s))
print("print:", s.replace(chr(0), "<NUL written raw, shows as nothing/blank>"))

h("Part 1, warm-up 2: UTF-8 vs UTF-16 vs UTF-32")
for text in ["hello", "hello world, once upon a time", "नमस्ते", "café", "你好", "😀"]:
    u8, u16, u32 = (text.encode(e) for e in ("utf-8", "utf-16", "utf-32"))
    print(f"{text!r:36} chars={len(text):2d} utf-8={len(u8):3d}B utf-16={len(u16):3d}B utf-32={len(u32):3d}B")
print("utf-16 'hello' bytes:", list("hello".encode("utf-16")), "(BOM + a zero byte after every ASCII letter)")
print("utf-32 'hello' bytes:", list("hello".encode("utf-32")))
print("utf-8  'hello' bytes:", list("hello".encode("utf-8")))

h("Part 1, warm-up 3: decoding one byte at a time")


def decode_bytewise(bs):
    return "".join(bytes([b]).decode("utf-8") for b in bs)


print("ASCII works:", decode_bytewise("hello".encode()))
try:
    decode_bytewise("café".encode())
except UnicodeDecodeError as e:
    print("'café' ->", list("café".encode()), "-> fails:", e)
print("correct (whole sequence):", "café".encode().decode("utf-8"))

h("Part 1, warm-up 4: an invalid two-byte sequence")
for bs in [bytes([0xC3, 0x28]), bytes([0xFF, 0xFF]), bytes([0x80, 0x80])]:
    try:
        bs.decode("utf-8")
    except UnicodeDecodeError as e:
        print(list(bs), [f"{b:08b}" for b in bs], "->", e.reason)

# ---------------------------------------------------------------- Part 5 warm-up (SGD lr)
h("Part 5 warm-up: SGD on mean(w^2), 10x10 parameter, 10 steps")
for lr in (1, 10, 100, 1000):
    torch.manual_seed(0)
    w = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = torch.optim.SGD([w], lr=lr)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = (w ** 2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    print(f"lr={lr:5d}: " + "  ".join(f"{l:.3g}" for l in losses))
print("gradient of mean(w^2) over 100 entries = 2w/100 = 0.02 w, so an SGD step maps w -> (1 - 0.02*lr) w:")
for lr in (1, 10, 100, 1000):
    f = 1 - 0.02 * lr
    print(f"  lr={lr:5d}: w factor {f:7.2f}, loss factor per step {f * f:10.4g}")

# ---------------------------------------------------------------- Question 3
h("Question 3: why divide by sqrt(d_k)?  (d_k = 32)")
torch.manual_seed(0)
d_k, n_keys, trials = 32, 16, 20_000
q = torch.randn(trials, 1, d_k)
k = torch.randn(trials, n_keys, d_k)
raw = (q @ k.transpose(-1, -2)).squeeze(1)  # (trials, n_keys), each score = sum of 32 products
scaled = raw / math.sqrt(d_k)
print(f"std of raw scores q.k    : {raw.std():.3f} (theory sqrt(d_k) = {math.sqrt(d_k):.3f})")
print(f"std of scaled scores     : {scaled.std():.3f} (theory 1)")


def stats(scores):
    p = torch.softmax(scores, -1)
    ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
    jac = (p * (1 - p)).sum(-1)  # trace of the softmax Jacobian: ~0 when saturated (one-hot)
    return p.max(-1).values.mean().item(), ent.mean().item(), jac.mean().item()


for name, sc in (("without scaling", raw), ("with 1/sqrt(d_k)", scaled)):
    mx, ent, jac = stats(sc)
    print(f"{name:17}: mean max-prob {mx:.3f} | mean entropy {ent:.3f} nats (uniform = {math.log(n_keys):.3f}) "
          f"| softmax-gradient mass sum p(1-p) {jac:.3f}")
one = torch.randn(1, d_k), torch.randn(n_keys, d_k)
s0 = (one[0] @ one[1].T)[0]
print("\none example of 6 of the 16 scores and their softmax:")
print("  raw scores       :", [round(x, 2) for x in s0[:6].tolist()])
print("  softmax(raw)     :", [round(x, 3) for x in torch.softmax(s0, -1)[:6].tolist()])
print("  scaled scores    :", [round(x, 2) for x in (s0 / math.sqrt(d_k))[:6].tolist()])
print("  softmax(scaled)  :", [round(x, 3) for x in torch.softmax(s0 / math.sqrt(d_k), -1)[:6].tolist()])

# ---------------------------------------------------------------- Question 4
h("Question 4: after RoPE, <q_i, k_j> depends only on i - j")
torch.manual_seed(0)
d_head = 32
rope = RoPE(10_000.0, d_head, 512)
qv, kv = torch.randn(d_head), torch.randn(d_head)


def rot_dot(i, j):
    qi = rope(qv.expand(1, i + 1, d_head).contiguous())[0, i]
    kj = rope(kv.expand(1, j + 1, d_head).contiguous())[0, j]
    return (qi @ kj).item()


print(f"un-rotated q.k = {(qv @ kv).item():.6f}")
for dist in (0, 1, 3, 10):
    vals = [rot_dot(j + dist, j) for j in (0, 7, 50, 200, 400)]
    print(f"i-j = {dist:2d}: " + "  ".join(f"({j + dist:3d},{j:3d}) -> {v:+.6f}" for j, v in zip((0, 7, 50, 200, 400), vals))
          + f"   spread = {max(vals) - min(vals):.2e}")
print("Why: R_i is block-diagonal with 2x2 rotations of angle i*w_k; R_i^T R_j = R_(j-i) because rotation angles add,")
print("so (R_i q).(R_j k) = q^T R_i^T R_j k = q^T R_(j-i) k  -> only the offset j - i enters.")
