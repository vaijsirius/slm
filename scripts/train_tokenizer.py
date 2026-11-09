"""Train the 10k BPE tokenizer on TinyStories and save vocab.json / merges.txt (Part 1).

    python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt --out-dir artifacts/tokenizer \
        --vocab-size 10000 --num-processes 8
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minilm.tokenizer import save_vocab_and_merges
from minilm.tokenizer import bpe


def total_rss_mb():
    try:
        import psutil
    except ImportError:
        return float("nan")
    me = psutil.Process()
    procs = [me] + me.children(recursive=True)
    tot = 0
    for p in procs:
        try:
            tot += p.memory_info().rss
        except Exception:
            pass
    return tot / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--num-processes", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--max-bytes", type=int, default=None, help="only use roughly the first N bytes")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    special = ["<|endoftext|>"]

    peak = [0.0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], total_rss_mb())
            time.sleep(0.5)

    threading.Thread(target=sampler, daemon=True).start()

    t0 = time.time()
    counts = bpe.count_pretokens(args.input, special, bpe.PAT, args.num_processes, args.max_bytes)
    t1 = time.time()
    print(f"pre-tokenisation: {t1 - t0:.1f}s, {len(counts):,} distinct pre-tokens, "
          f"{sum(counts.values()):,} total", flush=True)
    vocab = {i: bytes([i]) for i in range(256)}
    for t in special:
        vocab[len(vocab)] = t.encode("utf-8")
    merges = bpe.merge_loop(counts, vocab, args.vocab_size - len(vocab))
    t2 = time.time()
    stop.set()
    print(f"merging: {t2 - t1:.1f}s for {len(merges)} merges; total {t2 - t0:.1f}s", flush=True)

    save_vocab_and_merges(vocab, merges, os.path.join(args.out_dir, "vocab.json"),
                          os.path.join(args.out_dir, "merges.txt"))
    longest_id, longest = max(((i, b) for i, b in vocab.items() if i != 256), key=lambda kv: len(kv[1]))
    stats = {
        "input": args.input, "max_bytes": args.max_bytes, "vocab_size": len(vocab), "num_merges": len(merges),
        "pretokenize_seconds": round(t1 - t0, 1), "merge_seconds": round(t2 - t1, 1),
        "total_seconds": round(t2 - t0, 1), "peak_total_rss_mb": round(peak[0], 0),
        "num_processes": args.num_processes, "distinct_pretokens": len(counts),
        "longest_token_id": longest_id, "longest_token_bytes": len(longest),
        "longest_token": longest.decode("utf-8", errors="replace"),
    }
    with open(os.path.join(args.out_dir, "train_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
