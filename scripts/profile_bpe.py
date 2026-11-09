"""Question 1: profile BPE training with cProfile (single process, so the profiler sees everything).

    python scripts/profile_bpe.py --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000
"""
import argparse
import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minilm.tokenizer import train_bpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    pr = cProfile.Profile()
    pr.enable()
    train_bpe(a.input, a.vocab_size, ["<|endoftext|>"], num_processes=1)
    pr.disable()
    st = pstats.Stats(pr)
    cum = {}
    for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
        cum[name] = max(cum.get(name, 0), ct)
    total = cum.get("train_bpe", 0)
    print(f"train_bpe total: {total:.1f}s")
    for name in ("count_pretokens", "count_pretokens_in_text", "merge_loop", "_merge_word"):
        if name in cum:
            print(f"  {name:26s} {cum[name]:7.2f}s  ({100 * cum[name] / total:5.1f}% of train_bpe)")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(a.top)
    print(s.getvalue())
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime").print_stats(8)
    print(s.getvalue())


if __name__ == "__main__":
    main()
