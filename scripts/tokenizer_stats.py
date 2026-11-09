"""Compression ratio (bytes/token) on 10 random stories and single-process encoding speed (Part 2).

    python scripts/tokenizer_stats.py --text data/TinyStoriesV2-GPT4-valid.txt --tokenizer-dir artifacts/tokenizer
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minilm.tokenizer import Tokenizer

EOT = "<|endoftext|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = Tokenizer.from_files(os.path.join(a.tokenizer_dir, "vocab.json"), os.path.join(a.tokenizer_dir, "merges.txt"), [EOT])
    with open(a.text, encoding="utf-8") as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(EOT) if s.strip()]
    rng = random.Random(a.seed)
    sample = rng.sample(stories, 10)
    rows, tb, tt = [], 0, 0
    for s in sample:
        n_bytes, n_tok = len(s.encode("utf-8")), len(tok.encode(s))
        rows.append({"bytes": n_bytes, "tokens": n_tok, "bytes_per_token": round(n_bytes / n_tok, 3)})
        tb += n_bytes
        tt += n_tok
        assert tok.decode(tok.encode(s)) == s
    res = {"stories_sampled": 10, "per_story": rows, "mean_of_ratios": round(sum(r["bytes_per_token"] for r in rows) / 10, 3),
           "pooled_bytes_per_token": round(tb / tt, 3)}

    # single-process speed on the full file (fresh tokenizer => cold pre-token cache, as in a first run)
    tok2 = Tokenizer.from_files(os.path.join(a.tokenizer_dir, "vocab.json"), os.path.join(a.tokenizer_dir, "merges.txt"), [EOT])
    n_bytes = len(raw.encode("utf-8"))
    t0 = time.perf_counter()
    ids = tok2.encode(raw)
    dt = time.perf_counter() - t0
    res.update(file_bytes=n_bytes, file_tokens=len(ids), encode_seconds=round(dt, 2),
               bytes_per_second=round(n_bytes / dt), tokens_per_second=round(len(ids) / dt),
               whole_file_bytes_per_token=round(n_bytes / len(ids), 3))
    # streaming API speed (reads the file piece by piece, memory-bounded)
    tok3 = Tokenizer.from_files(os.path.join(a.tokenizer_dir, "vocab.json"), os.path.join(a.tokenizer_dir, "merges.txt"), [EOT])
    t0 = time.perf_counter()
    with open(a.text, encoding="utf-8") as f:
        cnt = sum(1 for _ in tok3.encode_iterable(f))
    dt2 = time.perf_counter() - t0
    res.update(iterable_seconds=round(dt2, 2), iterable_matches_encode=(cnt == len(ids)))
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
