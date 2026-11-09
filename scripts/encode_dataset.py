"""Encode a TinyStories text file into a uint16 .npy of token ids (Part 2).

    python scripts/encode_dataset.py --input data/TinyStoriesV2-GPT4-valid.txt --out data/valid.npy \
        --tokenizer-dir artifacts/tokenizer --num-processes 8
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minilm.tokenizer import Tokenizer
from minilm.tokenizer.bpe import find_chunk_boundaries

EOT = "<|endoftext|>"
_TOK = None


def _init(tok_dir):
    global _TOK
    _TOK = Tokenizer.from_files(os.path.join(tok_dir, "vocab.json"), os.path.join(tok_dir, "merges.txt"), [EOT])


def _encode_chunk(args):
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    ids = _TOK.encode(data.decode("utf-8", errors="ignore"))
    return np.asarray(ids, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--num-processes", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--chunks-per-process", type=int, default=8)
    args = ap.parse_args()

    size = os.path.getsize(args.input)
    bounds = find_chunk_boundaries(args.input, args.num_processes * args.chunks_per_process, EOT.encode())
    jobs = [(args.input, s, e) for s, e in zip(bounds[:-1], bounds[1:])]
    t0 = time.time()
    with Pool(args.num_processes, initializer=_init, initargs=(args.tokenizer_dir,)) as pool:
        parts = list(pool.imap(_encode_chunk, jobs))
    arr = np.concatenate(parts)
    elapsed = time.time() - t0
    assert arr.dtype == np.uint16
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, arr)
    stats = {"input": args.input, "bytes": size, "tokens": int(arr.shape[0]), "seconds": round(elapsed, 1),
             "bytes_per_second": round(size / elapsed), "num_processes": args.num_processes,
             "bytes_per_token": round(size / arr.shape[0], 4), "max_id": int(arr.max()), "dtype": str(arr.dtype)}
    print(json.dumps(stats, indent=2))
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
