"""Byte-level BPE training (Part 1).

``train_bpe(input_path, vocab_size, special_tokens)`` returns ``(vocab, merges)``:

* ``vocab``  : ``dict[int, bytes]``  - ids 0..255 are the raw bytes, then the special tokens,
               then one entry per merge, in the order the merges were learned.
* ``merges`` : ``list[tuple[bytes, bytes]]`` - the pairs merged, in order.

Algorithm (see the assignment):
 1. split the corpus on the special tokens so a merge can never straddle two stories;
 2. split every story into GPT-2 style pre-tokens and count them;
 3. repeatedly merge the most frequent adjacent pair (ties -> larger pair as a tuple of bytes);
 4. after a merge only the pre-tokens that contained the pair are touched, and the best pair is
    found with a lazily-updated max-heap, so 10k merges stay fast.
"""
from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex

# The GPT-2 pre-tokenisation pattern (given in the assignment).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def special_split_pattern(special_tokens: list[str]) -> str | None:
    """Regex alternation of the (escaped) special tokens, longest first. None if there are none."""
    if not special_tokens:
        return None
    return "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True))


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split ``text`` on the special tokens, *dropping* them (used when counting pairs)."""
    pat = special_split_pattern(special_tokens)
    return [text] if pat is None else regex.split(pat, text)


# --------------------------------------------------------------------------------------
# Step 1+2: pre-token counting (optionally in parallel)
# --------------------------------------------------------------------------------------
def count_pretokens_in_text(text: str, special_tokens: list[str], pat: str = PAT) -> Counter:
    """Count pre-tokens (as UTF-8 bytes) in ``text`` after splitting on the special tokens."""
    compiled = regex.compile(pat)
    counts: Counter = Counter()
    for story in split_on_special_tokens(text, special_tokens):
        counts.update(m.group() for m in compiled.finditer(story))
    return Counter({k.encode("utf-8"): v for k, v in counts.items()})


def find_chunk_boundaries(path: str, num_chunks: int, split_token: bytes) -> list[int]:
    """Byte offsets that cut the file into ~``num_chunks`` pieces, each starting at a special token.

    Cutting only at special tokens guarantees no story (and no UTF-8 character) is split.
    """
    size = os.path.getsize(path)
    if num_chunks <= 1 or size == 0:
        return [0, size]
    guesses = [i * size // num_chunks for i in range(num_chunks + 1)]
    guesses[-1] = size
    mini = 1 << 16
    with open(path, "rb") as f:
        for i in range(1, len(guesses) - 1):
            pos = guesses[i]
            f.seek(pos)
            while True:
                buf = f.read(mini)
                if not buf:  # EOF
                    guesses[i] = size
                    break
                j = buf.find(split_token)
                if j != -1:
                    guesses[i] = pos + j
                    break
                pos += max(1, len(buf) - len(split_token) + 1)
                f.seek(pos)
    return sorted(set(guesses))


def _count_chunk(args) -> Counter:
    path, start, end, special_tokens, pat = args
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    return count_pretokens_in_text(data.decode("utf-8", errors="ignore"), special_tokens, pat)


def count_pretokens(path: str, special_tokens: list[str], pat: str = PAT, num_processes: int = 1,
                    max_bytes: int | None = None, chunks_per_process: int = 8) -> Counter:
    """Pre-token counts for a whole file; ``max_bytes`` restricts to roughly the first N bytes."""
    size = os.path.getsize(path)
    limit = size if max_bytes is None else min(size, max_bytes)
    split_token = special_tokens[0].encode("utf-8") if special_tokens else None
    if split_token is None or (num_processes <= 1 and limit == size and size < (256 << 20)):
        # Small file (or nothing to align on): one chunk, no processes.
        with open(path, "rb") as f:
            data = f.read(limit)
        return count_pretokens_in_text(data.decode("utf-8", errors="ignore"), special_tokens, pat)
    n_chunks = max(1, num_processes * chunks_per_process)
    bounds = find_chunk_boundaries(path, n_chunks, split_token)
    if limit < size:  # keep only chunks that start before the limit
        kept = [b for b in bounds if b < limit]
        bounds = kept + [next((b for b in bounds if b >= limit), size)]
    jobs = [(path, s, e, special_tokens, pat) for s, e in zip(bounds[:-1], bounds[1:])]
    total: Counter = Counter()
    if num_processes <= 1:
        for j in jobs:
            total.update(_count_chunk(j))
    else:
        with Pool(num_processes) as pool:
            for c in pool.imap_unordered(_count_chunk, jobs):
                total.update(c)
    return total


# --------------------------------------------------------------------------------------
# Step 3+4: the merge loop
# --------------------------------------------------------------------------------------
class _HeapEntry:
    """Max-heap entry ordered by (count, pair-as-bytes); heapq is a min-heap so __lt__ is flipped."""

    __slots__ = ("count", "key", "pair")

    def __init__(self, count: int, key: tuple[bytes, bytes], pair: tuple[int, int]):
        self.count, self.key, self.pair = count, key, pair

    def __lt__(self, other: "_HeapEntry") -> bool:
        return (self.count, self.key) > (other.count, other.key)


def _merge_word(word: list[int], a: int, b: int, new_id: int) -> list[int]:
    out, i, n = [], 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def merge_loop(word_counts: dict[bytes, int], vocab: dict[int, bytes], num_merges: int):
    """Run ``num_merges`` BPE merges over ``word_counts``; mutates ``vocab``; returns the merges."""
    words = [list(w) for w in word_counts]  # token ids == byte values at the start
    freqs = list(word_counts.values())

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_counts[p] += freqs[wi]
            pair_words[p].add(wi)

    heap = [_HeapEntry(c, (vocab[p[0]], vocab[p[1]]), p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    next_id = max(vocab) + 1
    for _ in range(num_merges):
        best = None
        while heap:
            e = heapq.heappop(heap)
            if pair_counts.get(e.pair, 0) == e.count:  # otherwise stale -> skip
                best = e.pair
                break
        if best is None:  # nothing left to merge
            break
        a, b = best
        new_id = next_id
        next_id += 1
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        delta: dict[tuple[int, int], int] = defaultdict(int)
        for wi in list(pair_words.pop(best, ())):
            w, f = words[wi], freqs[wi]
            new_w = _merge_word(w, a, b, new_id)
            if len(new_w) == len(w):  # stale membership: pair no longer in this word
                continue
            for p in zip(w, w[1:]):
                delta[p] -= f
            for p in zip(new_w, new_w[1:]):
                delta[p] += f
                pair_words[p].add(wi)
            words[wi] = new_w
        for p, d in delta.items():
            if d == 0:
                continue
            c = pair_counts.get(p, 0) + d
            if c <= 0:
                pair_counts.pop(p, None)
            else:
                pair_counts[p] = c
                heapq.heappush(heap, _HeapEntry(c, (vocab[p[0]], vocab[p[1]]), p))
        pair_counts.pop(best, None)
    return merges


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str] | None = None,
              pat: str = PAT, num_processes: int = 1, max_bytes: int | None = None):
    """Train a byte-level BPE tokenizer. See module docstring.

    ``pat`` and ``max_bytes`` / ``num_processes`` are extras (toy examples use another pattern; the
    big run can use the first part of the file and several processes).
    """
    special_tokens = list(DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")
    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} is smaller than the {len(vocab)} initial tokens")
    word_counts = count_pretokens(str(input_path), special_tokens, pat, num_processes, max_bytes)
    merges = merge_loop(word_counts, vocab, num_merges)
    return vocab, merges
