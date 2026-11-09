"""Tokenizer class (Part 2) and vocab/merges (de)serialisation.

Files use the GPT-2 convention so they are human-readable: every byte is mapped to a printable
unicode character (``bytes_to_unicode``), ``vocab.json`` is ``{token_string: id}`` and
``merges.txt`` has one ``"left right"`` pair per line, in merge order.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import regex

from .bpe import PAT, special_split_pattern


# --------------------------------------------------------------------------------------
# bytes <-> printable unicode (GPT-2 trick) for readable files
# --------------------------------------------------------------------------------------
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_B2U = bytes_to_unicode()
_U2B = {u: b for b, u in _B2U.items()}


def _to_str(b: bytes) -> str:
    return "".join(_B2U[x] for x in b)


def _to_bytes(s: str) -> bytes:
    return bytes(_U2B[c] for c in s)


def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                          vocab_path: str, merges_path: str) -> None:
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({_to_str(b): i for i, b in sorted(vocab.items())}, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_to_str(a)} {_to_str(b)}\n")


def load_vocab_and_merges(vocab_path: str, merges_path: str):
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {i: _to_bytes(s) for s, i in raw.items()}
    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            left, right = line.split(" ")
            merges.append((_to_bytes(left), _to_bytes(right)))
    return vocab, merges


# --------------------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self.byte_to_id = {b: i for i, b in self.vocab.items()}
        for tok in self.special_tokens:  # a special token missing from the vocab goes at the end
            tb = tok.encode("utf-8")
            if tb not in self.byte_to_id:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = tb
                self.byte_to_id[tb] = new_id
        self.special_ids = {t: self.byte_to_id[t.encode("utf-8")] for t in self.special_tokens}
        # (left_id, right_id) -> (rank, merged_id); lower rank = learned earlier = applied first
        self._ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (a, b) in enumerate(self.merges):
            self._ranks[(self.byte_to_id[a], self.byte_to_id[b])] = (rank, self.byte_to_id[a + b])
        self._pat = regex.compile(PAT)
        sp = special_split_pattern(self.special_tokens)  # longest first => longest match wins
        self._special_re = regex.compile(f"({sp})") if sp else None
        self._max_special_len = max((len(t) for t in self.special_tokens), default=0)
        self._cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        vocab, merges = load_vocab_and_merges(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ encode
    def _encode_pretoken(self, pre: bytes) -> list[int]:
        cached = self._cache.get(pre)
        if cached is not None:
            return cached
        ids = [self.byte_to_id[bytes([x])] for x in pre]
        ranks = self._ranks
        while len(ids) > 1:
            best_rank, best_pair, best_new = None, None, None
            for pair in zip(ids, ids[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r[0] < best_rank):
                    best_rank, best_pair, best_new = r[0], pair, r[1]
            if best_pair is None:
                break
            a, b = best_pair
            out, i, n = [], 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) < 500_000:
            self._cache[pre] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        for m in self._pat.finditer(text):
            out.extend(self._encode_pretoken(m.group().encode("utf-8")))
        return out

    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            return self._encode_plain(text)
        out: list[int] = []
        # re.split with a capture group returns [text, special, text, special, ..., text]
        for i, part in enumerate(self._special_re.split(text)):
            if i % 2 == 1:
                out.append(self.special_ids[part])
            elif part:
                out.extend(self._encode_plain(part))
        return out

    def encode_iterable(self, iterable: Iterable[str], flush_chars: int = 1 << 16) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. an open file); memory stays bounded.

        The result is identical to ``encode("".join(iterable))``: we only encode up to the start of
        the last pre-token of the buffer (whose extent could still change when more text arrives) and
        never cut through something that could be the start of a special token.
        """
        buf = ""
        for piece in iterable:
            buf += piece
            if len(buf) < flush_chars:
                continue
            cut = self._safe_cut(buf)
            if cut > 0:
                yield from self.encode(buf[:cut])
                buf = buf[cut:]
        yield from self.encode(buf)

    def _safe_cut(self, buf: str) -> int:
        limit = len(buf)
        # hold back a trailing proper prefix of a special token ("<|endof")
        for tok in self.special_tokens:
            for k in range(min(len(tok) - 1, limit), 0, -1):
                if buf.endswith(tok[:k]):
                    limit = min(limit, len(buf) - k)
                    break
        head = buf[:limit]
        # start of the text after the last complete special token
        seg_start = 0
        if self._special_re is not None:
            for m in self._special_re.finditer(head):
                seg_start = m.end()
        last_start = None
        for m in self._pat.finditer(head, seg_start):
            last_start = m.start()
        return limit if last_start is None else last_start

    # ------------------------------------------------------------------ decode
    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
