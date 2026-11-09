"""Tests for BPE training (Part 1)."""
import random

import pytest

from minilm.tokenizer import PAT, train_bpe
from minilm.tokenizer.bpe import count_pretokens

EOT = "<|endoftext|>"


def _write(tmp_path, text, name="corpus.txt"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_hand_worked_example(tmp_path):
    """low x5, lower x2, widest x3, newest x6 (split on whitespace only): the worked BPE example.

    Pair counts at the start: (s,t)=9 and (e,s)=9 tie -> the larger tuple (s,t) wins. Then e+st,
    o+w (tie with l+o at 7, o>l), l+ow, w+est (tie at 6 with e+w, n+e: w is largest) and n+e.
    """
    text = "low " * 5 + "lower " * 2 + "widest " * 3 + "newest " * 6
    path = _write(tmp_path, text)
    vocab, merges = train_bpe(path, vocab_size=256 + 1 + 6, special_tokens=[EOT], pat=r"\S+")
    assert merges == [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")]
    assert vocab[257] == b"st" and vocab[262] == b"ne"
    assert vocab[256] == EOT.encode()


def test_tie_break_is_larger_tuple(tmp_path):
    # (A,B) and (B,A) tie at 1 occurrence each -> max picks (b"B", b"A")
    path = _write(tmp_path, "AB BA")
    _, merges = train_bpe(path, vocab_size=256 + 1 + 1, special_tokens=[EOT], pat=r"\S+")
    assert merges == [(b"B", b"A")]


def test_no_merge_crosses_story_boundary(tmp_path):
    story_a = "xyz xyz xyz xyz zyx " * 20
    story_b = "uvw uvw uvw uvw wvu " * 20
    path = _write(tmp_path, story_a + EOT + story_b)
    for pat in (PAT, r"\S+"):
        vocab, _ = train_bpe(path, vocab_size=256 + 1 + 40, special_tokens=[EOT], pat=pat)
        a_bytes, b_bytes = set(b"xyz "), set(b"uvw ")
        for i, tok in vocab.items():
            if i < 257:
                continue
            assert not (set(tok) & set(b"xyz") and set(tok) & set(b"uvw")), tok
            assert b"<|" not in tok and b"|>" not in tok, tok  # special token never leaks into merges
        _ = (a_bytes, b_bytes)


def test_special_token_not_counted_as_pair(tmp_path):
    # If specials were not removed, "<|" would be the most common pair here.
    path = _write(tmp_path, (EOT + "a b ") * 50)
    _, merges = train_bpe(path, vocab_size=256 + 1 + 5, special_tokens=[EOT])
    assert all(b"<" not in m[0] + m[1] for m in merges)


def test_vocab_size_exact_and_ids_unique(tmp_path):
    rng = random.Random(0)
    words = ["".join(rng.choice("abcdefgh") for _ in range(rng.randint(2, 7))) for _ in range(300)]
    text = EOT.join(" ".join(rng.choice(words) for _ in range(40)) for _ in range(30))
    path = _write(tmp_path, text)
    vocab, merges = train_bpe(path, vocab_size=400, special_tokens=[EOT])
    assert len(vocab) == 400
    assert len(merges) == 400 - 257
    assert len(set(vocab.keys())) == 400
    assert len(set(vocab.values())) == 400, "two ids share the same token"
    assert sorted(vocab) == list(range(400))
    # every merge result is the concatenation of its parts and appears in the vocab
    values = set(vocab.values())
    for a, b in merges:
        assert a in values and b in values and a + b in values


def test_merges_replay_in_order(tmp_path):
    """The i-th merge must only reference tokens that already existed (bytes or earlier merges)."""
    text = ("the cat sat on the mat. " * 30 + EOT) * 3
    path = _write(tmp_path, text)
    vocab, merges = train_bpe(path, vocab_size=300, special_tokens=[EOT])
    known = {bytes([i]) for i in range(256)}
    for a, b in merges:
        assert a in known and b in known
        known.add(a + b)


def test_vocab_too_small_raises(tmp_path):
    path = _write(tmp_path, "hello")
    with pytest.raises(ValueError):
        train_bpe(path, vocab_size=100, special_tokens=[EOT])


def test_multiprocess_counts_match_single_process(tmp_path):
    rng = random.Random(1)
    stories = [" ".join(rng.choice(["the", "cat", "dog", "ran", "é", "你好", "12", "!"]) for _ in range(30))
               for _ in range(200)]
    path = _write(tmp_path, EOT.join(stories))
    single = count_pretokens(path, [EOT], PAT, num_processes=1)
    multi = count_pretokens(path, [EOT], PAT, num_processes=2, chunks_per_process=3)
    assert single == multi
    assert sum(single.values()) > 0
