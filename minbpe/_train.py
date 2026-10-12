"""BasicTokenizer training, loaded on first use (off the encoding import path)."""
from heapq import heapify, heappop, heappush
from operator import itemgetter

from .base import get_stats, pop_top_pairs, _count_elements

_last = itemgetter(-1)
_first = itemgetter(0)


def basic_train(self, text, vocab_size, verbose):
    # body of BasicTokenizer.train
    assert vocab_size >= 256
    num_merges = vocab_size - 256
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255
    merges = {} # (int, int) -> int
    vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    # pair counts once, then updated locally around each merged occurrence
    stats = get_stats(ids)
    heap = [(-c, p) for p, c in stats.items()] # lazy max-heap of counts
    heapify(heap)
    if vocab_size <= 0x110000:
        # every id fits in one code point: ids as a str, merging done in C
        train_str(text_bytes.decode("latin-1"), num_merges, stats, heap,
                  merges, vocab, verbose)
    else:
        from ._slow import train_ints # ids beyond one code point
        train_ints(ids, num_merges, stats, heap, merges, vocab, verbose)
    self.merges = merges # used in encode()
    self.vocab = vocab   # used in decode()


def train_str(s, num_merges, stats, heap, merges, vocab, verbose):
    # s: the current ids as a str of code points chr(id)
    stats_get = stats.get
    for i in range(num_merges):
        top, cands = pop_top_pairs(heap, stats_get, heappush, heappop)
        if len(cands) == 1:
            pair = cands[0]
        else:
            # tie: the pair whose first occurrence is earliest
            best = -1
            for p in cands:
                j = s.find(chr(p[0]) + chr(p[1]))
                if best < 0 or j < best:
                    best = j
                    pair = p
        idx = 256 + i
        p0, p1 = pair
        # merge() semantics (left to right, non-overlapping) == str.replace;
        # chr(idx) is new, so the pieces between merged occurrences give the
        # neighbours (an empty inner piece: two adjacent occurrences)
        x = chr(idx)
        s = s.replace(chr(p0) + chr(p1), x)
        parts = s.split(x)
        left = {}
        _count_elements(left, map(_last, filter(None, parts[:-1])))
        right = {}
        _count_elements(right, map(_first, filter(None, parts[1:])))
        stats[pair] -= len(parts) - 1
        removed = [pair]
        added = {} # new pairs contain idx: none is in stats yet
        for a, c in left.items():
            a = ord(a)
            p = (a, p0)
            stats[p] -= c
            removed.append(p)
            added[(a, idx)] = c
        for b, c in right.items():
            b = ord(b)
            p = (p1, b)
            stats[p] -= c
            removed.append(p)
            added[(idx, b)] = c
        c = parts[1:-1].count("")
        if c:
            stats[(p1, p0)] -= c
            removed.append((p1, p0))
            added[(idx, idx)] = c
        for p in removed:
            if stats_get(p) == 0:
                del stats[p]
        stats.update(added)
        for p, c in added.items():
            heappush(heap, (-c, p))
        merges[pair] = idx
        vocab[idx] = vocab[p0] + vocab[p1]
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {top} occurrences")
