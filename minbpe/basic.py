"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

import sys
from heapq import heapify, heappop, heappush

from .base import Tokenizer, get_stats, apply_merges, pop_top_pairs


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        # count up the number of times every consecutive pair appears (once);
        # afterwards the counts are updated locally around each merged occurrence
        stats = get_stats(ids)
        # lazy max-heap of counts: no full scan of stats per merge
        heap = [(-c, p) for p, c in stats.items()]
        heapify(heap)
        if vocab_size <= 0x110000:
            # every token id fits in one code point: keep ids as a str so that
            # pair search and merging run in C (str.find / str.replace do the
            # same left-to-right, non-overlapping matching as merge())
            self._train_str(text_bytes.decode("latin-1"), num_merges, stats,
                            heap, merges, vocab, verbose)
            self.merges = merges
            self.vocab = vocab
            return
        # ids beyond one code point: merge a list of ints (loaded on first use)
        from ._slow import train_ints
        train_ints(ids, num_merges, stats, heap, merges, vocab, verbose)

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    @staticmethod
    def _train_str(s, num_merges, stats, heap, merges, vocab, verbose):
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
            ps = chr(p0) + chr(p1)
            # occurrences merged by merge(): left to right, non-overlapping
            find = s.find
            pos = []
            pos_app = pos.append
            j = find(ps)
            while j >= 0:
                pos_app(j)
                j = find(ps, j + 2)
            # update the counts exactly like _slow._update_stats
            n = len(s)
            k = len(pos)
            removed = [pair]
            rapp = removed.append
            added = set()
            aadd = added.add
            prev_end = -1
            for m in range(k):
                j = pos[m]
                if j and j != prev_end:
                    a = ord(s[j-1])
                    p = (a, p0)
                    stats[p] -= 1
                    rapp(p)
                    p = (a, idx)
                    stats[p] = stats_get(p, 0) + 1
                    aadd(p)
                if j + 2 < n:
                    b = ord(s[j+2])
                    p = (p1, b)
                    stats[p] -= 1
                    rapp(p)
                    if m + 1 < k and pos[m+1] == j + 2:
                        b = idx
                    p = (idx, b)
                    stats[p] = stats_get(p, 0) + 1
                    aadd(p)
                prev_end = j + 2
            stats[pair] -= k
            for p in removed:
                if stats_get(p) == 0:
                    del stats[p]
            for p in added:
                heappush(heap, (-stats[p], p))
            s = s.replace(ps, chr(idx))
            merges[pair] = idx
            vocab[idx] = vocab[p0] + vocab[p1]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {top} occurrences")

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        # repeatedly merge the pair with the lowest merge index
        # (int "no merge" sentinel above every merge index: cheap int compares)
        merges = self.merges
        if len(ids) >= 256 and merges:
            table = _replace_table(merges)
            if table is not None:
                # same result as apply_merges, with every merge done in C
                s = text_bytes.decode("latin-1")
                for ps, c in table:
                    s = s.replace(ps, c)
                return _ords(s)
        return apply_merges(ids, merges, max(merges.values(), default=0) + 1)


_UTF32 = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"


def _ords(s):
    # list(map(ord, s)), done in C (the code points as native uint32)
    return memoryview(s.encode(_UTF32, "surrogatepass")).cast("I").tolist()


def _replace_table(merges):
    """
    [(chr(p0) + chr(p1), chr(idx))] in increasing idx order, or None if the
    merges do not allow it. apply_merges always merges the present pair with
    the lowest idx; when every idx is a distinct int above both of its pair's
    tokens, a merge only creates pairs with a higher idx, so this is the same
    as replacing every pair in increasing idx order (str.replace matches left
    to right without overlaps, exactly like merge()).
    """
    if _memo[0] == merges:
        return _memo[1]
    _memo[:] = dict(merges), _build_table(merges)
    return _memo[1]


_memo = [None, None] # last (merges copy, table) of _replace_table


def _build_table(merges):
    items = sorted(merges.items(), key=lambda kv: kv[1])
    table = []
    app = table.append
    prev = -1
    for (p0, p1), idx in items:
        if (type(idx) is not int or type(p0) is not int or type(p1) is not int
                or idx <= prev or idx >= 0x110000 or p0 < 0 or p1 < 0
                or idx <= p0 or idx <= p1):
            return None
        prev = idx
        app((chr(p0) + chr(p1), chr(idx)))
    return table
