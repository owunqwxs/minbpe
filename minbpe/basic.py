"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from heapq import heapify, heappop, heappush

from operator import itemgetter

from .base import Tokenizer, get_stats, pop_top_pairs, _count_elements

_last = itemgetter(-1)
_first = itemgetter(0)


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
            # merge() semantics (left to right, non-overlapping) == str.replace;
            # chr(idx) is new, so splitting on it gives the text between merged
            # occurrences: a piece's last char is the left neighbour of the
            # next occurrence, its first char the right neighbour of the
            # previous one, and an empty inner piece means two adjacent
            # occurrences (pair (idx, idx), old (p1, p0), no left update)
            x = chr(idx)
            s = s.replace(ps, x)
            parts = s.split(x)
            # update the counts exactly like _slow._update_stats; new pairs
            # contain idx, so none is in stats yet: add them at the end
            left = {}
            _count_elements(left, map(_last, filter(None, parts[:-1])))
            right = {}
            _count_elements(right, map(_first, filter(None, parts[1:])))
            stats[pair] -= len(parts) - 1
            removed = [pair]
            added = {}
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

    def decode(self, ids):
        # given ids (list of integers), return Python string
        from ._slow import basic_decode # loaded on first use
        return basic_decode(self, ids)

    def encode(self, text):
        # given a string text, return the token ids (loaded on first use)
        from ._heap import basic_encode
        return basic_encode(self, text)
