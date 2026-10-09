"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from itertools import compress, islice

from .base import Tokenizer, get_stats, merge, apply_merges


def _update_stats(stats, old, new, pair, idx):
    """
    Update pair counts (stats of old) in place so they become the stats of new,
    where new = merge(old, pair, idx) and idx does not occur in old. Only pairs
    touching a merged occurrence change; zero counts are removed.
    """
    p0, p1 = pair
    get = stats.get
    n_old = len(old)
    n_new = len(new)
    k = new.count(idx)
    index = new.index
    removed = [pair]
    rapp = removed.append
    prev = -2
    q = -1
    for m in range(k):
        q = index(idx, q + 1)
        j = q + m # position of this occurrence in old
        if q and prev != q - 1:
            # left neighbour is untouched (an adjacent previous occurrence
            # already accounted for the pair between them)
            a = new[q-1]
            p = (a, p0)
            stats[p] -= 1
            rapp(p)
            p = (a, idx)
            stats[p] = get(p, 0) + 1
        if j + 2 < n_old:
            p = (p1, old[j+2])
            stats[p] -= 1
            rapp(p)
            p = (idx, new[q+1])
            stats[p] = get(p, 0) + 1
        prev = q
    stats[pair] -= k
    for p in removed:
        if get(p) == 0:
            del stats[p]


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
        stats_get = stats.get
        stats_values = stats.values()
        for i in range(num_merges):
            # find the pair with the highest count; ties go to the pair whose
            # first occurrence in ids is earliest, which is what max() over a
            # freshly recounted dict (insertion = first-seen order) would pick
            # (max/count/index over a plain list of counts run in C without a
            # per-pair key call; the first maximal key is the max() result)
            vals = list(stats_values)
            top = max(vals)
            if vals.count(top) == 1:
                pair = next(islice(stats, vals.index(top), None))
            else:
                cands = set(compress(stats, map(top.__eq__, vals)))
                for p in zip(ids, ids[1:]):
                    if p in cands:
                        pair = p
                        break
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            new = merge(ids, pair, idx)
            _update_stats(stats, ids, new, pair, idx)
            ids = new
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {top} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

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
        return apply_merges(ids, merges, max(merges.values(), default=0) + 1)
