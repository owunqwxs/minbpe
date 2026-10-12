"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from collections import Counter

from .base import Tokenizer, get_stats, merge


def _merge_fast(ids, pair, idx):
    # same result as base.merge (greedy left-to-right), but jumps between
    # occurrences of pair[0] with list.index and copies runs with slices
    a, b = pair
    n = len(ids)
    out = []
    extend = out.extend
    append = out.append
    find = ids.index
    i = 0
    while True:
        try:
            j = find(a, i)
        except ValueError:
            extend(ids[i:])
            return out
        if j + 1 < n and ids[j + 1] == b:
            extend(ids[i:j])
            append(idx)
            i = j + 2
        else:
            extend(ids[i:j + 1])
            i = j + 1


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
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            # (Counter keeps first-occurrence order, same as get_stats)
            stats = Counter(zip(ids, ids[1:]))
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = _merge_fast(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(map(self.vocab.__getitem__, ids))
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        get = self.merges.get
        inf = float("inf")
        key = lambda p: get(p, inf)
        while len(ids) >= 2:
            # find the pair with the lowest merge index (merge ranks are unique,
            # so the first occurrence of the minimum is the same pair as before)
            pair = min(zip(ids, ids[1:]), key=key)
            idx = get(pair)
            if idx is None:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            ids = merge_fast(ids, pair[0], pair[1], idx)
        return ids


def merge_fast(ids, a, b, idx):
    """Same result as merge(ids, (a, b), idx), using C-level search and slicing."""
    out = []
    extend = out.extend
    append = out.append
    index = ids.index
    n = len(ids)
    i = 0
    try:
        while True:
            j = index(a, i)
            if j + 1 < n and ids[j + 1] == b:
                extend(ids[i:j])
                append(idx)
                i = j + 2
            else:
                extend(ids[i:j + 1])
                i = j + 1
    except ValueError:
        extend(ids[i:])
    return out
