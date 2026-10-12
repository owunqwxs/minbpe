"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

from collections import defaultdict
from heapq import heapify, heappop, heappush

import regex as re
from .base import Tokenizer, merge, apply_merges, _count_elements, pop_top_pairs


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = self.compiled_pattern.findall(text)

        # input text preprocessing: identical chunks always merge identically, so
        # keep each distinct chunk once (in first-seen order) with its multiplicity.
        # Pair counts and their first-seen order are the same as over all chunks,
        # so max() picks the same pair, ties included.
        chunk_counts = {}
        _count_elements(chunk_counts, text_chunks)
        ids = [list(ch.encode("utf-8")) for ch in chunk_counts]
        weights = list(chunk_counts.values())

        # count the number of times every consecutive pair appears (once);
        # afterwards the counts are updated incrementally for changed chunks only
        # where[p]: indices of chunks that contain (or once contained) pair p,
        # possibly repeated; only those chunks can hold an occurrence of p when
        # it gets merged (chunk visiting order does not matter: chunks merge
        # independently, and a repeated or stale visit finds nothing to merge)
        stats = {}
        stats_get = stats.get
        where = defaultdict(list)
        k = 0
        for chunk_ids, w in zip(ids, weights):
            for p in zip(chunk_ids, chunk_ids[1:]):
                stats[p] = stats_get(p, 0) + w
                where[p].append(k)
            k += 1
        # lazy max-heap of (-count, pair): an entry is valid while it matches
        # the current count; every pair whose count changes is pushed again
        heap = [(-c, p) for p, c in stats.items()]
        heapify(heap)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # find the pair with the highest count; ties go to the pair whose
            # first occurrence (chunk order, then position) is earliest, which
            # is what max() over a freshly recounted dict would pick
            # (the tie-break does not depend on dict order, so the top pairs
            # come from the lazy heap; a merged pair's count drops to 0)
            try:
                top, tied = pop_top_pairs(heap, stats_get, heappush, heappop)
            except ValueError:
                max(()) # no positive count left: max()'s own error
            pair = tied[0]
            if len(tied) > 1:
                # earliest occurrence (chunk, position) among the tied pairs:
                # where[p] holds every chunk that contains p (plus stale ones),
                # so visit those in chunk order and stop once past the best
                best_k = len(ids)
                best_j = 0
                for p in tied:
                    for k in sorted(where[p]):
                        if k > best_k:
                            break
                        chunk_ids = ids[k]
                        for j, q in enumerate(zip(chunk_ids, chunk_ids[1:])):
                            if q == p:
                                break
                        else:
                            continue # stale entry: p no longer in chunk k
                        if k < best_k or j < best_j:
                            best_k = k
                            best_j = j
                            pair = p
                        break
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in the chunks that hold both
            # tokens, updating pair counts by the difference
            p0, p1 = pair
            # pairs whose count went up (all contain idx); a count that drops
            # to 0 stays 0 (new pairs contain idx), so those stay in stats
            # (new pairs contain idx, so none is in stats yet and no decrement
            # below touches one: count them apart, add them to stats at the end)
            added = {}
            added_get = added.get
            # fast-path merges of pair, subtracted from stats[pair] once at the
            # end (no fast-path neighbour pair equals pair: p0 occurs once)
            fast_total = 0
            # a merged pair never reappears, so its chunk set can be dropped
            for k in where.pop(pair, ()):
                old = ids[k]
                c = old.count(p0)
                if c == 1:
                    # fast path (most chunks): one occurrence of p0, so at
                    # most one merge; same stats updates, same order
                    j = old.index(p0)
                    e = j + 2
                    n_old = len(old)
                    if e > n_old or old[j+1] != p1:
                        continue # stale entry
                    w = weights[k]
                    fast_total += w
                    if j:
                        a = old[j-1]
                        stats[(a, p0)] -= w
                        p = (a, idx)
                        added[p] = added_get(p, 0) + w
                        where[p].append(k)
                    if e < n_old:
                        b = old[e]
                        stats[(p1, b)] -= w
                        p = (idx, b)
                        added[p] = added_get(p, 0) + w
                        where[p].append(k)
                    # splice in place (ids[k] is private to this chunk)
                    old[j] = idx
                    del old[j+1]
                    continue
                if not c:
                    continue # stale entry
                new = merge(old, pair, idx)
                n_old = len(old)
                m = n_old - len(new) # number of merged occurrences
                if not m:
                    continue
                ids[k] = new
                w = weights[k]
                stats[pair] -= w * m
                # only the pairs touching a merged occurrence change (idx does
                # not occur in old, so new[q] == idx marks exactly the merges)
                index = new.index
                q = -1
                for c in range(m):
                    q = index(idx, q + 1)
                    j = q + c # position of this occurrence in old
                    if q:
                        a = new[q-1]
                        if a != idx:
                            # (an adjacent previous occurrence already
                            # accounted for the pair between them)
                            stats[(a, p0)] -= w
                            p = (a, idx)
                            added[p] = added_get(p, 0) + w
                            where[p].append(k)
                    if j + 2 < n_old:
                        stats[(p1, old[j+2])] -= w
                        p = (idx, new[q+1])
                        added[p] = added_get(p, 0) + w
                        where[p].append(k)
            stats[pair] -= fast_total
            stats.update(added)
            for p, c in added.items():
                heappush(heap, (-c, p))
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {top} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        from ._slow import regex_decode # loaded on first use
        return regex_decode(self, ids)

    def _encode_chunk(self, text_bytes, inf=None):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        # then repeatedly merge the pair with the lowest merge index
        # (inf: optional int "no merge" rank above every merge index)
        return apply_merges(list(text_bytes), self.merges, inf)

    # chunk cache entries kept across encode calls (bounded memory)
    _CHUNK_CACHE_MAX = 1 << 16

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        from ._heap import encode_ordinary # loaded on first use
        return encode_ordinary(self, text)

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        from ._heap import regex_encode # loaded on first use
        return regex_encode(self, text, allowed_special)
