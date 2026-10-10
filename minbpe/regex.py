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
from itertools import chain
from .base import Tokenizer, merge, apply_merges, _count_elements
from .basic import _replace_table


def _chunk_table(merges):
    # basic._replace_table, or None if the 0xff chunk separator could clash
    # with a merge (as a pair token or as a minted token)
    if not merges:
        return None
    table = _replace_table(merges)
    if table is None:
        return None
    for ps, c in table:
        if "\xff" in ps or c == "\xff":
            return None
    return table


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
        text_chunks = re.findall(self.compiled_pattern, text)

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
        stats = {}
        stats_get = stats.get
        for chunk_ids, w in zip(ids, weights):
            if w == 1:
                _count_elements(stats, zip(chunk_ids, chunk_ids[1:]))
            else:
                for p in zip(chunk_ids, chunk_ids[1:]):
                    stats[p] = stats_get(p, 0) + w
        stats_values = stats.values()
        # lazy max-heap of (-count, pair): an entry is valid while it matches
        # the current count; every pair whose count changes is pushed again
        heap = [(-c, p) for p, c in stats.items()]
        heapify(heap)
        # where[p]: indices of chunks that contain (or once contained) pair p;
        # only those chunks can hold an occurrence of p when it gets merged
        # (chunk visiting order does not matter: chunks merge independently)
        where = defaultdict(set)
        for k, chunk_ids in enumerate(ids):
            for p in zip(chunk_ids, chunk_ids[1:]):
                where[p].add(k)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # find the pair with the highest count; ties go to the pair whose
            # first occurrence (chunk order, then position) is earliest, which
            # is what max() over a freshly recounted dict would pick
            # (the tie-break does not depend on dict order, so the top pairs
            # come from the heap: drop stale entries, then pop all valid ties)
            while heap:
                c, pair = heap[0]
                if stats_get(pair) == -c:
                    break
                heappop(heap)
            else:
                max(stats_values) # empty stats: same error as before
            top = -c
            heappop(heap)
            tied = None
            while heap and heap[0][0] == c:
                q = heappop(heap)[1]
                if stats_get(q) == top:
                    if tied is None:
                        tied = {pair}
                    tied.add(q)
            if tied is not None and len(tied) > 1:
                for q in tied:
                    heappush(heap, (c, q))
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
            touched = [pair]
            rapp = touched.append
            # fast-path merges of pair, subtracted from stats[pair] once at the
            # end (no fast-path neighbour pair equals pair: p0 occurs once)
            fast_total = 0
            # a merged pair never reappears, so its chunk set can be dropped
            for k in where.pop(pair, ()):
                old = ids[k]
                c = old.count(p0)
                if c < 2:
                    # fast path (most chunks): at most one occurrence of p0,
                    # so at most one merge; same stats updates, same order
                    if not c:
                        continue # stale entry
                    j = old.index(p0)
                    n_old = len(old)
                    if j + 1 == n_old or old[j+1] != p1:
                        continue # stale entry
                    w = weights[k]
                    fast_total += w
                    if j + 2 < n_old:
                        b = old[j+2]
                    # splice in place (ids[k] is private to this chunk)
                    old[j] = idx
                    del old[j+1]
                    if j:
                        a = old[j-1]
                        p = (a, p0)
                        stats[p] -= w
                        rapp(p)
                        p = (a, idx)
                        stats[p] = stats_get(p, 0) + w
                        where[p].add(k)
                        rapp(p)
                    if j + 2 < n_old:
                        p = (p1, b)
                        stats[p] -= w
                        rapp(p)
                        p = (idx, b)
                        stats[p] = stats_get(p, 0) + w
                        where[p].add(k)
                        rapp(p)
                    continue
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
                            p = (a, p0)
                            stats[p] -= w
                            rapp(p)
                            p = (a, idx)
                            stats[p] = stats_get(p, 0) + w
                            where[p].add(k)
                            rapp(p)
                    if j + 2 < n_old:
                        p = (p1, old[j+2])
                        stats[p] -= w
                        rapp(p)
                        p = (idx, new[q+1])
                        stats[p] = stats_get(p, 0) + w
                        where[p].add(k)
                        rapp(p)
            stats[pair] -= fast_total
            for p in set(touched):
                c = stats_get(p)
                if c:
                    heappush(heap, (-c, p))
                elif c == 0:
                    del stats[p]
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
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes, inf=None):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        # then repeatedly merge the pair with the lowest merge index
        # (inf: optional int "no merge" rank above every merge index)
        return apply_merges(list(text_bytes), self.merges, inf)

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        cache, inf = self._encode_state()
        return self._encode_ordinary(text, cache, inf)

    def _no_merge_rank(self):
        # int sentinel above every merge index: keeps rank comparisons int-int
        return max(self.merges.values(), default=0) + 1

    # chunk cache entries kept across encode calls (bounded memory)
    _CHUNK_CACHE_MAX = 1 << 16

    def _encode_state(self):
        # (chunk -> ids cache, no-merge rank), shared by encode calls as long
        # as the merges stay the same: train() and load() assign a new dict,
        # which (like any other change of the merges object or its size)
        # starts a fresh cache
        merges = self.merges
        state = self.__dict__.get("_enc_state")
        if state is None or state[0] is not merges or state[1] != len(merges):
            state = (merges, len(merges), self._no_merge_rank(), {},
                     _chunk_table(merges))
            self._enc_state = state
        cache = state[3]
        if len(cache) > self._CHUNK_CACHE_MAX:
            cache.clear()
        return cache, state[2]

    def _encode_ordinary(self, text, cache, inf):
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        table = self._enc_state[4]
        if table is not None:
            new = set(text_chunks).difference(cache)
            if len(new) >= 16:
                # encode all new chunks at once, every merge done in C: the
                # chunks are joined with byte 0xff (never produced by UTF-8 and
                # in no merge pair, so no merge crosses a chunk boundary), then
                # each merge is replaced in increasing idx order (same result
                # as apply_merges per chunk, see basic._replace_table)
                new = list(new)
                s = b"\xff".join(map(str.encode, new)).decode("latin-1")
                for ps, c in table:
                    s = s.replace(ps, c)
                cache.update(zip(new, [list(map(ord, p)) for p in s.split("\xff")]))
            elif new:
                encode_chunk = self._encode_chunk
                for chunk in new:
                    cache[chunk] = encode_chunk(chunk.encode("utf-8"), inf)
            return list(chain.from_iterable(map(cache.__getitem__, text_chunks)))
        # all chunks of text are encoded separately, then results are joined
        ids = []
        # chunks repeat a lot in natural text (" the", ","): encode each distinct
        # chunk once per call (cache: chunk -> ids, shared by the caller)
        cache_get = cache.get
        encode_chunk = self._encode_chunk
        for chunk in text_chunks:
            chunk_ids = cache_get(chunk)
            if chunk_ids is None:
                chunk_bytes = chunk.encode("utf-8") # raw bytes
                chunk_ids = cache[chunk] = encode_chunk(chunk_bytes, inf)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        # one chunk cache and sentinel for all ordinary parts (shared across calls)
        cache, inf = self._encode_state()
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            elif part:
                # this is an ordinary sequence, encode it normally
                ids.extend(self._encode_ordinary(part, cache, inf))
        return ids
