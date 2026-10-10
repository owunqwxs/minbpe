"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

from itertools import compress, islice

import regex as re
from .base import Tokenizer, merge, apply_merges, _count_elements


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
        for ch in text_chunks:
            chunk_counts[ch] = chunk_counts.get(ch, 0) + 1
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
        # where[t]: indices of chunks that contain (or once contained) token t
        where = [set() for _ in range(max(vocab_size, 256))]
        for k, chunk_ids in enumerate(ids):
            for t in set(chunk_ids):
                where[t].add(k)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # find the pair with the highest count; ties go to the pair whose
            # first occurrence (chunk order, then position) is earliest, which
            # is what max() over a freshly recounted dict would pick
            # (max/count/index over a plain list of counts run in C without a
            # per-pair key call; the first maximal key is the max() result)
            vals = list(stats_values)
            top = max(vals)
            if vals.count(top) == 1:
                pair = next(islice(stats, vals.index(top), None))
            else:
                cands = set(compress(stats, map(top.__eq__, vals)))
                for chunk_ids in ids:
                    if not cands.isdisjoint(zip(chunk_ids, chunk_ids[1:])):
                        for p in zip(chunk_ids, chunk_ids[1:]):
                            if p in cands:
                                pair = p
                                break
                        break
                top = stats[pair]
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in the chunks that hold both
            # tokens, updating pair counts by the difference
            p0, p1 = pair
            touched = []
            where_idx = where[idx]
            for k in where[p0] & where[p1]:
                old = ids[k]
                new = merge(old, pair, idx)
                if len(new) == len(old):
                    continue
                ids[k] = new
                w = weights[k]
                for p in zip(old, old[1:]):
                    stats[p] -= w
                    touched.append(p)
                for p in zip(new, new[1:]):
                    stats[p] = stats_get(p, 0) + w
                where_idx.add(k)
            for p in touched:
                if stats_get(p) == 0:
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
            state = (merges, len(merges), self._no_merge_rank(), {})
            self._enc_state = state
        cache = state[3]
        if len(cache) > self._CHUNK_CACHE_MAX:
            cache.clear()
        return cache, state[2]

    def _encode_ordinary(self, text, cache, inf):
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
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
