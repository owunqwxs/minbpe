"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base import Tokenizer, get_stats, merge


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

        # input text preprocessing: identical chunks always evolve identically,
        # so keep each unique chunk once (in first-occurrence order, which keeps
        # the pair insertion order and thus max() tie-breaking unchanged) and
        # weight it by its frequency
        weights = {}
        for ch in text_chunks:
            weights[ch] = weights.get(ch, 0) + 1
        ids = [list(ch.encode("utf-8")) for ch in weights]
        counts = list(weights.values())
        # chunks with fewer than 2 ids contribute no pairs; drop them
        keep = [j for j in range(len(ids)) if len(ids[j]) >= 2]
        ids = [ids[j] for j in keep]
        counts = [counts[j] for j in keep]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            get = stats.get
            for chunk_ids, w in zip(ids, counts):
                for p in zip(chunk_ids, chunk_ids[1:]):
                    stats[p] = get(p, 0) + w
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            a, b = pair
            new_ids = []
            new_counts = []
            for chunk_ids, w in zip(ids, counts):
                if a in chunk_ids:
                    chunk_ids = merge(chunk_ids, pair, idx)
                    if len(chunk_ids) < 2:
                        continue
                new_ids.append(chunk_ids)
                new_counts.append(w)
            ids = new_ids
            counts = new_counts
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

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
        # fast path: one combined lookup table (vocab takes precedence over specials)
        if self.inverse_special_tokens:
            table = {k: v.encode("utf-8") for k, v in self.inverse_special_tokens.items()}
            table.update(self.vocab)
        else:
            table = self.vocab
        try:
            return b"".join(map(table.__getitem__, ids)).decode("utf-8", errors="replace")
        except (KeyError, TypeError):
            pass  # fall through to the checked loop, which raises the proper error
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

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        merges = self.merges
        get = merges.get
        inf = float("inf")
        key = lambda p: get(p, inf)
        while len(ids) >= 2:
            # find the pair with the lowest merge index (merge ranks are unique,
            # so the first occurrence of the minimum is the same pair as before)
            pair = min(zip(ids, ids[1:]), key=key)
            idx = get(pair)
            if idx is None:
                break # nothing else can be merged anymore
            # merge all occurrences of the best pair, left to right
            a, b = pair
            newids = []
            append = newids.append
            n = len(ids)
            i = 0
            while i < n:
                x = ids[i]
                if x == a and i < n - 1 and ids[i + 1] == b:
                    append(idx)
                    i += 2
                else:
                    append(x)
                    i += 1
            ids = newids
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined;
        # identical chunks encode identically, so memoize within this call
        ids = []
        extend = ids.extend
        cache = {}
        encode_chunk = self._encode_chunk
        for chunk in text_chunks:
            chunk_ids = cache.get(chunk)
            if chunk_ids is None:
                chunk_ids = encode_chunk(chunk.encode("utf-8"))
                cache[chunk] = chunk_ids
            extend(chunk_ids)
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
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
