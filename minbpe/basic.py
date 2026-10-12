"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        # iteratively merge the most common pairs (loaded on first use)
        from ._train import basic_train
        basic_train(self, text, vocab_size, verbose)

    def decode(self, ids):
        # given ids (list of integers), return Python string
        from ._slow import basic_decode # loaded on first use
        return basic_decode(self, ids)

    def encode(self, text):
        # given a string text, return the token ids (loaded on first use)
        from ._heap import basic_encode
        return basic_encode(self, text)
