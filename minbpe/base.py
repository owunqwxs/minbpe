"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
try:
    from _collections import _count_elements
except ImportError: # pragma: no cover - pure python fallback
    def _count_elements(mapping, iterable):
        mapping_get = mapping.get
        for elem in iterable:
            mapping[elem] = mapping_get(elem, 0) + 1

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    # iterate consecutive elements; _count_elements does
    # counts[pair] = counts.get(pair, 0) + 1 for each pair, in C
    _count_elements(counts, zip(ids, ids[1:]))
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    p0, p1 = pair
    # jump between occurrences of pair[0] with list.index (in C) and copy the
    # untouched runs in between with slices; k = occurrences of p0 left at >= i
    k = ids.count(p0)
    if not k:
        return list(ids)
    newids = []
    extend = newids.extend
    append = newids.append
    index = ids.index
    last = len(ids) - 1
    same = p0 == p1
    i = 0
    while k:
        j = index(p0, i)
        k -= 1
        # if not at the very last position AND the pair matches, replace it
        if j < last and ids[j+1] == p1:
            extend(ids[i:j])
            append(idx)
            i = j + 2
            if same:
                k -= 1 # ids[j+1] was an occurrence of p0 too
        else:
            extend(ids[i:j+1])
            i = j + 1
    extend(ids[i:])
    return newids

def apply_merges(ids, merges, inf=None):
    """
    Repeatedly merge the pair with the lowest merge index until none applies.
    Same result as min() over get_stats(ids) keyed by merges.get(p, inf): the
    rank list is in position order, so .index() finds the first-seen pair
    among the minimal ones, and the minimal rank is the merged token itself.
    The rank list is computed once and then patched locally around each merged
    occurrence (only the pairs touching a merged token change), instead of
    being recomputed over the whole sequence after every merge.
    inf: optional "no merge" rank, must be greater than every value in merges
    (an int sentinel keeps all rank comparisons int-int, which are much
    cheaper than int-float ones); defaults to float("inf").
    """
    from ._heap import apply_merges # loaded on first use (encoding only)
    return apply_merges(ids, merges, inf)

def pop_top_pairs(heap, stats_get, heappush, heappop):
    """
    Lazy max-heap over pair counts (training): heap holds (-count, pair)
    entries, and for every pair in stats some entry has a count >= its current
    count (callers push the new count whenever a count increases; decreases
    are fixed up here when the stale entry surfaces). Returns (top, cands):
    the maximal count and every pair with that count, i.e. the same as
    max(stats.values()) and the pairs equal to it. The entries of the
    candidates are pushed back, so the heap invariant still holds afterwards.
    """
    while True:
        if not heap:
            raise ValueError("max() arg is an empty sequence")
        neg, p = heap[0]
        c = stats_get(p)
        if c == -neg:
            break
        heappop(heap)
        if c and c < -neg:
            heappush(heap, (-c, p)) # count went down: re-file it (not 0)
    top = -neg
    cands = []
    while heap and heap[0][0] == neg:
        p = heappop(heap)[1]
        c = stats_get(p)
        if c == top:
            if p not in cands:
                cands.append(p)
        elif c and c < top:
            heappush(heap, (-c, p))
    for p in cands:
        heappush(heap, (neg, p))
    return top, cands

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters (pretty printing only)
    from ._slow import replace_control_characters as f # loaded on first use
    return f(s)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        from ._slow import save # loaded on first use
        save(self, file_prefix)

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        from ._heap import load # loaded on first use (with encoding)
        load(self, model_file)
