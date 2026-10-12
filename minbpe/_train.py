"""BasicTokenizer training, loaded on first use (off the encoding import path)."""
from heapq import heapify, heappop, heappush
from operator import itemgetter

from .base import get_stats, merge, _count_elements

_last = itemgetter(-1)
_first = itemgetter(0)

def pop_top_pairs(heap, stats_get, heappush, heappop):
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

def regex_train(self, text, vocab_size, verbose=False):
    # body of RegexTokenizer.train (ties: earliest first occurrence, as max())
    from collections import defaultdict
    assert vocab_size >= 256
    num_merges = vocab_size - 256
    chunk_counts = {}
    _count_elements(chunk_counts, self.compiled_pattern.findall(text))
    ids = [list(ch.encode("utf-8")) for ch in chunk_counts]
    weights = list(chunk_counts.values())
    stats = {}
    stats_get = stats.get
    where = defaultdict(list)
    for k, (chunk_ids, w) in enumerate(zip(ids, weights)):
        for p in zip(chunk_ids, chunk_ids[1:]):
            stats[p] = stats_get(p, 0) + w
            where[p].append(k)
    heap = [(-c, p) for p, c in stats.items()]
    heapify(heap)
    merges = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i in range(num_merges):
        try:
            top, tied = pop_top_pairs(heap, stats_get, heappush, heappop)
        except ValueError:
            max(()) # no positive count left: max()'s own error
        pair = tied[0]
        if len(tied) > 1:
            best_k, best_j = len(ids), 0
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
                        best_k, best_j, pair = k, j, p
                    break
        idx = 256 + i
        regex_merge(ids, weights, where, stats, heap, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {top} occurrences")
    self.merges = merges # used in encode()
    self.vocab = vocab   # used in decode()


def regex_merge(ids, weights, where, stats, heap, pair, idx):
    # one merge step of RegexTokenizer.train over the chunks holding pair
    p0, p1 = pair
    added = {} # new pairs (all contain idx, so none is in stats yet)
    added_get = added.get
    fast_total = 0
    for k in where.pop(pair, ()):
        old = ids[k]
        c = old.count(p0)
        if c == 1:
            # fast path: one occurrence of p0, so at most one merge
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
        index = new.index
        q = -1
        for c in range(m):
            q = index(idx, q + 1)
            j = q + c # position of this occurrence in old
            if q:
                a = new[q-1]
                if a != idx:
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


def basic_train(self, text, vocab_size, verbose):
    # body of BasicTokenizer.train
    assert vocab_size >= 256
    num_merges = vocab_size - 256
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255
    merges = {} # (int, int) -> int
    vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    # pair counts once, then updated locally around each merged occurrence
    stats = get_stats(ids)
    heap = [(-c, p) for p, c in stats.items()] # lazy max-heap of counts
    heapify(heap)
    if vocab_size <= 0x110000:
        # every id fits in one code point: ids as a str, merging done in C
        train_str(text_bytes.decode("latin-1"), num_merges, stats, heap,
                  merges, vocab, verbose)
    else:
        from ._slow import train_ints # ids beyond one code point
        train_ints(ids, num_merges, stats, heap, merges, vocab, verbose)
    self.merges = merges # used in encode()
    self.vocab = vocab   # used in decode()


def train_str(s, num_merges, stats, heap, merges, vocab, verbose):
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
        # merge() semantics (left to right, non-overlapping) == str.replace;
        # chr(idx) is new, so the pieces between merged occurrences give the
        # neighbours (an empty inner piece: two adjacent occurrences)
        x = chr(idx)
        s = s.replace(chr(p0) + chr(p1), x)
        parts = s.split(x)
        left = {}
        _count_elements(left, map(_last, filter(None, parts[:-1])))
        right = {}
        _count_elements(right, map(_first, filter(None, parts[1:])))
        stats[pair] -= len(parts) - 1
        removed = [pair]
        added = {} # new pairs contain idx: none is in stats yet
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
