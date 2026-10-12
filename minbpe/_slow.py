"""Rarely used code, loaded on first use (off the package import path)."""
from heapq import heappop, heappush

from .base import merge, pop_top_pairs


def _update_stats(stats, old, new, pair, idx):
    p0, p1 = pair
    get = stats.get
    n_old = len(old)
    k = new.count(idx)
    index = new.index
    removed = [pair]
    rapp = removed.append
    added = set() # pairs whose count went up
    aadd = added.add
    prev = -2
    q = -1
    for m in range(k):
        q = index(idx, q + 1)
        j = q + m # position of this occurrence in old
        if q and prev != q - 1:
            a = new[q-1]
            p = (a, p0)
            stats[p] -= 1
            rapp(p)
            p = (a, idx)
            stats[p] = get(p, 0) + 1
            aadd(p)
        if j + 2 < n_old:
            p = (p1, old[j+2])
            stats[p] -= 1
            rapp(p)
            p = (idx, new[q+1])
            stats[p] = get(p, 0) + 1
            aadd(p)
        prev = q
    stats[pair] -= k
    for p in removed:
        if get(p) == 0:
            del stats[p]
    return added


def train_ints(ids, num_merges, stats, heap, merges, vocab, verbose):
    stats_get = stats.get
    for i in range(num_merges):
        top, cands = pop_top_pairs(heap, stats_get, heappush, heappop)
        if len(cands) == 1:
            pair = cands[0]
        else:
            cands = set(cands)
            for p in zip(ids, ids[1:]):
                if p in cands:
                    pair = p
                    break
        idx = 256 + i
        new = merge(ids, pair, idx)
        for p in _update_stats(stats, ids, new, pair, idx):
            heappush(heap, (-stats[p], p))
        ids = new
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {top} occurrences")


def regex_decode(self, ids):
    part_bytes = []
    for idx in ids:
        if idx in self.vocab:
            part_bytes.append(self.vocab[idx])
        elif idx in self.inverse_special_tokens:
            part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
        else:
            raise ValueError(f"invalid token id: {idx}")
    return b"".join(part_bytes).decode("utf-8", errors="replace")


def encode_plain(self, text_chunks, cache, inf):
    ids = []
    cache_get = cache.get
    encode_chunk = self._encode_chunk
    for chunk in text_chunks:
        chunk_ids = cache_get(chunk)
        if chunk_ids is None:
            chunk_ids = cache[chunk] = encode_chunk(chunk.encode("utf-8"), inf)
        ids.extend(chunk_ids)
    return ids
