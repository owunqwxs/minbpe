"""Rarely used code, loaded on first use (off the package import path)."""
from heapq import heapify, heappop, heappush
from itertools import repeat

from .base import merge, render_token
from ._train import pop_top_pairs


def replace_control_characters(s):
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    import unicodedata # only needed for pretty printing, keep it off the import path
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)


def save(self, file_prefix):
    # body of Tokenizer.save: the model file (used by load()), then the vocab
    with open(file_prefix + ".model", 'w') as f:
        # the version, pattern and special tokens, then the merges
        f.write("minbpe v1\n")
        f.write(f"{self.pattern}\n")
        f.write(f"{len(self.special_tokens)}\n")
        for special, idx in self.special_tokens.items():
            f.write(f"{special} {idx}\n")
        for idx1, idx2 in self.merges:
            f.write(f"{idx1} {idx2}\n")
    save_vocab(self, file_prefix + ".vocab")


def basic_decode(self, ids):
    # body of BasicTokenizer.decode
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    return text_bytes.decode("utf-8", errors="replace")


def save_vocab(self, vocab_file):
    # the human-readable .vocab file of Tokenizer.save (lossy: errors='replace')
    inverted_merges = {idx: pair for pair, idx in self.merges.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
        for idx, token in self.vocab.items():
            s = render_token(token)
            if idx in inverted_merges:
                idx0, idx1 = inverted_merges[idx]
                s0 = render_token(self.vocab[idx0])
                s1 = render_token(self.vocab[idx1])
                f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
            else:
                f.write(f"[{s}] {idx}\n")


def apply_merges(ids, merges, inf=None):
    # see base.apply_merges (same function, kept off the import path)
    if len(ids) < 2:
        return ids
    if len(ids) >= 256:
        return apply_merges_heap(ids, merges)
    merges_get = merges.get
    if inf is None:
        inf = float("inf")
    ranks = list(map(merges_get, zip(ids, ids[1:]), repeat(inf)))
    ids = list(ids) # private copy: the single-occurrence path edits it in place
    while ranks:
        idx = min(ranks)
        if idx == inf:
            break # nothing else can be merged anymore
        j = ranks.index(idx)
        k = ranks.count(idx)
        if k == 1:
            ids[j] = idx
            del ids[j+1]
            del ranks[j]
            if j:
                ranks[j-1] = merges_get((ids[j-1], idx), inf)
            if j < len(ranks):
                ranks[j] = merges_get((idx, ids[j+1]), inf)
            continue
        p0 = ids[j]
        p1 = ids[j+1]
        last = len(ranks) - 1 # last pair position
        newids = []
        ext = newids.extend
        app = newids.append
        newranks = []
        rext = newranks.extend
        rapp = newranks.append
        index = ranks.index
        s = 0 # start of the not yet copied part of ids / ranks
        first = True
        while True:
            k -= 1
            if ids[j] == p0 and ids[j+1] == p1:
                # merge the occurrence at j (same left-to-right order as merge())
                if s < j:
                    if not first:
                        rapp(merges_get((idx, ids[s]), inf))
                    rext(ranks[s:j-1])
                    rapp(merges_get((ids[j-1], idx), inf))
                    ext(ids[s:j])
                elif not first:
                    rapp(merges_get((idx, idx), inf))
                app(idx)
                first = False
                s = j + 2
                if j < last and ranks[j+1] == idx:
                    k -= 1 # overlapping position j+1 is skipped
                start = s
            else:
                start = j + 1
            if not k:
                break
            j = index(idx, start)
        if s <= last + 1:
            rapp(merges_get((idx, ids[s]), inf))
            rext(ranks[s:])
            ext(ids[s:])
        ids = newids
        ranks = newranks
    return ids


def apply_merges_heap(ids, merges):
    """
    Same result as base.apply_merges, for long sequences: a doubly linked list
    over the positions plus a lazily validated heap of (rank, position); each
    round merges the minimal rank's first pair left to right, like merge().
    """
    merges_get = merges.get
    n = len(ids)
    val = list(ids)
    nxt = list(range(1, n + 1)) # n means "no next"
    prv = list(range(-1, n - 1)) # -1 means "no previous"
    heap = [(r, i) for i, r in enumerate(map(merges_get, zip(val, val[1:])))
            if r is not None]
    heapify(heap)
    while heap:
        r = heap[0][0]
        pair = None
        last = -1
        merged = []
        rest = []
        while heap and heap[0][0] == r:
            p = heappop(heap)[1]
            if p == last:
                continue # duplicate entry (pops come out in position order)
            last = p
            v = val[p]
            if v is None:
                continue # position was merged away
            q = nxt[p]
            if q == n:
                continue
            pr = (v, val[q])
            if merges_get(pr) != r:
                continue # stale entry
            if pair is None:
                pair = pr
            elif pr != pair:
                rest.append(p) # another pair with the same rank: later round
                continue
            val[p] = r
            val[q] = None
            q2 = nxt[q]
            nxt[p] = q2
            if q2 < n:
                prv[q2] = p
            merged.append(p)
        for p in rest:
            heappush(heap, (r, p))
        for p in merged:
            a = prv[p]
            if a >= 0:
                rk = merges_get((val[a], r))
                if rk is not None:
                    heappush(heap, (rk, a))
            b = nxt[p]
            if b < n:
                rk = merges_get((r, val[b]))
                if rk is not None:
                    heappush(heap, (rk, p))
    return [v for v in val if v is not None]


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
