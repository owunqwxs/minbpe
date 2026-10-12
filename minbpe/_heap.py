"""Encoding-only helpers, loaded on first use (off the package import path)."""
from heapq import heapify, heappop, heappush
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


def _split_specials(text, special):
    # re.split of the escaped tokens alternation (non-empty tokens), no regex
    parts = []
    pos = 0
    while True:
        best = -1
        for k in special:
            i = text.find(k, pos)
            if i != -1 and (best == -1 or i < best):
                best, tok = i, k
        if best == -1:
            parts.append(text[pos:])
            return parts
        parts += text[pos:best], tok
        pos = best + len(tok)


def _apply_merges_heap(ids, merges):
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
