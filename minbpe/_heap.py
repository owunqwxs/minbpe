"""Encoding-only helpers, loaded on first use (off the package import path)."""
from itertools import repeat
import sys

_UTF32 = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"


def _ords(s):
    # list(map(ord, s)), done in C (the code points as native uint32)
    return memoryview(s.encode(_UTF32, "surrogatepass")).cast("I").tolist()


def _replace_table(merges):
    """
    [(chr(p0) + chr(p1), chr(idx))] in increasing idx order, or None if the
    merges do not allow it. apply_merges always merges the present pair with
    the lowest idx; when every idx is a distinct int above both of its pair's
    tokens, a merge only creates pairs with a higher idx, so this is the same
    as replacing every pair in increasing idx order (str.replace matches left
    to right without overlaps, exactly like merge()).
    """
    if _memo[0] == merges:
        return _memo[1]
    _memo[:] = dict(merges), _build_table(merges)
    return _memo[1]


_memo = [None, None] # last (merges copy, table) of _replace_table


def _build_table(merges):
    items = sorted(merges.items(), key=lambda kv: kv[1])
    table = []
    app = table.append
    prev = -1
    for (p0, p1), idx in items:
        if (type(idx) is not int or type(p0) is not int or type(p1) is not int
                or idx <= prev or idx >= 0x110000 or p0 < 0 or p1 < 0
                or idx <= p0 or idx <= p1):
            return None
        prev = idx
        app((chr(p0) + chr(p1), chr(idx)))
    return table


def regex_encode(self, text, allowed_special):
    # body of RegexTokenizer.encode
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
    # split the text on exact matches of the special tokens (kept as parts)
    if "" in special:
        import regex as re
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
    else:
        special_chunks = _split_specials(text, special)
    ids = []
    # one chunk cache and sentinel for all ordinary parts (shared across calls)
    cache, inf = self._encode_state()
    for part in special_chunks:
        if part in special:
            ids.append(special[part])
        elif part:
            ids.extend(self._encode_ordinary(part, cache, inf))
    return ids


def apply_merges(ids, merges, inf=None):
    # see base.apply_merges (same function, kept off the import path)
    if len(ids) < 2:
        return ids
    if len(ids) >= 256:
        return _apply_merges_heap(ids, merges)
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
    # long sequences (loaded on first use), see _slow.apply_merges_heap
    from ._slow import apply_merges_heap
    return apply_merges_heap(ids, merges)
