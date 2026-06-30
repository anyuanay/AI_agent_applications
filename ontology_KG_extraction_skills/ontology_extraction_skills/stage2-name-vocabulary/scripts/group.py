"""Group synonymous candidates into one meaning.

This is the first of Stage 2's three steps (group, then select, then name) and
the hard one. Stage 1 already did string-level dedup; here we do **meaning-level**
dedup, a different and harder operation. ``IC``, ``incident commander``, and
``the commander`` are one concept even though only the last two share any
characters.

Two grouping strategies, picked by what is available:

  - **Semantic** (default): agglomerative clustering over context-enriched
    embeddings. Candidates whose embeddings sit within ``threshold`` cosine
    distance of each other (average linkage) collapse into one group. This is the
    method the article specifies: group by meaning, using embeddings and the
    surrounding context.
  - **Lexical fallback** (``--no-embeddings``): no semantic signal,
    so each candidate stays its own group except where one is an **acronym whose
    letters are the initials** of another (``IC`` with ``incident commander``,
    ``LCM`` with ``least common multiple``). This one rule is high-precision and
    needs no embeddings, so it catches the headline abbreviation case offline;
    everything else (general synonymy) is left to the semantic path.

Grouping is done **within a kind**: concepts only ever merge with concepts,
relations only with relations. The KIND from Stage 1 is never crossed.
"""

from __future__ import annotations

import numpy as np


def _connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find: collapse a list of pairwise links into groups of indices."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


_STOP = {"of", "the", "a", "an", "and", "or", "to", "in", "for", "by", "with",
         "is", "are", "be", "on", "as", "at", "number", "numbers"}


def _content_tokens(cand: dict) -> set[str]:
    """Content tokens of a candidate's dedup key (minus a few stop/over-common words).

    ``number`` is treated as too common to count as shared evidence, so
    ``whole number`` and ``prime number`` are not linked on it alone; a more
    specific shared token (``prime`` in ``prime number`` / ``prime factor``) still
    counts.
    """
    return {t for t in (cand.get("key") or "").split()
            if t not in _STOP and any(ch.isalpha() for ch in t)}


def _acronym_letters(surface: str) -> str | None:
    letters = surface.replace(".", "").strip()
    if letters.isalpha() and letters.isupper() and 2 <= len(letters) <= 6:
        return letters.casefold()
    return None


def _initials(cand: dict) -> str:
    toks = (cand.get("key") or "").split()
    return "".join(t[0] for t in toks if t).casefold()


def lexically_compatible(a: dict, b: dict) -> bool:
    """Whether two candidates may be merged on top of an embedding match.

    The guard that keeps embeddings honest. Two candidates are compatible when
    they share a content token (``real number`` / ``Real Numbers``;
    ``prime factor`` / ``prime factorization``) or one is the acronym of the
    other (``LCD`` / ``least common denominator``). This blocks the co-hyponym and
    antonym pairs that sit just as close in embedding space (``numerator`` /
    ``denominator``, ``length`` / ``width``, ``addition`` / ``subtraction``) while
    still merging genuine synonyms and abbreviations.
    """
    if _content_tokens(a) & _content_tokens(b):
        return True
    sa = a.get("canonical") or a.get("key") or ""
    sb = b.get("canonical") or b.get("key") or ""
    ac_a, ac_b = _acronym_letters(sa), _acronym_letters(sb)
    if ac_a and ac_a == _initials(b):
        return True
    if ac_b and ac_b == _initials(a):
        return True
    return False


def group_semantic(embeddings: np.ndarray, candidates: list[dict],
                   threshold: float, guard: bool = True) -> list[list[int]]:
    """Group candidates by embedding proximity, gated by a lexical guard.

    An edge links two candidates when their embeddings are within ``threshold``
    cosine distance AND (with ``guard``) they are lexically compatible. Connected
    components of that graph are the groups, so no target cluster count is needed.
    A zero (failed) embedding has no edges and stays a singleton.

    The guard is what makes context-enriched embeddings safe in a homogeneous
    domain: without it, terms like ``numerator`` and ``denominator`` merge because
    their contexts coincide; with it, only genuine synonyms and abbreviations do.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return [[i] for i in range(n)]

    sim = embeddings @ embeddings.T          # cosine sim (vectors are unit norm)
    keep = sim >= (1.0 - threshold)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if not keep[i, j]:
                continue
            if guard and not lexically_compatible(candidates[i], candidates[j]):
                continue
            edges.append((i, j))
    return _connected_components(n, edges)


def _acronym_letters(surface: str) -> str | None:
    """Return the lowercased letters of an acronym surface, or None.

    ``"IC"`` -> ``"ic"``, ``"L.C.M."`` -> ``"lcm"``. Only a single all-caps token
    block of 2 to 6 letters qualifies, so common words are never treated as
    acronyms.
    """
    letters = surface.replace(".", "").strip()
    if letters.isalpha() and letters.isupper() and 2 <= len(letters) <= 6:
        return letters.casefold()
    return None


def group_lexical(candidates: list[dict]) -> list[list[int]]:
    """Offline fallback: merge an acronym with its spelled-out expansion only.

    For every acronym candidate (``IC``), look for a multi-word candidate whose
    token initials spell it (``incident commander`` -> ``ic``) and link them.
    High precision and embedding-free, so the headline abbreviation case still
    merges offline. All other candidates stay singletons: general meaning-merge
    is the semantic path's job.
    """
    n = len(candidates)
    initials_index: dict[str, list[int]] = {}
    for i, c in enumerate(candidates):
        toks = (c.get("key") or "").split()
        if len(toks) >= 2:
            initials = "".join(t[0] for t in toks if t)
            initials_index.setdefault(initials.casefold(), []).append(i)

    edges: list[tuple[int, int]] = []
    for i, c in enumerate(candidates):
        ac = _acronym_letters(c.get("canonical") or c.get("key") or "")
        if ac:
            for j in initials_index.get(ac, []):
                if j != i:
                    edges.append((i, j))
    return _connected_components(n, edges)
