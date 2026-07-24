"""Deterministic document-structure mining for scope.

A well-written source carries its own ontology scaffold, and a textbook carries an
unusually rich one: a table of contents, per-section learning objectives,
explicit definitions ("... are called the counting numbers"), and bold key terms.
These are a curated, high-precision list of the domain's *general concepts and
topics*, the exact thing the recall-first surfacing stage drowns in worked-example
instances.

This module pulls that scaffold out with no LLM and no guessing, so the scope is
grounded in what the author themselves marked as important:

  - **topics** from the chapter outline (``1.1 Introduction to Whole Numbers``)
    and the non-boilerplate section headings,
  - **objectives** from the "you will be able to" bullet lists,
  - **defined terms** from "is/are called X" / "X is defined as" cues and from
    bold ``**Term:**`` markers.

The result is the skeleton the LLM scoper summarizes and the seed of concepts the
rest of the pipeline is scoped to.
"""

from __future__ import annotations

import re

# Headings that are document scaffolding, not domain topics.
_HEADING_NOISE = re.compile(
    r"\b(example|solution|try\s*it|how\s*to|be\s*prepared|manipulative\s*mathematics"
    r"|exercise|exercises|chapter\s*outline|learning\s*objectives"
    r"|key\s*concepts|key\s*terms|glossary|self\s*check|practice|answers?|step"
    r"|review|everyday\s*math)\b",
    re.I,
)
# Math/markup debris that should never be a topic.
_MATH_DEBRIS = re.compile(r"[$\\{}|=^_]")

# Section headings whose CONTENT is exercises / worked examples / activities, i.e.
# instances and problems, not the domain's concepts. These sections are dropped
# from the text before concept surfacing so their specific numbers and worked
# values never enter the ontology.
_EXCLUDE_SECTION = re.compile(
    r"\b(example|solution|try\s*it|how\s*to|be\s*prepared|manipulative\s*mathematics"
    r"|exercises?|practice(\s*makes\s*perfect)?|self\s*check|answers?"
    r"|chapter\s*(review|practice\s*test)|review\s*exercises?)\b",
    re.I,
)

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
_OUTLINE = re.compile(r"^\d+\.\d+\s+(.+?)\s*$", re.M)
_OBJ_CUE = re.compile(r"you will be able to", re.I)
_BOLD = re.compile(r"\*\*([^*]+?)\*\*")

# "... (is|are) called [the|a|an] TERM ..." up to a clause boundary.
_CALLED = re.compile(
    r"\b(?:is|are)\s+called\s+(?:the\s+|a\s+|an\s+)?"
    r"([a-z][a-z\- ]{2,40}?)(?=[.,;:]| and | or | which | that | when | because | so |\bof\b| if )",
    re.I,
)
_DEFINED_AS = re.compile(
    r"\b([A-Za-z][A-Za-z\- ]{2,40}?)\s+(?:is|are)\s+(?:defined as|the name)\b", re.I)

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip().strip(".,;:!?()[]{}\"'").strip()


def _is_topic(h: str) -> bool:
    """A clean, topic-like phrase: no noise/math debris, mostly letters, 2-9 words."""
    if not h or _HEADING_NOISE.search(h) or _MATH_DEBRIS.search(h):
        return False
    words = re.findall(r"[A-Za-z]+", h)
    if len(words) < 2 or len(words) > 9:   # too short = label, too long = sentence
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in h)
    if alpha / max(len(h), 1) < 0.85:      # reject tables / number-heavy lines
        return False
    return True


def headings(source: str) -> list[str]:
    return [_clean(m.group(2)) for m in _HEADING.finditer(source)]


def topic_candidates(source: str) -> list[str]:
    """Outline entries plus non-boilerplate section headings, de-duplicated."""
    topics: list[str] = []
    seen: set[str] = set()
    for m in _OUTLINE.finditer(source):
        t = _clean(m.group(1))
        if _is_topic(t) and t.casefold() not in seen:
            seen.add(t.casefold()); topics.append(t)
    for h in headings(source):
        if _is_topic(h) and h.casefold() not in seen:
            seen.add(h.casefold()); topics.append(h)
    return topics


def objectives(source: str) -> list[str]:
    """Bullet items under each 'you will be able to:' learning-objective block."""
    lines = source.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if not _OBJ_CUE.search(line):
            continue
        started = False
        for nxt in lines[i + 1:i + 15]:
            s = nxt.strip()
            if s.startswith("#"):
                break
            if not s:
                if started:          # blank after items ends the block
                    break
                continue             # skip blank lines before the items start
            s = _clean(re.sub(r"^[-*•]\s*", "", s))
            if s:
                out.append(s)
                started = True
    # De-duplicate, preserve order.
    seen, dedup = set(), []
    for o in out:
        if o.casefold() not in seen:
            seen.add(o.casefold()); dedup.append(o)
    return dedup


def defined_terms(source: str) -> list[str]:
    """Terms the text explicitly defines, the highest-precision concept seeds."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(t: str):
        t = _clean(t)
        # Keep short noun-phrase-like terms; drop obvious non-terms.
        if 1 <= len(t.split()) <= 4 and t and t.casefold() not in seen \
                and re.search(r"[a-z]", t):
            seen.add(t.casefold()); terms.append(t)

    for m in _CALLED.finditer(source):
        add(m.group(1))
    for m in _DEFINED_AS.finditer(source):
        add(m.group(1))
    # Bold "**Term:**" markers (colon-terminated bold are usually definitions).
    for m in _BOLD.finditer(source):
        b = m.group(1).strip()
        if b.endswith(":") and not _HEADING_NOISE.search(b):
            add(b.rstrip(":"))
    return terms


def content_regions(source: str) -> tuple[str, list[dict]]:
    """Split the source by headings and drop exercise / example / activity sections.

    A section runs from one heading to the next (of any level). Sections whose
    heading is an exercise, worked example, solution, activity, or practice block
    are removed, because they carry instances and specific problems, not concepts.
    The preamble before the first heading is kept. Returns (content_text,
    dropped_sections) where each dropped record has its heading and char span.
    """
    heads = [(m.start(), len(m.group(1)), _clean(m.group(2)))
             for m in _HEADING.finditer(source)]
    if not heads:
        return source, []
    n = len(heads)
    kept_spans: list[tuple[int, int]] = []
    dropped: list[dict] = []

    if source[:heads[0][0]].strip():
        kept_spans.append((0, heads[0][0]))      # preamble before first heading

    i = 0
    while i < n:
        pos, level, htext = heads[i]
        if _EXCLUDE_SECTION.search(htext):
            # Drop this section AND all nested subsections: advance to the next
            # heading at the same or a higher level (a sibling/ancestor).
            j = i + 1
            while j < n and heads[j][1] > level:
                j += 1
            block_end = heads[j][0] if j < n else len(source)
            dropped.append({"heading": htext, "start": pos, "end": block_end,
                            "chars": block_end - pos})
            i = j
        else:
            # Keep this heading and its own text up to the next heading of any
            # level; nested subsections are evaluated in later iterations.
            end = heads[i + 1][0] if i + 1 < n else len(source)
            kept_spans.append((pos, end))
            i += 1
    content = "".join(source[s:e] for s, e in kept_spans)
    return content, dropped


def parse(source: str) -> dict:
    """Return the full structural skeleton used to scope the ontology."""
    return {
        "topics": topic_candidates(source),
        "objectives": objectives(source),
        "defined_terms": defined_terms(source),
    }


def skeleton_text(struct: dict, cap: int = 80) -> str:
    """A compact text rendering of the skeleton to feed the LLM scoper."""
    parts = ["TOPICS:"]
    parts += [f"- {t}" for t in struct["topics"][:cap]]
    parts += ["", "LEARNING OBJECTIVES:"]
    parts += [f"- {o}" for o in struct["objectives"][:cap]]
    parts += ["", "DEFINED TERMS:"]
    parts += [f"- {t}" for t in struct["defined_terms"][:cap]]
    return "\n".join(parts)
