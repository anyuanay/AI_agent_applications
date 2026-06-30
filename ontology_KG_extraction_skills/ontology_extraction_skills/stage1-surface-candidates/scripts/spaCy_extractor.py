"""spaCy extractor: NER + noun-phrase terms + predicate relations.

The recall floor and the hallucination-free anchor. Every candidate it emits is
a *literal span copied out of the text*, so it physically cannot invent a term
that is not there. The signals, all literal:

Concept mentions
  - **NER** (``doc.ents``)          -> named things.
  - **Noun-phrase terms** (``doc.noun_chunks`` whose head is a NOUN/PROPN) ->
    the recall workhorse; most concepts surface here.
  - **Gerund nominalizations** (``tag_ == VBG`` in a nominal dependency role) ->
    "rounding", "factoring": concepts that ``noun_chunks`` misses because their
    head is a verb form.
  - **Adjective qualities** (``pos_ == ADJ`` in an amod/acomp role), *opt-in* via
    ``adjectives=True`` -> "prime", "even", "rational": qualities of things. Off
    by default because it also pulls in generic adjectives.

Relation mentions (relations are not only verbs)
  - **Verbs** (``pos_ == VERB``)                -> "add", "divide".
  - **Verb phrases**: verb + particle ("add up") and verb + preposition
    ("divide by", "consist of"), built from the dependency parse.
  - **Copular predicates**: "be" + adjective/noun (+ preposition) ->
    "is equal to", "is divisible by", "is a multiple of".

Stop words are never surfaced on their own: every concept must keep at least one
non-stop content word, so a lone "two", "it", or "the" is dropped. Leading
determiners and pronouns are trimmed ("the foundation" -> "foundation"), but
content words that merely sit on spaCy's stop list ("whole", "first") are kept,
so "whole numbers" stays intact. Function words inside a relation predicate (the
copula and prepositions in "is divisible by") are kept on purpose, because there
the predicate is the relation.

POS (``pos_``), fine-grained TAG (``tag_``), and dependency (``dep_``) drive the
choices above: POS gates which noun-chunk heads count, TAG finds gerund concepts,
and DEP builds the verb phrases and copular predicates and the nominal roles that
qualify a gerund.

Only flat mentions with offsets are emitted: no hierarchy, no domain/range.

The document is processed in offset-tracked blocks so spaCy never sees the whole
file at once, yet every emitted offset still points into the *original* file.
All relation phrases are emitted as *contiguous* spans, so an offset always
re-slices back to the exact predicate text.
"""

from __future__ import annotations

import re
from typing import Iterator

import spacy

from candidate import (
    RawCandidate,
    SOURCE_NER,
    SOURCE_TERM,
    SOURCE_ADJ,
    SOURCE_VERB,
    KIND_CONCEPT,
    KIND_RELATION,
)
from textnorm import key_from_tokens, clean_surface


# spaCy's default cap is 1,000,000 chars; we block-split well below it, but raise
# the ceiling so a single huge block can never crash the pipe.
_MAX_LEN = 2_000_000

# Leading words trimmed from a noun phrase: determiners / possessives / pronouns
# ("the foundation" -> "foundation", "your study" -> "study"). These are
# structural, unlike content-bearing words that merely sit on spaCy's stop list
# ("whole", "first", "several"), which we must NOT trim or we mangle terms like
# "whole numbers".
_TRIM_FRONT_POS = {"DET", "PRON"}
_TRIM_FRONT_DEP = {"poss", "det", "predet"}

# A noun chunk only yields a concept if its head is one of these parts of speech
# (drops chunks headed by a pronoun, number, etc.).
_NOUN_HEAD_POS = {"NOUN", "PROPN"}

# Dependency roles in which a gerund (TAG == VBG) is acting as a noun, so it is a
# concept rather than part of a verb phrase ("Rounding helps", "after factoring").
_GERUND_NOMINAL_DEPS = {
    "nsubj", "nsubjpass", "dobj", "dative", "pobj", "attr", "oprd", "conj",
}

# Dependency roles in which an adjective expresses a quality of a thing, surfaced
# only when the opt-in `adjectives` flag is set: attributive ("PRIME number") and
# predicative ("the number is EVEN"). Stop-list adjectives ("more", "same") are
# dropped by the usual content check.
_ADJ_QUALITY_DEPS = {"amod", "acomp", "oprd"}

# Verbs too generic to be useful relation mentions on their own (as a bare verb).
_STOP_VERBS = {
    "be", "have", "do", "go", "get", "make", "use", "see", "let", "begin",
    "need", "will", "can", "may", "find",  # 'find' is borderline; keep noisy-low
}

# Dependency labels that extend a verb rightward into a verb phrase:
# particles ("add UP") and prepositions ("divide BY", "consist OF").
_VERB_EXT_DEPS = {"prt", "prep", "agent"}
# Predicate complements of a copula ("is DIVISIBLE", "is a MULTIPLE").
_COP_COMP_DEPS = {"acomp", "attr", "oprd"}
# Tokens allowed *inside* a copular predicate span between "is" and the
# preposition ("is A multiple": det=a, advmod/amod modifiers).
_COP_INNER_DEPS = {"det", "advmod", "amod", "neg", "nummod", "compound"}

# A candidate must contain at least one alphabetic run of length >= 2.
_HAS_WORD = re.compile(r"[A-Za-z]{2,}")
_BLOCK_SPLIT = re.compile(r"\n\s*\n")
# Markdown / HTML debris that a noun chunk sometimes swallows.
_HTML_ENTITY = re.compile(r"&[a-z]{2,5};?", re.IGNORECASE)
# Bare HTML-entity remnants left after spaCy splits "&gt;" into "&","gt",";".
_ENTITY_REMNANTS = {"gt", "lt", "amp", "quot", "nbsp", "apos"}


def load_nlp(model: str = "en_core_web_sm"):
    """Load spaCy with only what Stage 1 needs (tagger, parser, NER, lemmatizer)."""
    nlp = spacy.load(model)
    nlp.max_length = _MAX_LEN
    return nlp


def _iter_blocks(text: str) -> Iterator[tuple[str, int]]:
    """Yield ``(block_text, base_offset)`` so local spans map back to the file.

    Splitting on blank lines keeps blocks small (good for memory and for the
    parser) while ``base_offset`` lets us translate every entity offset into a
    global offset into the original input.
    """
    pos = 0
    for piece in _BLOCK_SPLIT.split(text):
        # Recover the true offset of this piece (split drops the delimiter).
        idx = text.find(piece, pos)
        if idx < 0:
            idx = pos
        if piece.strip():
            yield piece, idx
        pos = idx + len(piece)


def _looks_like_term(surface: str) -> bool:
    """Junk filter that still preserves recall.

    Rejects markdown debris, pure numbers/symbols, and one-character noise, but
    keeps anything with a real word in it. This is a *junk* filter, not a
    *precision* gate: precision is recovered in a later step, never here.
    """
    if not surface or len(surface) < 2:
        return False
    if not _HAS_WORD.search(surface):
        return False
    if surface.casefold() in _ENTITY_REMNANTS:
        return False
    # markdown table rows / cell fragments leak in as noun chunks: drop them.
    if "|" in surface:
        return False
    # HTML entities (&gt; &amp; ...) and other markup debris.
    if "&" in surface and _HTML_ENTITY.search(surface):
        return False
    # markdown / file debris
    if surface.startswith(("#", "!", "<", "http", "www.", "img-", "&")):
        return False
    if surface.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
        return False
    return True


def _trim_edges(toks):
    """Trim leading determiners/possessives/pronouns and edge punctuation.

    "the foundation" -> "foundation"; "your study" -> "study"; "a whole number,"
    -> "whole number". Only *structural* leading words are removed; content words
    that happen to be on spaCy's stop list ("whole", "first") are kept, so
    "whole numbers" stays intact. Interior tokens are never touched.
    """
    toks = [t for t in toks if not t.is_space]
    while toks and (
        toks[0].pos_ in _TRIM_FRONT_POS or toks[0].dep_ in _TRIM_FRONT_DEP
    ):
        toks = toks[1:]
    while toks and toks[0].is_punct:
        toks = toks[1:]
    while toks and toks[-1].is_punct:
        toks = toks[:-1]
    return toks


def _has_content(toks) -> bool:
    """True iff at least one token is a non-stop word (not a bare number/symbol).

    This is the rule that keeps stop words from being surfaced on their own: a
    lone "two", "three", "the", or "it" has no content token and is dropped,
    while a phrase like "whole numbers" survives on its head ("numbers").
    """
    return any((not t.is_stop) and _HAS_WORD.search(t.text) for t in toks)


def _emit_concept(out, text, base, toks, source, ner_label=None):
    """Emit a concept candidate from a token span after stop-word trimming.

    Shared by NER, noun chunks, and gerunds so the stop-word rule and provenance
    handling are identical for all three. Offsets index the original file once
    ``base`` is added.
    """
    toks = _trim_edges(toks)
    if not toks or not _has_content(toks):
        return
    start = toks[0].idx
    end = toks[-1].idx + len(toks[-1].text)
    surface = clean_surface(text[start:end])
    if not _looks_like_term(surface):
        return
    key = key_from_tokens(toks)
    if not key:
        return
    out.append(RawCandidate(
        text=surface, kind=KIND_CONCEPT, source=source, key=key,
        start=base + start, end=base + end, ner_label=ner_label,
    ))


def _emit_relation(out, doc, text, base, start_i, end_i, seen):
    """Emit a relation candidate for the token range [start_i, end_i].

    The range is trimmed of edge punctuation/space, keyed by lemma, and only
    emitted if it survives the junk filter. Offsets stay contiguous so the
    occurrence text re-slices exactly to the predicate.
    """
    if (start_i, end_i) in seen:
        return
    toks = [doc[i] for i in range(start_i, end_i + 1)]
    while toks and (toks[0].is_space or toks[0].is_punct):
        toks = toks[1:]
    while toks and (toks[-1].is_space or toks[-1].is_punct):
        toks = toks[:-1]
    if not toks:
        return
    key = key_from_tokens(toks)
    if not key or key in _STOP_VERBS or not _HAS_WORD.search(key):
        return
    s = toks[0].idx
    e = toks[-1].idx + len(toks[-1].text)
    surface = clean_surface(text[s:e])
    if not _looks_like_term(surface):
        return
    seen.add((start_i, end_i))
    out.append(RawCandidate(
        text=surface, kind=KIND_RELATION, source=SOURCE_VERB, key=key,
        start=base + s, end=base + e,
    ))


def _cop_span_clean(be, comp, prep, left_i, right_i) -> bool:
    """True iff every token in [left_i, right_i] belongs to the copular predicate
    (no foreign word, e.g. an object, sneaked in)."""
    allowed = {be.i, comp.i, prep.i}
    for parent in (be, comp):
        for c in parent.children:
            if c.dep_ in _COP_INNER_DEPS:
                allowed.add(c.i)
    return all(i in allowed for i in range(left_i, right_i + 1))


def _extract_relations(out, doc, text, base):
    """Surface relation mentions: bare verbs, verb phrases, copular predicates.

    Relations are not only verbs. We surface, all as contiguous literal spans:
      - the bare verb                         ("divide", "add")
      - verb + particle / preposition         ("add up", "divide by", "consist of")
      - copular adjective/noun predicates     ("is equal to", "is a multiple of")
    """
    seen: set[tuple[int, int]] = set()
    for tok in doc:
        # (A) verbs and verb phrases ---------------------------------------
        if tok.pos_ == "VERB" and not tok.is_space and not tok.is_stop:
            lemma = (tok.lemma_ or tok.text).strip().casefold()
            if lemma and lemma not in _STOP_VERBS and _HAS_WORD.search(lemma):
                # the bare verb (always)
                _emit_relation(out, doc, text, base, tok.i, tok.i, seen)
                # verb phrase: extend contiguously over particles/prepositions
                ext_idx = {c.i for c in tok.children if c.dep_ in _VERB_EXT_DEPS}
                end_i = tok.i
                while (end_i + 1) in ext_idx:
                    end_i += 1
                if end_i > tok.i:
                    _emit_relation(out, doc, text, base, tok.i, end_i, seen)

        # (B) copular predicates ("is divisible by", "is a multiple of") ----
        # spaCy's English model heads the copula on "be" (OntoNotes scheme):
        # "is" is the head, the predicate is an acomp/attr child, and the
        # preposition hangs off either.
        if tok.lemma_ == "be" and tok.pos_ in ("AUX", "VERB") and not tok.is_space:
            comp = next((c for c in tok.children
                         if c.dep_ in _COP_COMP_DEPS and c.i > tok.i), None)
            if comp is not None:
                prep = next((c for c in comp.children
                             if c.dep_ in ("prep", "agent")), None)
                if prep is None:
                    prep = next((c for c in tok.children
                                 if c.dep_ in ("prep", "agent") and c.i > comp.i),
                                None)
                # Require a preposition so we keep relational predicates
                # ("is equal to") and skip bare attributes ("is prime").
                if prep is not None and tok.i < prep.i:
                    if _cop_span_clean(tok, comp, prep, tok.i, prep.i):
                        _emit_relation(out, doc, text, base, tok.i, prep.i, seen)


def extract(nlp, text: str, base: int = 0,
            adjectives: bool = False) -> list[RawCandidate]:
    """Run all spaCy signals over one block of text.

    ``base`` is added to every local offset so the returned offsets index the
    original file. When ``adjectives`` is true, attributive/predicative
    adjectives are also surfaced as quality concepts (tagged ``adj``).
    """
    out: list[RawCandidate] = []
    doc = nlp(text)

    # --- NER: named things become concept mentions -------------------------
    for ent in doc.ents:
        _emit_concept(out, text, base, list(ent), SOURCE_NER, ner_label=ent.label_)

    # --- Noun-phrase term extraction: the recall workhorse -----------------
    for chunk in doc.noun_chunks:
        # POS gate: only chunks headed by a real noun/proper noun (drops chunks
        # headed by a pronoun, number, etc.).
        if chunk.root.pos_ not in _NOUN_HEAD_POS:
            continue
        _emit_concept(out, text, base, list(chunk), SOURCE_TERM)

    # --- Gerund nominalizations: concepts noun_chunks misses ----------------
    # A VBG in a nominal dependency role ("Rounding helps", "after factoring") is
    # a concept whose head is a verb form, so noun_chunks never yields it.
    for tok in doc:
        if (tok.tag_ == "VBG" and tok.dep_ in _GERUND_NOMINAL_DEPS
                and not tok.is_stop):
            _emit_concept(out, text, base, [tok], SOURCE_TERM)

    # --- Adjective qualities (opt-in): "prime", "even", "rational" ----------
    if adjectives:
        for tok in doc:
            if (tok.pos_ == "ADJ" and tok.dep_ in _ADJ_QUALITY_DEPS
                    and not tok.is_stop):
                _emit_concept(out, text, base, [tok], SOURCE_ADJ)

    # --- Relations: verbs, verb phrases, and copular predicates ------------
    _extract_relations(out, doc, text, base)

    return out


def extract_document(nlp, text: str,
                     adjectives: bool = False) -> list[RawCandidate]:
    """Run the spaCy extractor over a whole document, block by block."""
    out: list[RawCandidate] = []
    for block, base in _iter_blocks(text):
        out.extend(extract(nlp, block, base=base, adjectives=adjectives))
    return out
