"""spaCy pass: NER + noun-chunk mentions, sentence segmentation, dep-parse triples.

Covers all four sub-tasks described in the spec:
  - NER entity extraction with mention_type classification
  - Noun-phrase chunks not already captured by NER
  - Sentence segmentation -> Sentence records
  - Dependency parse -> (nsubj, root-verb, dobj) triples with negation detection
"""
from __future__ import annotations

import re
from typing import Any

import spacy
from spacy.tokens import Doc, Span, Token

from models import DepTriple, RawMention, Sentence
from preprocess import is_kg_sentence

# spaCy entity types -> mention_type mapping
_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "EVENT", "FAC",
    "NORP", "LOC", "PRODUCT", "WORK_OF_ART",
}
_VALUE_TYPES = {
    "CARDINAL", "QUANTITY", "DATE", "TIME", "MONEY", "PERCENT",
}


def load_nlp(model: str = "en_core_web_sm") -> Any:
    return spacy.load(model)


def _ent_mention_type(label: str) -> str:
    if label in _VALUE_TYPES:
        return "value"
    return "entity"


def _token_to_sent_index(doc: Doc) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for i, sent in enumerate(doc.sents):
        for tok in sent:
            mapping[tok.i] = i
    return mapping


def _span_text(token: Token) -> str:
    """Get the canonical surface form for a dep-parse nsubj/dobj token.

    Uses the token's full subtree (left_edge to right_edge) to capture
    compound modifiers (e.g. 'Commander Diaz', 'HazmatTeam Alpha'),
    then strips whitespace.
    """
    doc = token.doc
    span = doc[token.left_edge.i : token.right_edge.i + 1]
    return span.text.strip()


def run(
    text: str,
    nlp: Any,
) -> tuple[list[Sentence], list[RawMention], list[DepTriple]]:
    """Run the full spaCy pass over *text*.

    Returns:
        sentences   — ordered Sentence records (used throughout the pipeline)
        raw_mentions — NER + noun-chunk RawMention records
        dep_triples  — dependency-parse triples
    """
    doc = nlp(text)
    tok_to_sent = _token_to_sent_index(doc)

    # ── Sentences (KG-bearing only) ───────────────────────────────────────
    sentences: list[Sentence] = []
    kg_sent_indices: set[int] = set()
    for i, sent in enumerate(doc.sents):
        if is_kg_sentence(sent):
            sentences.append(Sentence(sentence_index=i, text=sent.text.strip()))
            kg_sent_indices.add(i)

    # ── NER mentions ──────────────────────────────────────────────────────
    raw_mentions: list[RawMention] = []
    ner_token_indices: set[int] = set()

    for ent in doc.ents:
        label = ent.label_
        if label not in _ENTITY_TYPES and label not in _VALUE_TYPES:
            continue
        mtype = _ent_mention_type(label)
        sent_idx = tok_to_sent.get(ent.start, 0)
        raw_mentions.append(RawMention(
            surface_form=ent.text.strip(),
            mention_type=mtype,
            source="ner",
            sentence_index=sent_idx,
        ))
        for i in range(ent.start, ent.end):
            ner_token_indices.add(i)

    # ── Noun-chunk mentions (not already in NER) ──────────────────────────
    for chunk in doc.noun_chunks:
        # Skip pronouns (we, it, they, this used as pronoun, etc.)
        if chunk.root.pos_ == "PRON":
            continue
        # Skip if any token in the chunk is already covered by NER
        if any(tok.i in ner_token_indices for tok in chunk):
            continue
        surface = chunk.text.strip()
        if not surface:
            continue
        sent_idx = tok_to_sent.get(chunk.start, 0)
        raw_mentions.append(RawMention(
            surface_form=surface,
            mention_type="entity",
            source="noun_chunk",
            sentence_index=sent_idx,
        ))

    # ── Dependency-parse triples (KG sentences only) ─────────────────────
    dep_triples: list[DepTriple] = []

    for sent in doc.sents:
        sent_idx = tok_to_sent.get(sent.start, 0)
        if sent_idx not in kg_sent_indices:
            continue

        # Find root verb(s) in this sentence
        for tok in sent:
            if tok.dep_ != "ROOT" or tok.pos_ != "VERB":
                continue

            # Negation: any child with dep_ == "neg"
            negated = any(child.dep_ == "neg" for child in tok.children)

            # Collect nsubj and dobj children
            nsubjs = [child for child in tok.children if child.dep_ == "nsubj"]
            dobjs  = [child for child in tok.children if child.dep_ == "dobj"]

            if not nsubjs or not dobjs:
                continue

            pred_text = tok.text.strip()
            for subj_tok in nsubjs:
                if subj_tok.pos_ == "PRON":
                    continue
                subj_text = _span_text(subj_tok)
                for obj_tok in dobjs:
                    if obj_tok.pos_ == "PRON":
                        continue
                    obj_text = _span_text(obj_tok)
                    if not subj_text or not obj_text:
                        continue
                    dep_triples.append(DepTriple(
                        subject=subj_text,
                        predicate=pred_text,
                        object=obj_text,
                        sentence_index=sent_idx,
                        negated=negated,
                    ))

    return sentences, raw_mentions, dep_triples
