"""Predicate mapping: surface predicate → ontology property IRI.

Algorithm (spec §Predicate Mapping):
  Step 1: lemma-match + string-match against property labels AND inverse labels.
          Direct hit → (property_iri, kind, inverted=False, high).
          Inverse hit → (property_iri, kind, inverted=True, high).
          Direct wins when both match at the same step.
  Step 2: embedding cosine over all direct+inverse label entries.
          Above threshold → high confidence.
  Step 3: LLM fallback with property list + inverse labels → low confidence.
  None   → caller drops the triple.

Caches results by normalized predicate to avoid redundant work.
"""
from __future__ import annotations

import re
from typing import Optional

from embedder import Embedder
from llm_client import GeminiClient


def _normalize_pred(surface: str, nlp=None) -> str:
    """Lowercase + spaCy lemmatize (falls back to lowercase-only)."""
    s = surface.strip().lower()
    if nlp is None:
        return s
    doc = nlp(s)
    return " ".join(tok.lemma_ for tok in doc)


# ── pre-build property embedding table ───────────────────────────────────────

class PropertyEmbeddingTable:
    """Pre-computed embeddings for property labels + inverse labels.

    Each entry: (label_text, property_iri, is_inverse_match)
    is_inverse_match=True means the label belongs to the declared inverse,
    so a match on this entry sets inverted=True and predicate_iri=property_iri.
    """

    def __init__(self, properties: list[dict], embedder: Embedder) -> None:
        self._entries: list[tuple[str, str, bool]] = []
        self._embs:    list[list[float]]           = []

        texts: list[str] = []
        for prop in properties:
            # Direct labels
            for lbl in [prop["label"]] + prop.get("aliases", []):
                self._entries.append((lbl, prop["iri"], False))
                texts.append(lbl)
            # Inverse labels (if any)
            for lbl in prop.get("inverse_labels", []) + prop.get("inverse_aliases", []):
                self._entries.append((lbl, prop["iri"], True))
                texts.append(lbl)

        print(f"  [pred_mapper] pre-embedding {len(texts)} property label texts …")
        self._embs = embedder.embed_batch(texts)

    def best_match(
        self,
        query_text: str,
        embedder: Embedder,
    ) -> tuple[str, bool, float]:
        """Return (property_iri, is_inverse, cosine_score)."""
        q_emb = embedder.embed(query_text)
        best_idx, best_score = embedder.top_match(q_emb, self._embs)
        _, prop_iri, is_inv = self._entries[best_idx]
        return prop_iri, is_inv, best_score


# ── predicate mapper ──────────────────────────────────────────────────────────

class PredicateMapper:
    """Maps surface predicates to ontology property IRIs.

    Returns (property_iri, kind, inverted, confidence) or None.
    """

    def __init__(
        self,
        properties: list[dict],
        embedder: Embedder,
        gemini: GeminiClient,
        nlp=None,
        threshold: float = 0.45,
    ) -> None:
        self._table     = PropertyEmbeddingTable(properties, embedder)
        self._embedder  = embedder
        self._gemini    = gemini
        self._nlp       = nlp
        self._threshold = threshold
        self._props     = properties
        self._prop_map  = {p["iri"]: p for p in properties}

        # Build normalized string-match index for Step 1
        # key: normalized_label -> (property_iri, is_inverse)
        self._label_index: dict[str, tuple[str, bool]] = {}
        for prop in properties:
            for lbl in [prop["label"]] + prop.get("aliases", []):
                norm = _normalize_pred(lbl, nlp)
                if norm and norm not in self._label_index:
                    self._label_index[norm] = (prop["iri"], False)
            for lbl in prop.get("inverse_labels", []) + prop.get("inverse_aliases", []):
                norm = _normalize_pred(lbl, nlp)
                if norm and norm not in self._label_index:
                    self._label_index[norm] = (prop["iri"], True)

        # Cache: normalized_surface -> (property_iri, kind, inverted, confidence) | None
        self._cache: dict[str, Optional[tuple[str, str, bool, str]]] = {}

        self.stats = {
            "string_hit": 0, "embedding_hit": 0,
            "llm_hit": 0, "llm_miss": 0, "cache_hit": 0,
        }

    def _kind(self, prop_iri: str) -> str:
        return self._prop_map.get(prop_iri, {}).get("kind", "object_property")

    def map(
        self,
        surface: str,
        sentence: str,
    ) -> Optional[tuple[str, str, bool, str]]:
        """Return (property_iri, kind, inverted, confidence) or None."""
        norm_key = _normalize_pred(surface, self._nlp)

        if norm_key in self._cache:
            self.stats["cache_hit"] += 1
            return self._cache[norm_key]

        # Step 1: string / lemma match
        # Check both the normalized surface AND the original lowercase
        for try_key in [norm_key, surface.strip().lower()]:
            if try_key in self._label_index:
                prop_iri, is_inv = self._label_index[try_key]
                result = (prop_iri, self._kind(prop_iri), is_inv, "high")
                self.stats["string_hit"] += 1
                self._cache[norm_key] = result
                return result

        # Step 2: embedding
        query = f"{surface} — {sentence}"
        prop_iri, is_inv, score = self._table.best_match(query, self._embedder)
        if score >= self._threshold:
            result = (prop_iri, self._kind(prop_iri), is_inv, "high")
            self.stats["embedding_hit"] += 1
            self._cache[norm_key] = result
            return result

        # Step 3: LLM
        if self._gemini is None:
            self.stats["llm_miss"] += 1
            self._cache[norm_key] = None
            return None

        llm_props_for_prompt = [
            {
                "iri":            p["iri"],
                "label":          p["label"],
                "kind":           p["kind"],
                "domain_label":   _local(p.get("domain", "") or ""),
                "range_label":    _local(p.get("range",  "") or ""),
                "inverse_of":     p.get("inverse_of"),
                "inverse_labels": p.get("inverse_labels", []),
            }
            for p in self._props
        ]
        llm_result = self._gemini.map_predicate(surface, sentence, llm_props_for_prompt)
        if llm_result:
            llm_iri, direction = llm_result
            # Validate
            known = {p["iri"] for p in self._props}
            if llm_iri not in known:
                local_matches = [p["iri"] for p in self._props
                                 if p["local_name"] == llm_iri.lstrip(":")]
                llm_iri = local_matches[0] if local_matches else None
            if llm_iri:
                is_inv = (direction == "inverse")
                result = (llm_iri, self._kind(llm_iri), is_inv, "low")
                self.stats["llm_hit"] += 1
                self._cache[norm_key] = result
                return result

        self.stats["llm_miss"] += 1
        self._cache[norm_key] = None
        return None


def _local(iri: str) -> str:
    if "#" in iri:
        return iri.split("#")[-1]
    if "/" in iri:
        return iri.split("/")[-1]
    return iri
