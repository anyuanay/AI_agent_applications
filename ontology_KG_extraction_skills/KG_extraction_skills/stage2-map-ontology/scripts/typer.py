"""Entity typing: map a surface form to an ontology class IRI.

Used for both subject typing and object typing (object_property path).

Algorithm (spec §Subject Typing):
  1. Embed "[surface] — [sentence]" and cosine against pre-built class label embeddings.
     If top score ≥ threshold → high confidence.
  2. Else escalate to LLM classifier → low confidence.
  3. If LLM returns None → unmapped (caller drops the triple).

Results are cached by surface form to avoid redundant embedding/LLM calls
across the many triples that share the same entity mention.
"""
from __future__ import annotations

from typing import Optional

from embedder import Embedder
from iri_registry import IRIRegistry
from llm_client import GeminiClient


# ── pre-build the class embedding table ──────────────────────────────────────

class ClassEmbeddingTable:
    """Pre-computed embeddings for all class labels + aliases.

    Each entry maps one label string to a class IRI. During lookup, the
    query "[surface] — [sentence]" is compared against all entries; the
    highest-scoring class IRI wins.
    """

    def __init__(self, classes: list[dict], embedder: Embedder) -> None:
        self._entries: list[tuple[str, str]] = []   # (label_text, class_iri)
        self._embs:    list[list[float]]     = []

        texts: list[str] = []
        for cls in classes:
            self._entries.append((cls["label"], cls["iri"]))
            texts.append(cls["label"])
            for alias in cls.get("aliases", []):
                self._entries.append((alias, cls["iri"]))
                texts.append(alias)

        print(f"  [typer] pre-embedding {len(texts)} class label texts …")
        self._embs = embedder.embed_batch(texts)

    def best_match(
        self,
        query_text: str,
        embedder: Embedder,
    ) -> tuple[str, float]:
        """Return (class_iri, cosine_score) of the best matching class."""
        q_emb = embedder.embed(query_text)
        best_idx, best_score = embedder.top_match(q_emb, self._embs)
        return self._entries[best_idx][1], best_score

    def all_classes_for_llm(self, classes: list[dict]) -> list[dict]:
        """Return a condensed version of classes for the LLM prompt."""
        return [
            {"iri": c["iri"], "label": c["label"], "comment": c.get("comment")}
            for c in classes
        ]


# ── entity typer ─────────────────────────────────────────────────────────────

class EntityTyper:
    """Maps surface entity mentions to ontology class IRIs.

    Caches results by surface form so the same mention is resolved once.
    """

    def __init__(
        self,
        classes: list[dict],
        embedder: Embedder,
        gemini: GeminiClient,
        iri_registry: IRIRegistry,
        threshold: float = 0.50,
    ) -> None:
        self._table     = ClassEmbeddingTable(classes, embedder)
        self._embedder  = embedder
        self._gemini    = gemini
        self._registry  = iri_registry
        self._threshold = threshold
        self._classes   = classes

        # Cache: surface_form.lower() -> (class_iri, confidence) | None
        self._cache: dict[str, Optional[tuple[str, str]]] = {}

        self.stats = {
            "embedding_hit": 0, "llm_hit": 0,
            "llm_miss": 0, "cache_hit": 0,
        }

    def type_entity(
        self,
        surface: str,
        sentence: str,
    ) -> Optional[tuple[str, str, str]]:
        """Return (individual_iri, class_iri, confidence) or None.

        Mints the individual IRI via the registry on success.
        """
        cache_key = surface.strip().lower()

        if cache_key in self._cache:
            self.stats["cache_hit"] += 1
            cached = self._cache[cache_key]
            if cached is None:
                return None
            class_iri, conf = cached
            ind_iri = self._registry.mint(class_iri, surface)
            return ind_iri, class_iri, conf

        # Step 1: embedding
        query = f"{surface} — {sentence}"
        class_iri, score = self._table.best_match(query, self._embedder)

        if score >= self._threshold:
            self.stats["embedding_hit"] += 1
            self._cache[cache_key] = (class_iri, "high")
            ind_iri = self._registry.mint(class_iri, surface)
            return ind_iri, class_iri, "high"

        # Step 2: LLM fallback
        if self._gemini is None:
            self.stats["llm_miss"] += 1
            self._cache[cache_key] = None
            return None

        llm_iri = self._gemini.classify_entity(
            surface, sentence,
            self._table.all_classes_for_llm(self._classes),
        )
        if llm_iri:
            # Validate the returned IRI is a known class
            known = {c["iri"] for c in self._classes}
            if llm_iri not in known:
                # Try matching local name
                local_matches = [c["iri"] for c in self._classes
                                 if c["local_name"] == llm_iri.lstrip(":")]
                llm_iri = local_matches[0] if local_matches else None

        if llm_iri:
            self.stats["llm_hit"] += 1
            self._cache[cache_key] = (llm_iri, "low")
            ind_iri = self._registry.mint(llm_iri, surface)
            return ind_iri, llm_iri, "low"

        self.stats["llm_miss"] += 1
        self._cache[cache_key] = None
        return None
