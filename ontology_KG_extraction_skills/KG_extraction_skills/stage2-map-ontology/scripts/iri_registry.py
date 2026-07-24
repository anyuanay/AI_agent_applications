"""IRI minting registry: slug derivation, deduplication, collision handling."""
from __future__ import annotations

import re


def make_slug(surface_form: str) -> str:
    """Lowercase + replace non-alphanumeric with underscores + collapse runs."""
    s = surface_form.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


class IRIRegistry:
    """Tracks minted IRIs per (class_iri, slug), handles slug collisions.

    Registry key: (class_iri, slug_key) -> (surface_form, full_iri)

    On collision (same key, different surface form), appends _2, _3, ...
    until a free slot or a match to the same surface form is found.
    """

    def __init__(self, namespace: str) -> None:
        # Namespace is expected to end with '#', e.g. "http://ex.org/onto#"
        self._ns  = namespace.rstrip("#")
        self._reg: dict[tuple[str, str], tuple[str, str]] = {}

    def mint(self, class_iri: str, surface_form: str) -> str:
        """Return the canonical IRI for (class_iri, surface_form).

        Creates and registers it on first call; returns the same IRI on
        repeated calls with the same surface form.
        """
        slug = make_slug(surface_form)
        norm_surface = surface_form.strip().lower()
        local_name   = _local_name(class_iri)

        # Walk through the slot sequence: base, _2, _3, ...
        slot_slug = slug
        counter   = 1
        while True:
            key = (class_iri, slot_slug)
            if key not in self._reg:
                # Free slot → mint and register
                iri = f"{self._ns}#{local_name}_{slot_slug}"
                self._reg[key] = (norm_surface, iri)
                return iri
            reg_surface, reg_iri = self._reg[key]
            if reg_surface == norm_surface:
                # Same entity, repeated mention → reuse
                return reg_iri
            # Different entity at this slug → try next counter
            counter += 1
            slot_slug = f"{slug}_{counter}"


def _local_name(iri: str) -> str:
    if "#" in iri:
        return iri.split("#")[-1]
    if "/" in iri:
        return iri.split("/")[-1]
    return iri
