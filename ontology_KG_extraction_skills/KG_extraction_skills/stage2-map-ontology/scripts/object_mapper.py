"""Object mapping: route to entity typing or literal parsing based on predicate kind.

For object_property predicates  → type the object as a class instance (same as
                                   subject typing via EntityTyper).
For datatype_property predicates → parse the object surface as a typed literal:
                                    integer, decimal, date, or string.
                                    Unit strings (e.g. "47 psi") are split into
                                    numeric value + discarded unit annotation.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from typer import EntityTyper


# ── literal parsing ───────────────────────────────────────────────────────────

# Patterns tried in order (most specific first)
_INT_RE    = re.compile(r"^-?\d{1,3}(?:,\d{3})*$")           # 1,000 or 42
_FLOAT_RE  = re.compile(r"^-?\d+(?:\.\d+)?$")                 # 3.14
_DATE_RE   = re.compile(r"^\d{4}-\d{2}-\d{2}$")              # 2024-01-15
_UNIT_RE   = re.compile(r"^(-?\d+(?:\.\d+)?)\s+([A-Za-z%°]+)$")  # "47 psi"
_PCT_RE    = re.compile(r"^(-?\d+(?:\.\d+)?)\s*%$")           # "3.5%"


def parse_literal(surface: str) -> Optional[tuple[Any, str]]:
    """Parse surface form into (value, xsd_type_str).

    Returns None only if the surface string is empty. Every non-empty
    string has at least an xsd:string interpretation.
    """
    s = surface.strip()
    if not s:
        return None

    # Percentage (before generic float, "3.5%" should not match float)
    m = _PCT_RE.match(s)
    if m:
        try:
            return float(m.group(1)), "xsd:decimal"
        except ValueError:
            pass

    # Unit string: separate value and unit, keep only the value
    m = _UNIT_RE.match(s)
    if m:
        num_str = m.group(1)
        try:
            if "." in num_str:
                return float(num_str), "xsd:decimal"
            return int(num_str), "xsd:integer"
        except ValueError:
            pass

    # Plain integer (with optional thousands separators)
    clean = s.replace(",", "")
    if _INT_RE.match(s):
        try:
            return int(clean), "xsd:integer"
        except ValueError:
            pass

    # Plain decimal / float
    if _FLOAT_RE.match(clean):
        try:
            n = float(clean)
            return (int(n), "xsd:integer") if n == int(n) else (n, "xsd:decimal")
        except ValueError:
            pass

    # ISO date
    if _DATE_RE.match(s):
        return s, "xsd:date"

    # Fall back to string
    return s, "xsd:string"


# ── object mapper ─────────────────────────────────────────────────────────────

class ObjectMapper:
    """Routes object mapping through entity typing or literal parsing.

    Returns a dict with either:
      object_property path  → {"object_iri": ..., "object_type": ..., "confidence": ...}
      datatype_property path → {"object_literal": ..., "object_datatype": ..., "confidence": "high"}
    Or None if mapping fails (caller drops the triple).
    """

    def __init__(self, entity_typer: EntityTyper) -> None:
        self._typer = entity_typer

    def map(
        self,
        surface: str,
        sentence: str,
        predicate_kind: str,
    ) -> Optional[dict]:
        if predicate_kind == "datatype_property":
            return self._map_literal(surface)
        else:
            return self._map_entity(surface, sentence)

    def _map_entity(self, surface: str, sentence: str) -> Optional[dict]:
        result = self._typer.type_entity(surface, sentence)
        if result is None:
            return None
        ind_iri, class_iri, confidence = result
        return {
            "object_iri":    ind_iri,
            "object_type":   class_iri,
            "confidence":    confidence,
        }

    def _map_literal(self, surface: str) -> Optional[dict]:
        parsed = parse_literal(surface)
        if parsed is None:
            return None
        value, xsd_type = parsed
        return {
            "object_literal":  value,
            "object_datatype": xsd_type,
            "confidence":      "high",
        }
