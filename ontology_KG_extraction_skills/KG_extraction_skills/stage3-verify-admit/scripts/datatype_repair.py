"""Datatype compatibility check and repair for datatype-property triples.

check_and_repair() verifies that an object literal's XSD type is compatible
with the property's declared range. If incompatible it attempts a cast repair.
Returns (ok, value, xsd_type, verdict) where verdict is:
  "pass"    — types already compatible or no range declared
  "repaired"— cast succeeded; value and xsd_type reflect the corrected form
  "reject"  — incompatible and cast failed
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

_XSD_INT      = "xsd:integer"
_XSD_DECIMAL  = "xsd:decimal"
_XSD_FLOAT    = "xsd:float"
_XSD_DOUBLE   = "xsd:double"
_XSD_BOOL     = "xsd:boolean"
_XSD_DATE     = "xsd:date"
_XSD_STRING   = "xsd:string"
_XSD_NON_NEG  = "xsd:nonNegativeInteger"
_NUMERIC      = {_XSD_INT, _XSD_DECIMAL, _XSD_FLOAT, _XSD_DOUBLE, _XSD_NON_NEG}


def _to_short_xsd(range_iri: Optional[str]) -> Optional[str]:
    """Normalise a range IRI to short xsd: form, or None if not an XSD type."""
    if not range_iri:
        return None
    if range_iri.startswith("xsd:"):
        return range_iri
    if "XMLSchema#" in range_iri:
        return "xsd:" + range_iri.split("XMLSchema#")[-1]
    return None


def _compatible(actual: str, target: str) -> bool:
    if actual == target:
        return True
    # Integer widens to decimal/float/double
    if actual == _XSD_INT and target in (_XSD_DECIMAL, _XSD_FLOAT, _XSD_DOUBLE):
        return True
    # nonNegativeInteger is a subtype of integer and numeric types
    if actual == _XSD_NON_NEG and target in _NUMERIC:
        return True
    # xsd:string accepts anything as a fallback when target is string
    if target == _XSD_STRING:
        return True
    return False


def _cast(text: str, target: str) -> Optional[tuple[object, str]]:
    """Try to coerce text to target XSD type. Returns (value, xsd_type) or None."""
    t = text.strip()
    if target in (_XSD_INT, _XSD_NON_NEG):
        m = re.match(r"^-?\d+", t)
        if m:
            v = int(m.group())
            if target == _XSD_NON_NEG and v < 0:
                return None
            return v, target
        return None
    if target in (_XSD_DECIMAL, _XSD_FLOAT, _XSD_DOUBLE):
        m = re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?", t)
        if m and m.group():
            return float(m.group()), target
        return None
    if target == _XSD_BOOL:
        if t.lower() in ("true", "1", "yes"):
            return True, _XSD_BOOL
        if t.lower() in ("false", "0", "no"):
            return False, _XSD_BOOL
        return None
    if target == _XSD_DATE:
        try:
            d = date.fromisoformat(t[:10])
            return d.isoformat(), _XSD_DATE
        except ValueError:
            return None
    if target == _XSD_STRING:
        return t, _XSD_STRING
    return None


def check_and_repair(
    literal: object,
    actual_xsd: Optional[str],
    declared_range: Optional[str],
) -> tuple[bool, object, str, str]:
    """
    Check and optionally repair a datatype literal.

    Returns (ok, value, xsd_type, verdict).
    """
    target = _to_short_xsd(declared_range)
    if target is None or actual_xsd is None:
        return True, literal, actual_xsd or _XSD_STRING, "pass"

    if _compatible(actual_xsd, target):
        return True, literal, actual_xsd, "pass"

    result = _cast(str(literal), target)
    if result is not None:
        val, xsd = result
        return True, val, xsd, "repaired"

    return False, literal, actual_xsd, "reject"
