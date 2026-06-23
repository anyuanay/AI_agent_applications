"""Dataclasses shared across the extraction skills.

These mirror the structures in ``scima.ontology_learning`` so a skill can be
read and run on its own. When the backing package is importable we re-export its
versions to guarantee the two never diverge; otherwise we fall back to local
definitions with the same fields.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


def backing_repo_root() -> Path:
    """Absolute path to ``ontology_kg_for_agents`` (the backing package repo)."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "ontology_kg_for_agents"


def ensure_scima_importable() -> None:
    """Put the backing package on ``sys.path`` if it is present on disk."""
    backing = backing_repo_root()
    if backing.is_dir() and str(backing) not in sys.path:
        sys.path.insert(0, str(backing))


# Prefer the canonical definitions from the backing package.
ensure_scima_importable()
try:
    from scima.ontology_learning import ProposedClass, RiteResult  # type: ignore
except Exception:  # pragma: no cover - fallback when backing repo is absent
    @dataclass(frozen=True)
    class ProposedClass:  # type: ignore[no-redef]
        """A candidate OWL class as it moves through extraction and review."""

        iri: str
        parent: str
        label: str
        confidence: float = 0.0
        source: str = ""

        @property
        def local_name(self) -> str:
            return self.iri.split(":", 1)[-1]

    @dataclass
    class RiteResult:  # type: ignore[no-redef]
        """Outcome of one Refine-Inspect-Test-Extend pass."""

        accepted: list = field(default_factory=list)
        rejected: list = field(default_factory=list)

        def accepted_iris(self) -> set:
            return {c.iri for c in self.accepted}


__all__ = [
    "ProposedClass",
    "RiteResult",
    "backing_repo_root",
    "ensure_scima_importable",
]
