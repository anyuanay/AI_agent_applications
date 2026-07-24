"""
Skills — progressive disclosure (Part 8).

A skill is a packaged, multi-step capability. Only its `description` is always in
context; the full procedure (SKILL.md body) loads on demand when the skill is
entered. This loader reads `skills/<name>/SKILL.md`, parses its frontmatter, and
makes the procedure available to whatever runs inside the `with skill(...)` block.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

# The procedure text of the currently-loaded skill, if any (progressive disclosure).
_ACTIVE: list[str] = []


def _parse_skill(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into (frontmatter dict, body)."""
    meta: dict = {}
    body = text
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        for line in fm.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
    return meta, body.strip()


def load_skill(name: str) -> tuple[dict, str]:
    path = SKILLS_DIR / name / "SKILL.md"
    return _parse_skill(path.read_text(encoding="utf-8"))


@contextmanager
def skill(name: str):
    """Load a skill's procedure into context only for the duration of the block."""
    meta, body = load_skill(name)
    _ACTIVE.append(body)
    try:
        yield body
    finally:
        _ACTIVE.pop()


def active_skill() -> str | None:
    return _ACTIVE[-1] if _ACTIVE else None
