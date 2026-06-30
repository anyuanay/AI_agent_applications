"""LLM extractor: the complementary lift over the spaCy signals.

The spaCy extractor cannot hallucinate but misses the implicit; the LLM catches
the multi-word, implicit, and relational candidates the spaCy methods miss (it
reads "the value left over after division" and proposes ``Remainder`` even though
no clean noun phrase names it).

Two rules are enforced here:

  1. **Terms only.** The model is prompted for a flat list of terms. Any
     structure it volunteers (parents, hierarchy) is discarded: structure is a
     later step's job, and the model is least trustworthy exactly when it
     guesses hierarchy.

  2. **Tag the lift.** Whatever the LLM proposes is tagged ``llm``. The
     orchestrator then checks each term against the source text: terms found
     verbatim get literal-span provenance; terms that are *not* in the text are
     kept (recall-first) but stay ``llm_only`` and non-literal, so a later review
     can give them a harder look.

Model: ``gemini-3.1-flash-lite`` via the ``google-genai`` SDK. The API key is
read from a ``.env`` file (``GOOGLE_API_KEY``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from candidate import RawCandidate, SOURCE_LLM, KIND_CONCEPT, KIND_RELATION
from textnorm import clean_surface


DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Prompt is strict: terms only, two flat lists, no definitions, no hierarchy.
_SYSTEM = (
    "You are an ontology term spotter. Your only job is to maximize recall of "
    "terms that could become ontology elements. You do NOT build structure.\n\n"
    "From the passage, list:\n"
    "  - concepts: domain noun concepts (things, types, roles, artifacts), "
    "including multi-word and IMPLICIT concepts that are described but not named "
    "by a clean noun phrase.\n"
    "  - relations: relationships between concepts, given as a verb or short "
    "verb phrase (e.g. 'add', 'is divisible by', 'simplifies to', 'consists "
    "of'), including copular and prepositional predicates.\n\n"
    "STRICT RULES:\n"
    "  - Terms only. Do NOT output parents, sub/super-classes, hierarchy, "
    "definitions, or sentences.\n"
    "  - Prefer the canonical noun form; keep it short.\n"
    "  - It is fine to include a concept that is implied but not literally "
    "written; that is the point of this stage.\n"
    "  - Do not invent domain-foreign terms; stay grounded in the passage's "
    "subject matter.\n"
    'Respond with ONLY a JSON object: {"concepts": [...], "relations": [...]}.'
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def load_api_key(env_path: Optional[Path] = None) -> str:
    """Read GOOGLE_API_KEY from the environment or a .env file.

    Search order: explicit ``env_path`` -> ``GOOGLE_API_KEY`` already in the
    environment -> the nearest ``.env`` walking up from this file. We avoid a
    hard dependency on python-dotenv's auto-discovery so the script behaves the
    same no matter where it is launched from.
    """
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]

    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / ".env")

    for c in candidates:
        if c and c.is_file():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    raise RuntimeError(
        "GOOGLE_API_KEY not found in environment or any .env file. "
        "Add GOOGLE_API_KEY=... to the skill's .env."
    )


def chunk_text(text: str, chunk_chars: int) -> list[str]:
    """Split text into ~chunk_chars pieces on paragraph boundaries."""
    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if size + len(p) > chunk_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def sample_chunks(chunks: list[str], max_chunks: int) -> list[str]:
    """Evenly sample at most ``max_chunks`` chunks across the document.

    Even sampling (rather than just the first N) keeps recall representative of
    the whole document when a cap is set for cost/time during testing.
    """
    if max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks
    step = len(chunks) / max_chunks
    return [chunks[int(i * step)] for i in range(max_chunks)]


def _response_text(resp) -> str:
    """Concatenate the text parts of a Gemini response.

    We read ``parts`` directly rather than the ``resp.text`` shortcut: thinking
    models (gemini-3.x) attach a non-text ``thought_signature`` part, and the
    shortcut prints a warning every call. We just want the text parts.
    """
    try:
        parts = resp.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return ""
    return "".join(getattr(p, "text", "") or "" for p in parts)


class LLMTerms(BaseModel):
    """Validated shape of the model's JSON reply: two flat lists of term strings.

    This is the one untrusted boundary in the extractor. The reply may contain
    extra keys, ``null`` lists, scalars instead of lists, or non-string items;
    the validators below coerce each entry to a clean term string and drop
    anything malformed, so callers always get ``list[str]``.
    """

    model_config = ConfigDict(extra="ignore")

    concepts: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)

    @field_validator("concepts", "relations", mode="before")
    @classmethod
    def _as_str_list(cls, value):
        # Tolerate null, a bare scalar, or a non-list: normalize to a list of
        # stripped, non-empty strings; skip nested objects/arrays.
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
            elif isinstance(item, (int, float, bool)):
                s = str(item).strip()
            else:
                continue  # skip dicts / nested lists the model should not emit
            if s:
                out.append(s)
        return out


def _parse_terms(raw: str) -> tuple[list[str], list[str]]:
    """Validate the model's JSON reply into (concepts, relations).

    The reply text may be wrapped in prose, so we first extract the JSON object,
    then validate it with :class:`LLMTerms`. On any failure (no JSON, invalid
    JSON, wrong shape) we return empty lists, consistent with recall-first: a
    missing LLM lift is acceptable because the spaCy floor still stands.
    """
    m = _JSON_BLOCK.search(raw or "")
    if not m:
        return [], []
    try:
        terms = LLMTerms.model_validate_json(m.group(0))
    except ValidationError:
        return [], []
    return terms.concepts, terms.relations


class LLMExtractor:
    """Thin wrapper over the Gemini client for terms-only extraction."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        cfg = dict(
            system_instruction=_SYSTEM,
            temperature=0.0,
            response_mime_type="application/json",
        )
        # Stage 1 is term-spotting, not reasoning: disable thinking where the
        # model supports it. Keeps it fast and inexpensive and silences the SDK's
        # "non-text parts (thought_signature)" warning. Tolerate older SDKs.
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:  # noqa: BLE001
            pass
        self._config = types.GenerateContentConfig(**cfg)

    def extract_chunk(self, chunk: str) -> list[RawCandidate]:
        """Return terms-only RawCandidates for one chunk (no offsets yet).

        Offsets are resolved later by the orchestrator against the original file,
        which is what decides literal-span vs. implicit. A network/parse failure
        degrades to an empty list rather than aborting the run (recall-first: a
        missing LLM lift is acceptable; the spaCy floor still stands).
        """
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=chunk,
                config=self._config,
            )
            text = _response_text(resp)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"  [llm] chunk failed, skipping: {exc}")
            return []

        concepts, relations = _parse_terms(text)
        out: list[RawCandidate] = []
        for c in concepts:
            s = clean_surface(c)
            if s:
                out.append(RawCandidate(text=s, kind=KIND_CONCEPT,
                                        source=SOURCE_LLM, key=""))
        for r in relations:
            s = clean_surface(r)
            if s:
                out.append(RawCandidate(text=s, kind=KIND_RELATION,
                                        source=SOURCE_LLM, key=""))
        return out
