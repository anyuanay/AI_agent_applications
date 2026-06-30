"""LLM ontology synthesis (Gemini), fenced by the candidate vocabulary.

A purely bottom-up Stage 3 would build the taxonomy from lexical heads,
Hearst patterns, and clustering. This alternative lets an LLM *synthesize* a
parsimonious ontology instead, which gives a cleaner hierarchy and, crucially,
real domain/range on relations (the bottom-up co-occurrence heuristic's weak
point). The danger of LLM synthesis is hallucination, so it is fenced on both
ends:

  - it is given ONLY the grounded salient candidates and the Stage 0 scope, and
  - it is told to use the candidate labels as classes, omit redundant ones, and
    coin only a few clearly-needed abstract parents (which it must flag).

Anything it returns that is not a candidate (and not flagged coined) is treated as
an ungrounded introduction and flagged downstream, so grounding survives.

Two calls keep each response bounded and focused: one for the taxonomy, one for
the relations (domain/range over the synthesized classes). Model:
``gemini-3.1-flash-lite`` (same as Stage 1), key from ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "gemini-3.1-flash-lite"
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def load_api_key(env_path: Optional[Path] = None) -> str:
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
    raise RuntimeError("GOOGLE_API_KEY not found; add it to a .env file.")


_TAXO_SYSTEM = (
    "You are an ontology engineer building an is-a taxonomy for a domain from a "
    "list of candidate concepts. Aim for good COVERAGE of the domain's concepts, "
    "organized into a clean hierarchy.\n"
    "RULES:\n"
    "  - KEEP every distinct domain concept from the candidates (e.g. Denominator, "
    "Numerator, SquareRoot, AbsoluteValue, PrimeFactorization, LeastCommonMultiple, "
    "Addition, Exponent, Base, Zero). Place each under its most specific correct "
    "parent. Do NOT over-prune.\n"
    "  - OMIT only: exact duplicates and near-synonyms (keep one), procedural or "
    "verb-like phrases (Solve, Simplify, Translate, Evaluate), and one-off fragments.\n"
    "  - You MAY introduce abstract parent classes that are NOT in the list when a "
    "group of candidates shares an unnamed parent (e.g. ArithmeticOperation over "
    "Addition/Subtraction); mark each such class coined=true.\n"
    "  - Give each class exactly one immediate parent: the label of a broader class "
    "(a candidate or a coined parent), or null if it is top-level. Prefer an "
    "existing candidate as parent over coining.\n"
    'Respond ONLY as JSON: {"classes": [{"label": "...", "parent": "..."|null, '
    '"coined": false}]}.'
)

_REL_SYSTEM = (
    "You define the key relations (object properties) of an ontology. Produce a "
    "COMPREHENSIVE set of meaningful, general relations among the given classes.\n"
    "RULES:\n"
    "  - Draw on the candidate relations, the suggested relations, and the "
    "competency questions; add obvious domain relations they imply.\n"
    "  - Each relation has a clean camelCase label, plus a domain and range that "
    "are EACH a class label from the provided ontology classes.\n"
    "  - Aim for breadth: cover the important relationships a user would ask about "
    "(e.g. isDivisibleBy, isAFactorOf, isAMultipleOf, hasNumerator, isASolutionOf, "
    "simplifiesTo). Omit only vague or purely procedural predicates.\n"
    'Respond ONLY as JSON: {"relations": [{"label": "...", "domain": "...", '
    '"range": "..."}]}.'
)


class LLMSynthesizer:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _config(self, system: str):
        cfg = dict(system_instruction=system, temperature=0.0,
                   response_mime_type="application/json")
        try:
            cfg["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
        except Exception:  # noqa: BLE001
            pass
        return self._types.GenerateContentConfig(**cfg)

    def _ask(self, system: str, prompt: str) -> dict:
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self._config(system))
            parts = resp.candidates[0].content.parts or []
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception as exc:  # noqa: BLE001
            print(f"    [llm] call failed: {str(exc)[:120]}")
            return {}
        m = _JSON.search(text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}

    def taxonomy(self, domain: str, competency: list[str],
                 concepts: list[tuple[str, int]]) -> list[dict]:
        """Return [{label, parent, coined}] for the synthesized taxonomy."""
        clist = "\n".join(f"- {lab} (x{mc})" for lab, mc in concepts)
        cq = "\n".join(f"- {q}" for q in (competency or [])[:10])
        prompt = (f"DOMAIN: {domain}\n\nCOMPETENCY QUESTIONS:\n{cq}\n\n"
                  f"CANDIDATE CONCEPTS (label x mention_count):\n{clist}")
        data = self._ask(_TAXO_SYSTEM, prompt)
        out = []
        for c in (data.get("classes") or []):
            if isinstance(c, dict) and c.get("label"):
                out.append({"label": str(c["label"]).strip(),
                            "parent": (str(c["parent"]).strip()
                                       if c.get("parent") else None),
                            "coined": bool(c.get("coined"))})
        return out

    def relations(self, class_labels: list[str], rel_labels: list[str],
                  domain: str = "", competency: Optional[list] = None,
                  suggested: Optional[list] = None) -> list[dict]:
        """Return [{label, domain, range}] with domain/range as class labels."""
        cq = "\n".join(f"- {q}" for q in (competency or [])[:10])
        sug = ", ".join(suggested or [])
        prompt = (f"DOMAIN: {domain}\n\nCOMPETENCY QUESTIONS:\n{cq}\n\n"
                  f"SUGGESTED RELATIONS: {sug}\n\n"
                  "ONTOLOGY CLASSES:\n" + "\n".join(f"- {c}" for c in class_labels)
                  + "\n\nCANDIDATE RELATIONS:\n"
                  + "\n".join(f"- {r}" for r in rel_labels))
        data = self._ask(_REL_SYSTEM, prompt)
        out = []
        for r in (data.get("relations") or []):
            if isinstance(r, dict) and r.get("label") and r.get("domain") and r.get("range"):
                out.append({"label": str(r["label"]).strip(),
                            "domain": str(r["domain"]).strip(),
                            "range": str(r["range"]).strip()})
        return out
