"""LLM scoping: turn the document skeleton into a domain scope (optional).

Given the deterministic skeleton (topics, objectives, defined terms), an LLM
writes the parts that need synthesis rather than extraction:

  - a one-paragraph **domain statement** (what this ontology is about),
  - a cleaned **topic** list and a list of **key concepts** (general kinds, not
    worked-example instances),
  - **competency questions** the ontology should be able to answer, which bound
    its scope and fix its abstraction level, and
  - an **out-of-scope** note naming the kinds of things that are instances/facts,
    not concepts (this is the type/instance boundary the later stages enforce).

The LLM only ever sees the grounded skeleton, never invents a domain, and is asked
for general concepts, so it raises abstraction without fabricating content. Model:
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
    raise RuntimeError("GOOGLE_API_KEY not found; add it to .env or run with --no-llm.")


_SYSTEM = (
    "You are scoping an ontology for a domain from a document's structural "
    "skeleton (its topics, learning objectives, and defined terms). An ontology "
    "captures the GENERAL concepts and relationships of a domain, NOT individual "
    "facts or worked-example values. Produce a tight, abstract scope.\n\n"
    "Return ONLY a JSON object with keys:\n"
    '  "domain_statement": one or two sentences naming the domain and its scope.\n'
    '  "topics": cleaned list of the main topics.\n'
    '  "key_concepts": general domain concepts (kinds of things), short noun '
    "phrases, NOT specific instances. Prefer singular canonical forms.\n"
    '  "relations": important general relationships between concepts, as short '
    "verb phrases (e.g. 'is divisible by', 'is a factor of').\n"
    '  "competency_questions": 6 to 10 questions the ontology should answer.\n'
    '  "out_of_scope": kinds of things in the text that are instances/examples/'
    "facts, not concepts (e.g. specific numbers, specific worked examples).\n"
)


class LLMScope:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        cfg = dict(system_instruction=_SYSTEM, temperature=0.0,
                   response_mime_type="application/json")
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:  # noqa: BLE001
            pass
        self._config = types.GenerateContentConfig(**cfg)

    def summarize(self, skeleton: str) -> dict:
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=skeleton, config=self._config)
            parts = resp.candidates[0].content.parts or []
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception as exc:  # noqa: BLE001
            print(f"  [llm] scope call failed: {str(exc)[:120]}")
            return {}
        m = _JSON.search(text)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
        # Coerce to the expected shapes, tolerating omissions.
        def as_list(v):
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return []
        return {
            "domain_statement": str(data.get("domain_statement", "")).strip(),
            "topics": as_list(data.get("topics")),
            "key_concepts": as_list(data.get("key_concepts")),
            "relations": as_list(data.get("relations")),
            "competency_questions": as_list(data.get("competency_questions")),
            "out_of_scope": as_list(data.get("out_of_scope")),
        }
