"""Optional LLM type/instance tiebreak.

After the deterministic gate removes the digit/LaTeX particulars, what remains are
clean common-noun phrases, most of which are real classes but some of which are
named individuals with no numeric tell (a person ``Marissa``, a day ``Tuesday``, a
place, a brand). spaCy NER is too noisy to separate these on this corpus, but an
LLM, told the domain, does it reliably.

Given the domain statement and the out-of-scope note from Stage 0, the model sorts
a batch of terms into ``class`` (a general kind in this domain), ``individual`` (a
specific named thing/value), or ``non_concept`` (not a meaningful term). It is a
tiebreak over the survivors, not the primary filter. Model:
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
    raise RuntimeError("GOOGLE_API_KEY not found; add it to .env or drop --llm.")


class LLMClassifier:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 domain: str = "", out_of_scope: Optional[list] = None):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        oos = "; ".join(out_of_scope or [])
        system = (
            "You sort candidate terms for an ontology of this domain:\n"
            f"  DOMAIN: {domain or 'unspecified'}\n"
            f"  OUT OF SCOPE (these are instances/examples, NOT concepts): {oos}\n\n"
            "An ontology holds GENERAL concepts (kinds/types), not specific "
            "individuals or example values. Classify each term as exactly one of:\n"
            '  "class": a general kind in this domain (e.g. fraction, equation).\n'
            '  "individual": a specific named thing or value (a person, a date, a '
            "place, a specific number or worked example).\n"
            '  "non_concept": not a meaningful domain term (fragment, debris).\n'
            'Respond ONLY as JSON mapping each term to its label: {"term": "class"}.'
        )
        cfg = dict(system_instruction=system, temperature=0.0,
                   response_mime_type="application/json")
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:  # noqa: BLE001
            pass
        self._config = types.GenerateContentConfig(**cfg)

    def _ask(self, prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self._config)
            parts = resp.candidates[0].content.parts or []
            return "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception as exc:  # noqa: BLE001
            print(f"    [llm] classify call failed: {str(exc)[:100]}")
            return ""

    def classify(self, terms: list[str], batch: int = 60) -> dict[str, str]:
        """Return {term: label} for the given terms; missing terms default to class."""
        out: dict[str, str] = {}
        for i in range(0, len(terms), batch):
            chunk = terms[i:i + batch]
            prompt = "Terms:\n" + "\n".join(f"- {t}" for t in chunk)
            m = _JSON.search(self._ask(prompt))
            if not m:
                continue
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            for k, v in data.items():
                if isinstance(v, str):
                    lab = v.strip().lower()
                    if lab in ("class", "individual", "non_concept"):
                        out[str(k)] = lab
        return out
