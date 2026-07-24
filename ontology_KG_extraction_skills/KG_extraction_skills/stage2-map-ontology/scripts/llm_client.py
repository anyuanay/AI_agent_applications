"""Gemini LLM client for Stage 2 mapping fallbacks.

Two thin call wrappers:
  classify_entity  — map a surface mention to an ontology class IRI
  map_predicate    — map a surface predicate to an ontology property IRI
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "gemini-3.1-flash-lite"
_MAX_RETRIES  = 2
_RETRY_DELAY  = 4.0
_JSON_BLOCK   = re.compile(r"\{.*?\}", re.DOTALL)


def load_api_key(env_path: Optional[Path] = None) -> str:
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / ".env")

    for c in candidates:
        if c.is_file():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val

    raise RuntimeError(
        "GOOGLE_API_KEY not found in environment or any .env file."
    )


def _response_text(resp) -> str:
    try:
        parts = resp.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return ""
    return "".join(getattr(p, "text", "") or "" for p in parts)


def _parse_json(text: str) -> Optional[dict]:
    """Extract and parse the first JSON object from text."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._model  = model
        cfg: dict = {"temperature": 0.0, "response_mime_type": "application/json"}
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        self._config = types.GenerateContentConfig(**cfg)
        self.calls   = 0

    def _call(self, prompt: str) -> Optional[dict]:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model, contents=prompt, config=self._config
                )
                self.calls += 1
                return _parse_json(_response_text(resp))
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                else:
                    print(f"    [llm] call failed: {exc}")
                    return None
        return None

    def classify_entity(
        self,
        surface: str,
        sentence: str,
        classes: list[dict],
    ) -> Optional[str]:
        """Return the best-matching class IRI or None.

        classes: list of {"iri": ..., "label": ..., "comment": ...}
        """
        cls_lines = "\n".join(
            f"  {c['iri']}: {c['label']}"
            + (f" — {c['comment'][:120]}" if c.get("comment") else "")
            for c in classes
        )
        prompt = (
            f"You are an ontology instance classifier.\n\n"
            f"Surface mention: \"{surface}\"\n"
            f"Context sentence: \"{sentence}\"\n\n"
            f"Ontology classes:\n{cls_lines}\n\n"
            "Which class does this mention belong to as an instance?\n"
            'Return ONLY JSON: {"class_iri": "<iri>"} or {"class_iri": null} '
            "if no class fits."
        )
        result = self._call(prompt)
        if not result:
            return None
        iri = result.get("class_iri")
        return str(iri).strip() if iri else None

    def map_predicate(
        self,
        surface: str,
        sentence: str,
        properties: list[dict],
    ) -> Optional[tuple[str, str]]:
        """Return (property_iri, direction) or None.

        direction is "direct" or "inverse".
        properties: list of {"iri", "label", "kind", "domain_label", "range_label",
                              "inverse_of", "inverse_labels"}
        """
        prop_lines: list[str] = []
        for p in properties:
            parts = [f"  {p['iri']}: \"{p['label']}\" [{p['kind']}]"]
            if p.get("domain_label") or p.get("range_label"):
                parts[0] += f" dom={p.get('domain_label','')} rng={p.get('range_label','')}"
            if p.get("inverse_labels"):
                inv = ", ".join(f'"{lbl}"' for lbl in p["inverse_labels"])
                parts[0] += f" | inverse labels: {inv}"
            prop_lines.append(parts[0])
        props_block = "\n".join(prop_lines)

        prompt = (
            f"You are an ontology predicate mapper.\n\n"
            f"Surface predicate: \"{surface}\"\n"
            f"Context sentence: \"{sentence}\"\n\n"
            f"Ontology properties:\n{props_block}\n\n"
            "Which property does the surface predicate map to?\n"
            "If the surface predicate matches the property in its forward direction, set direction=direct.\n"
            "If it matches an inverse label (the property expressed in reverse), set direction=inverse.\n"
            'Return ONLY JSON: {"property_iri": "<iri>", "direction": "direct|inverse"} '
            'or {"property_iri": null} if no property fits.'
        )
        result = self._call(prompt)
        if not result:
            return None
        iri = result.get("property_iri")
        if not iri:
            return None
        direction = str(result.get("direction", "direct")).strip().lower()
        if direction not in ("direct", "inverse"):
            direction = "direct"
        return str(iri).strip(), direction
