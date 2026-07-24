"""Optional LLM refinement (Refine verb): name the coined parents.

This is where the pipeline's deferred naming lands. Stage 2 named extracted
concepts deterministically and Stage 3 left coined parents with placeholder names
(``CoinedFamily7``, flagged ``needs_naming``). Here, under review, an LLM proposes
a real name for each accepted coined family from its children. Naming is a Refine
edit: human-approvable, recorded, never a silent rewrite.

Model: ``gemini-3.1-flash-lite`` via ``google-genai`` (same as Stage 1), key from
``GOOGLE_API_KEY``. Off unless ``--llm`` is passed; without it, coined parents keep
their placeholder names and stay flagged for a human to name.
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


class LLMRefiner:
    """Thin Gemini wrapper that names coined families in batches."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        cfg = dict(temperature=0.0, response_mime_type="application/json")
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
            print(f"    [llm] call failed: {str(exc)[:100]}")
            return ""

    def name_families(self, families: list[list[str]], batch: int = 20) -> list[str]:
        """Return one common-parent name per family (PascalCase-ish noun phrase)."""
        out: list[str] = [""] * len(families)
        for start in range(0, len(families), batch):
            chunk = families[start:start + batch]
            listing = "\n".join(
                f"{i}: {', '.join(fam[:8])}" for i, fam in enumerate(chunk))
            prompt = (
                "Each numbered line is a set of sibling subclasses in an ontology. "
                "For each, give a single short common-parent category name (a noun "
                "phrase, no explanation). Respond ONLY as JSON mapping the number "
                'to the name, e.g. {"0": "ResponderUnit"}.\n\n' + listing
            )
            m = _JSON.search(self._ask(prompt))
            if not m:
                continue
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            for k, v in data.items():
                try:
                    idx = int(k)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(chunk) and isinstance(v, str) and v.strip():
                    out[start + idx] = v.strip()
        return out
