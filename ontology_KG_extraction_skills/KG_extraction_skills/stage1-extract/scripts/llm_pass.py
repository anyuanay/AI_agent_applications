"""LLM pass: batched entity/value surfacing + triple extraction via Gemini.

One Gemini call per batch of N sentences. Each call returns per-sentence:
  - entity and value mentions (feeds entity mention merge)
  - raw (subject, predicate, object) triples (-> _llm_triples.json)

Negated triples are included with negated: true so they can be tracked.
LLM failures on a batch are logged and skipped gracefully; the spaCy
floor remains intact.

Uses the google-genai SDK with gemini-3.1-flash-lite (thinking disabled
for speed, matching the existing ontology-extraction pipeline conventions).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from models import LLMTriple, RawMention, Sentence

DEFAULT_MODEL  = "gemini-3.1-flash-lite"
DEFAULT_BATCH_SIZE = 7
_MAX_RETRIES   = 2
_RETRY_DELAY   = 5.0


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
        if c and c.is_file():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val

    raise RuntimeError(
        "GOOGLE_API_KEY not found in environment or any .env file. "
        "Add GOOGLE_API_KEY=... to .env or pass --env /path/to/.env"
    )


def _response_text(resp) -> str:
    """Concatenate text parts, skipping thought_signature parts (gemini-3.x)."""
    try:
        parts = resp.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return ""
    return "".join(getattr(p, "text", "") or "" for p in parts)


def _build_prompt(batch: list[Sentence]) -> str:
    indices = [s.sentence_index for s in batch]
    lines = [f"{s.sentence_index}: {s.text}" for s in batch]
    sentences_block = "\n".join(lines)
    index_list = ", ".join(str(i) for i in indices)

    return (
        f"Below are {len(batch)} sentences from a source document, "
        f"indexed {index_list}.\n\n"
        "For EACH sentence, perform two tasks:\n\n"
        "1. Extract all entity and value mentions. Classify each as:\n"
        "   - entity: a named thing, a type, a category, a concept, an event\n"
        "   - value: a number, measurement, date, time, percentage, monetary "
        "amount, or string literal\n"
        '   Keep unit strings together with their number (e.g., "47 psi").\n'
        "   Do not include pronouns or generic stop words.\n\n"
        "2. Extract all (subject, predicate, object) triples readable from the "
        "sentence. Use only the entities and values from task 1. Be thorough — "
        "pull every relationship the sentence states. Do not filter by any schema. "
        "If a sentence negates a relationship, include the triple but set "
        'negated to true. Never silently omit negated triples.\n\n'
        "Return ONLY a JSON object with this exact structure (no markdown, "
        "no explanation):\n"
        '{"results": {"<sentence_index>": {"mentions": ['
        '{"surface": "<surface_form>", "mention_type": "entity|value"}], '
        '"triples": [{"subject": "<subj>", "predicate": "<pred>", '
        '"object": "<obj>", "negated": false}]}}}\n\n'
        f"Sentences:\n{sentences_block}"
    )


def _parse_response(
    text: str, batch: list[Sentence]
) -> tuple[list[RawMention], list[LLMTriple]]:
    """Parse the LLM JSON response into RawMentions and LLMTriples."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON in response: {cleaned[:200]}")
        data = json.loads(m.group(0))

    results = data.get("results", {})
    sent_index_set = {s.sentence_index for s in batch}

    raw_mentions: list[RawMention] = []
    llm_triples:  list[LLMTriple]  = []

    for key, val in results.items():
        try:
            sent_idx = int(key)
        except (ValueError, TypeError):
            continue
        if sent_idx not in sent_index_set:
            continue

        for mention in val.get("mentions", []) or []:
            surface = str(mention.get("surface", "")).strip()
            mtype   = str(mention.get("mention_type", "entity")).strip().lower()
            if not surface:
                continue
            if mtype not in ("entity", "value"):
                mtype = "entity"
            raw_mentions.append(RawMention(
                surface_form=surface,
                mention_type=mtype,
                source="llm",
                sentence_index=sent_idx,
            ))

        for triple in val.get("triples", []) or []:
            subj = str(triple.get("subject", "")).strip()
            pred = str(triple.get("predicate", "")).strip()
            obj  = str(triple.get("object", "")).strip()
            neg  = bool(triple.get("negated", False))
            if not subj or not pred or not obj:
                continue
            llm_triples.append(LLMTriple(
                subject=subj,
                predicate=pred,
                object=obj,
                sentence_index=sent_idx,
                negated=neg,
            ))

    return raw_mentions, llm_triples


class GeminiExtractor:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._model  = model

        cfg: dict = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
        }
        # Disable thinking for speed/cost (gemini-3.x models support this)
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

        self._config = types.GenerateContentConfig(**cfg)

    def extract_batch(
        self,
        batch: list[Sentence],
    ) -> tuple[list[RawMention], list[LLMTriple]]:
        prompt = _build_prompt(batch)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=self._config,
                )
                text = _response_text(resp)
                return _parse_response(text, batch)
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                else:
                    print(f"    [llm_pass] batch failed after "
                          f"{_MAX_RETRIES + 1} attempts: {exc}")
                    return [], []
        return [], []


def run(
    sentences: list[Sentence],
    api_key: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[RawMention], list[LLMTriple]]:
    """Run the LLM pass over all sentences in batches.

    Returns:
        raw_mentions — RawMention records (source="llm")
        llm_triples  — LLMTriple records
    """
    extractor = GeminiExtractor(api_key=api_key, model=model)
    all_mentions: list[RawMention] = []
    all_triples:  list[LLMTriple]  = []

    batches = [
        sentences[i : i + batch_size]
        for i in range(0, len(sentences), batch_size)
    ]
    total = len(batches)
    for idx, batch in enumerate(batches, 1):
        mentions, triples = extractor.extract_batch(batch)
        all_mentions.extend(mentions)
        all_triples.extend(triples)
        if idx % 10 == 0 or idx == total:
            print(
                f"    batch {idx}/{total}: "
                f"+{len(mentions)} mentions, +{len(triples)} triples  "
                f"[cumulative: {len(all_mentions)} mentions, "
                f"{len(all_triples)} triples]"
            )

    return all_mentions, all_triples
