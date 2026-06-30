---
name: stage2b-salience
description: Keeps the salient core of the named vocabulary and parks the long tail. Scores each concept's centrality to the domain from four grounded signals (match to the Stage 0 scope's author-marked terms, mention frequency, spread across the document, and hallucination-free grounding), then keeps concepts that are scope-matched or frequent (optionally capped to a top-N) and parks the rest as feedback. This is what turns an on-topic but long vocabulary into a focused ontology. Use after naming the vocabulary and before inducing structure, or when asked to "rank concepts by importance/salience", "keep the core concepts", "prune the long tail", or "shrink the ontology to its key terms". Reads a <stem>_vocabulary.json (and the Stage 0 scope) and writes <stem>_vocabulary_salient.json next to it.
---

# Rank by Salience

This skill performs the **salience** step of ontology extraction: keep the core,
park the tail. After gating removed particulars, the vocabulary is on-topic but
still long, and a domain's ontology is its *salient* concepts, not every common
noun an author used once. This stage scores centrality and splits the vocabulary
into a kept core and a parked tail.

The single job of this step: **rank concepts by how central they are to the
domain, and keep the central ones.** Nothing is deleted; parked concepts are
feedback for a later pass.

## When to use this skill

- You have a Stage 2 vocabulary that is still larger than the domain's real
  conceptualization, and a Stage 0 scope to anchor importance.
- The user asks to "rank by salience/importance", "keep the core concepts",
  "prune the long tail", or "shrink the ontology".

## Mechanism: four grounded signals

`scripts/salience.py` scores each concept in [0, 1] from signals that are already
in the pipeline's artifacts, so the score is explainable:

- **Scope match (strongest).** The concept is one the author marked important: it
  appears in the Stage 0 scope's `key_terms` (defined terms + general key
  concepts). A defined term is salient by construction. Procedural `topics`
  ("Divide Integers") are deliberately not used, so tasks are not kept as classes.
- **Frequency.** How often the concept is mentioned (`log1p(mention_count)`,
  normalized).
- **Spread.** Across how much of the document it appears (occurrences bucketed
  over the text). A concept used throughout is more central than one in a single
  passage.
- **Grounding.** Backed by a hallucination-free extractor, not LLM-only.

`score = 0.35·scope_match + 0.30·frequency + 0.20·spread + 0.15·grounding`

## Keep vs park

A concept is **kept** if it is scope-matched **or** frequent
(`mention_count >= --min-mentions`), and (when `--top-n` is set) within the top N
by score. Everything else is **parked** with a reason. Relations are kept when
`mention_count >= --relation-min-mentions`. The score and its components are
written onto every concept for transparency.

## How to run

```bash
cd scripts
python rank_salience.py <VOCABULARY_JSON> [options]
```

The result is written next to the vocabulary as `<VOCABULARY_JSON_stem>_salient.json`.
Its `concepts`/`relations` keys hold the kept core, so Stage 3 consumes it
directly; `parked` holds the long tail. The Stage 0 scope is auto-detected.

| Option | Default | Meaning |
|---|---|---|
| `--scope PATH` | auto-detect | Stage 0 scope JSON, for the scope-match signal. |
| `--min-mentions N` | `3` | Keep a concept seen at least N times. |
| `--relation-min-mentions N` | `2` | Keep a relation seen at least N times. |
| `--top-n N` | `0` (no cap) | Also cap kept concepts to the top N by salience score. |
| `--output PATH` | auto | Override the output path. |

```bash
# Defaults (keep scope-matched or seen >= 3 times)
python rank_salience.py INPUT_vocabulary.json

# Hard cap to the 150 most salient concepts
python rank_salience.py INPUT_vocabulary.json --top-n 150
```

## Output schema

`<stem>_vocabulary_salient.json`:

```jsonc
{
  "stage": "2b-salience",
  "vocabulary_file": "..._vocabulary.json", "scope_file": "..._scope.json",
  "input_file": "..._content.md",
  "generated_at": "ISO-8601 UTC",
  "config": { "min_mentions": 3, "top_n": 0, "relation_min_mentions": 2 },
  "stats": { "in_concepts": 0, "kept_concepts": 0, "parked_concepts": 0,
             "in_relations": 0, "kept_relations": 0, "parked_relations": 0,
             "scope_matched": 0 },
  "concepts": [ { "id": "...", "label": "Fraction", "salience": 0.71,
                  "salience_components": { "scope_match": 1.0, "freq": 0.6, "spread": 0.4, "grounded": 1.0 },
                  "...": "..." } ],
  "relations": [ { "...": "..." } ],
  "parked": [ { "id": "...", "label": "...", "salience": 0.12, "park_reason": "not in scope; mention_count < 3" } ]
}
```

## Files

```
stage2b-salience/
├── SKILL.md                 # this file
├── requirements.txt
├── scripts/
│   ├── rank_salience.py     # orchestrator + CLI (run this)
│   └── salience.py          # the four-signal salience score
└── references/
    └── method.md            # the salience method in detail
```

## Requirements

- Python 3.10+ only (pure standard library; no API keys, no models)

## Output and what comes next

The kept `concepts`/`relations` are the focused core that Stage 3 (synthesize
structure) wires into a taxonomy. The `parked` tail is feedback: a later pass over
more documents may promote a parked concept once it earns more mentions. Because
salience is just a ranking with simple keep rules, it is cheap to re-run at a
different `--min-mentions` or `--top-n` without touching the rest of the pipeline.

## Notes and limitations

- **Frequency is measured on the content text.** Because exercises were removed in
  Stage 0, a concept that is mostly *drilled* rather than *explained* can score
  low; the scope-match rule is what rescues such concepts when the author defined
  them. Parking is reversible, so a borderline concept is never lost.
- **Duplicates remain for Stage 4.** Exact-duplicate labels and singular/plural
  variants that survive here are merged downstream (Stage 4 merges duplicate
  labels; Stage 3 disambiguates ids).
- **Tune, do not trust, the thresholds.** `--min-mentions` and `--top-n` are
  dials for how large a core you want; the defaults aim at a focused ontology, not
  a fixed size.
