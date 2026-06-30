---
name: stage2-name-vocabulary
description: Turns the Stage 1 candidate bag into a clean, flat vocabulary of named ontology elements. Groups synonymous mentions by meaning (context-enriched OpenAI embeddings, lexically-guarded clustering, with an offline lexical fallback), drops only the obvious non-concepts, and gives each survivor one canonical label (PascalCase concept / camelCase relation) while keeping every surface as an alternate label and pooling provenance. Use as the second step of ontology extraction, after surfacing candidates, or when asked to "name the vocabulary", "group synonymous terms", or "canonicalize candidate concepts and relations". Reads a <stem>_candidates.json plus its source text and writes <stem>_candidates_vocabulary.json next to it.
---

# Name the Vocabulary

This skill performs the **naming** step of ontology extraction: take the noisy
*bag of candidate mentions* from Stage 1 and turn it into a clean, flat
**vocabulary of named elements**. This is the *term to concept* transition: many
surface strings collapse into one named thing.

The single job of this step: **decide identity.** Which mentions are the same
thing, what each thing is called, and which mentions were never concepts at all.
The output is a flat set of concept names and a flat set of relation names, each
carrying its cluster of lexicalizations and its pooled provenance. There are
still **no edges, no parents, no domain/range**: structure is Stage 3's job.

## When to use this skill

- You have run Stage 1 (surface candidates) and have a
  `<stem>_candidates.json`, plus the original source text it came from.
- You want synonymous mentions merged into one named element, each with a clean
  canonical label and its alternate labels preserved.
- The user asks to "name the vocabulary", "group / merge synonymous terms",
  "canonicalize candidates", or "produce concept and relation names".

Do **not** use this skill to assign parents, build a taxonomy, or set
domain/range. Those belong to Stage 3 (synthesize structure). Do not use it as a hard
precision gate either: pruning to precision is Stage 4 (review). This step keeps
almost everything plausible on purpose.

## Three design principles (carried from Stage 1)

- **Recall-first.** Selection here is minimal: only the obvious non-concepts are
  dropped, and even those are not deleted. A rejected element is written to a
  `dropped` list with its reason, as feedback Stage 4 can overrule, never a silent
  deletion.
- **Provenance everywhere.** Every named element pools the provenance of its
  members: the extractor `sources`, the `ner_labels`, and the literal
  `occurrences` (offsets into the original file). `mention_count` keeps the true
  total even when the stored occurrence list is capped.
- **Grounding by kind.** The `concept` / `relation` kind from Stage 1 is carried
  straight through and never crossed: concepts group only with concepts,
  relations only with relations, and casing follows the kind.

## Mechanism: group, then select, then name

### 1. Group synonymous mentions by meaning

`scripts/group.py`. Stage 1 already did **string-level** dedup; this is
**meaning-level** dedup, a different and harder operation. `IC`,
`incident commander`, and `the commander` are one concept even though only two of
them share characters.

- **Semantic (default).** Each candidate is embedded *together with its
  surrounding context* with **OpenAI `text-embedding-3-small`**
  (`scripts/embed.py`, 1536 dims by default). Context is what lets an
  abbreviation meet its expansion: the sentences around `IC` and
  `incident commander` look alike even when the strings do not. Two candidates
  are linked when their embeddings are within `--group-threshold` cosine distance
  **and** they pass a **lexical guard** (they share a content token, or one is the
  acronym of the other). Connected components of that graph are the groups, so no
  target cluster count is needed. The guard is what keeps embeddings honest in a
  homogeneous domain: without it, co-hyponyms like `numerator` and `denominator`
  merge because their contexts coincide; with it, only genuine synonyms
  (`real number` / `Real Numbers`) and abbreviations (`LCD` /
  `least common denominator`) do.
- **Lexical fallback (`--no-embeddings`).** No semantic signal, so each candidate
  stays its own group except where one is an **acronym whose letters are the
  initials** of another (`IC` with `incident commander`, `LCM` with
  `least common multiple`). That one rule is high-precision and needs no
  embeddings; general synonymy is left to the semantic path.

### 2. Select, minimally

`scripts/selection.py`. Drop only the obvious non-concepts: symbol/number-only
surfaces, document boilerplate (`Page`, `Step`, `Figure`, `Table`, ...), sentence
fragments (an over-long multi-word span the chunker over-captured), and the
narrow case of one-off ungrounded LLM noise that appears nowhere in the text.
Everything else survives. This is **not** the precision gate. Nothing is deleted:
each rejection goes to the `dropped` list with a `drop_reason`.

### 3. Name each survivor

`scripts/name.py`. Each group gets one **canonical label** in ontology casing:
`PascalCase` for a concept (`IncidentCommander`, `LeastCommonMultiple`),
`camelCase` for a relation (`isDivisibleBy`, `dispatchedTo`). The representative
surface the label is coined from is chosen to prefer a grounded, spelled-out,
frequently-seen, longer form, so an expansion (`incident commander`) wins over its
acronym (`IC`). Every surface in the group is kept as an `alt_label`, so the link
back to the text is never lost.

## How to run

```bash
cd scripts
python name_vocabulary.py <CANDIDATES_JSON> [options]
```

The result is written next to the candidates file as
`<CANDIDATES_JSON_stem>_vocabulary.json` (that is, `<INPUT>_candidates_vocabulary.json`).

The source text is read from the candidates file's `input_file` field by default;
pass `--source` to override (needed when the file has moved).

| Option | Default | Meaning |
|---|---|---|
| `--source PATH` | candidates' `input_file` | Original text, used for context-aware embeddings. |
| `--no-embeddings` | off | Lexical grouping only (acronym/initials). No API key, deterministic. |
| `--embed-model` | `text-embedding-3-small` | OpenAI embedding model id. |
| `--embed-dim` | `1536` | Embedding dimensionality (`text-embedding-3-*` can shorten below 1536). |
| `--group-threshold` | `0.07` | Cosine-distance cutoff for merging; smaller is stricter (fewer merges). |
| `--no-merge-guard` | off | Disable the lexical guard (lets co-hyponyms merge; not recommended). |
| `--context-windows` | `3` | Occurrence windows per candidate folded into its embedding text. |
| `--context-width` | `60` | Context characters on each side of an occurrence. |
| `--max-candidates` | `0` (all) | Cap candidates **per kind** (strongest by mention count) for a quick test. |
| `--batch-size` | `256` | Inputs per embedding request (OpenAI allows up to 2048). |
| `--cache PATH` | `<stem>_embcache.npz` | Embedding cache; reuse makes re-tuning the threshold instant. |
| `--no-cache` | off | Do not read or write the embedding cache. |
| `--env PATH` | auto | Explicit path to a `.env` holding `OPENAI_API_KEY`. |
| `--output` | auto | Override the output path. |

Examples:

```bash
# Full semantic pass (OpenAI embeddings over every candidate)
python name_vocabulary.py INPUT_candidates.json

# Offline, no API calls (acronym merges only)
python name_vocabulary.py INPUT_candidates.json --no-embeddings

# Quick test on the 300 strongest candidates per kind
python name_vocabulary.py INPUT_candidates.json --max-candidates 300

# Looser merging (more synonyms merged, more risk of over-merge)
python name_vocabulary.py INPUT_candidates.json --group-threshold 0.10

# Shorter (cheaper) embeddings
python name_vocabulary.py INPUT_candidates.json --embed-dim 512
```

## Output schema

`<stem>_candidates_vocabulary.json`:

```jsonc
{
  "stage": "2-name-vocabulary",
  "candidates_file": "..._candidates.json",
  "input_file": "....md",
  "generated_at": "ISO-8601 UTC",
  "config": {
    "grouping": "semantic",            // "semantic" | "lexical"
    "embed_model": "text-embedding-3-small", "embed_dim": 1536,
    "group_threshold": 0.07, "context_windows": 3, "context_width": 60,
    "max_candidates": 0
  },
  "stats": {
    "candidate_concepts_in": 0, "candidate_relations_in": 0,
    "concepts": 0, "relations": 0, "elements_total": 0,
    "dropped_concepts": 0, "dropped_relations": 0,
    "merge_ratio_concepts": 0.0, "merge_ratio_relations": 0.0,
    "literal_span_elements": 0, "llm_only_elements": 0
  },
  "concepts": [
    {
      "id": "concept:incidentcommander",
      "label": "IncidentCommander",        // canonical, PascalCase for concepts
      "kind": "concept",
      "representative": "incident commander",  // surface the label was coined from
      "alt_labels": ["IC", "incident commander", "the commander"],
      "members": ["ic", "incident commander"],  // Stage 1 candidate keys merged here
      "sources": ["llm", "ner", "term"],   // pooled extractor tags
      "ner_labels": [],
      "occurrences": [ { "start": 1423, "end": 1440, "text": "incident commander" } ],  // pooled, capped at 50
      "mention_count": 42,                 // true total across members
      "literal_span": true,                // at least one member is verbatim in the source
      "llm_only": false                    // no hallucination-free extractor backed any member
    }
  ],
  "relations": [
    { "id": "relation:isdivisibleby", "label": "isDivisibleBy", "kind": "relation", ... }
  ],
  "dropped": [
    { "label": "Page", "kind": "concept", "drop_reason": "document boilerplate, not a domain concept", ... }
  ]
}
```

## Files

```
stage2-name-vocabulary/
├── SKILL.md                 # this file
├── requirements.txt
├── scripts/
│   ├── name_vocabulary.py   # orchestrator + CLI (run this)
│   ├── embed.py             # OpenAI context-enriched embeddings (batched, cached, 429 backoff)
│   ├── group.py             # semantic clustering + lexical (acronym) fallback
│   ├── selection.py         # minimal selection (drop only obvious non-concepts)
│   ├── name.py              # representative pick + canonical labeling + merge
│   └── element.py           # VocabularyElement data model
└── references/
    └── method.md            # the naming method in detail
```

## Requirements

- Python 3.10+
- `numpy`
- `openai` (for embeddings; not needed with `--no-embeddings`)
- `OPENAI_API_KEY` in a `.env` file (for embeddings). The default model
  `text-embedding-3-small` runs comfortably within OpenAI's Tier-1 limits
  (3,000 RPM / 1,000,000 TPM) for a corpus of this size.

## Output and what comes next

The artifact is a flat vocabulary of named concepts and relations, each with its
lexicalizations and pooled provenance, and **nothing else**: no parents, no edges,
no domain/range. Stage 3 (synthesize structure) takes this vocabulary and wires it
into an `rdfs:subClassOf` DAG with domain/range and axioms.

## Notes and limitations

- **Merge threshold is a dial, not a truth.** `--group-threshold` trades recall of
  synonym merges against the risk of merging distinct-but-related terms
  (`number` vs `whole number`). The default is conservative; tighten it
  (`0.10`) if you see over-merging, loosen it if obvious synonyms stay split.
  Stage 4 review is where remaining merge errors are corrected.
- **Embeddings are cached.** Vectors are cached to `<stem>_embcache.npz`, so
  re-running at a different `--group-threshold` is instant (the cache is keyed by
  model and dimension). Delete the cache (or `--no-cache`) to force a re-embed,
  for example after changing the context window.
- **The lexical guard is what makes the threshold safe.** In a tight domain (a
  single math chapter), unrelated concepts already sit close in embedding space.
  The guard (shared content token or acronym match) blocks co-hyponym and antonym
  merges that a pure distance threshold cannot. Disable it with `--no-merge-guard`
  only if you specifically want distance-only merging.
- **Selection is intentionally light.** Document boilerplate and a few one-off
  noise terms are dropped; everything else, including plenty of imperfect terms,
  survives to review. Do not tighten selection here to "clean up" the output;
  that is Stage 4's job and doing it here violates recall-first.
- **Offsets stay authoritative.** Pooled `occurrences` keep the original Stage 1
  offsets; each `text` still equals `source[start:end]` in the original file.
```
