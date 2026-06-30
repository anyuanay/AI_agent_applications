---
name: stage1-surface-candidates
description: Surfaces a high-recall, zero-structure bag of candidate ontology terms (concepts and relations) from a source document by unioning a hallucination-free spaCy extractor (NER + noun-phrase terms + verb/copular predicates) with a Gemini LLM extractor, then string-level dedup. Use when you need to extract candidate concepts and relations from text as the first step of building or growing an ontology, or when asked to "surface candidates" or "extract ontology terms". Writes <INPUT_FILE_stem>_candidates.json next to the input.
---

# Surface Candidates

This skill performs the **surfacing** step of ontology extraction: read a source
document and produce a high-recall bag of candidate ontology terms, both
**concepts** and **relations**, each tagged with its provenance. It is the first
step of a larger pipeline; later steps (naming the vocabulary, inducing
structure, reviewing) are out of scope here and are where pruning and structure
happen.

The single job of this step: **maximize recall of terms that could become
ontology elements, with zero structural commitment.** The input is a source
document; the output is a noisy *bag of candidate mentions*, each one a
concept-mention or a relation-mention. There are no parents, no edges, no
domain/range. Surfacing deliberately over-generates; precision is bought back
later.

## When to use this skill

- You have a source document (text or markdown) and want the candidate concepts
  and relations that could seed or grow an ontology.
- You need a clean, provenance-bearing candidate set for a downstream
  naming/structuring step to consume.
- The user asks to "surface candidates", "extract ontology terms / concepts", or
  "find candidate classes and relations".

Do **not** use this skill to build a taxonomy, assign parents, or prune terms.
Those belong to later steps. This step intentionally keeps everything plausible.

## Three design principles

- **Recall-first.** Nothing the text supports is dropped here. The junk filter
  removes only markup debris (table pipes, HTML entities, image paths), never
  plausible terms. Precision is recovered later, never now.
- **Provenance everywhere.** Every candidate carries where it came from: the
  extractor(s) that surfaced it and the exact character offsets of every literal
  occurrence in the original file. By construction, each recorded occurrence text
  equals `source[start:end]`.
- **Grounding by kind.** Each mention is typed now, `concept` or `relation`. What
  it means to justify it downstream depends on this kind.

## Mechanism: union of two extractors, then dedup

Neither extractor alone is enough. **The union is the point:** the spaCy
extractor cannot hallucinate but misses the implicit; the LLM catches the
implicit but can hallucinate.

### spaCy extractor: the recall floor and hallucination-free anchor

`scripts/spaCy_extractor.py`. Every candidate it emits is a literal span copied
out of the text, so it physically cannot invent a term that is not there.

**Concept mentions** come mostly from noun phrases, with POS/TAG/DEP used to
sharpen what counts:

- **NER** (`doc.ents`): named things.
- **Noun-phrase terms** (`doc.noun_chunks` whose head is a `NOUN`/`PROPN`, with
  leading determiners/possessives trimmed): the recall workhorse. Most concepts
  surface here, single-word (`fraction`, `integer`) and multi-word
  (`whole number`, `least common multiple`).
- **Gerund nominalizations** (`tag_ == VBG` in a nominal dependency role such as
  subject or object of a preposition): concepts whose head is a verb form, so
  `noun_chunks` never yields them (`rounding`, `factoring`, `graphing`).
- **Adjective qualities** (`pos_ == ADJ` in an `amod`/`acomp` role), **opt-in via
  `--adjectives`**, tagged `adj`: qualities of things (`prime`, `even`,
  `rational`, `divisible`). Off by default because it also surfaces generic
  adjectives (`available`, `basic`, `correct`); enable it when you want
  candidate qualities and will prune them downstream.

**Stop words are never surfaced on their own.** Every concept must keep at least
one non-stop content word, so a lone `two`, `it`, or `the` is dropped. Leading
determiners and pronouns are trimmed (`the foundation` → `foundation`), but
content words that merely sit on spaCy's stop list (`whole`, `first`) are kept,
so `whole numbers` stays intact. Function words inside a relation predicate (the
copula and prepositions in `is divisible by`) are kept on purpose.

**Relation mentions are not only verbs.** Relations live in several syntactic
shapes, and the extractor surfaces all of them, each as a contiguous literal span
so its offsets re-slice exactly to the predicate text:

| Shape | spaCy signal | Examples |
|---|---|---|
| Bare verb | `pos_ == VERB` | `add`, `divide`, `simplify` |
| Verb + particle | verb + `prt` child | `add up`, `carry over`, `write out` |
| Verb + preposition | verb + `prep`/`agent` child | `divide by`, `consist of`, `depend on`, `rounded to` |
| Copular predicate | `be` + `acomp`/`attr` complement + preposition | `is divisible by`, `is equal to`, `is a multiple of` |

Notes on relation extraction:

- Bare auxiliaries and over-generic verbs (`be`, `have`, `do`, `make`, `use`,
  ...) are dropped as standalone relations, but `be` still seeds **copular**
  predicates (`is equal to`).
- Copular predicates require a preposition, so relational predicates
  (`is divisible by`) are kept while bare attributes (`is prime`) are not.
- Verb phrases are emitted *in addition to* the bare verb (recall-first), so
  `divided by` yields both `divide` and `divide by`.
- The dependency-parse extension is contiguous only; when an object interrupts
  the predicate (`divide the number by ...`), only the bare verb is emitted, so
  provenance offsets stay exact.

### LLM extractor (Gemini): the complementary lift

`scripts/llm_extractor.py`, model `gemini-3.1-flash-lite`. It catches the
multi-word, implicit, and relational candidates the spaCy methods miss (it reads
"the value left over after division" and proposes `Remainder` even though no
clean noun phrase names it). Two rules are enforced:

- **Terms only.** The model is prompted for two flat lists (concepts, relations).
  Any structure it volunteers (parents, hierarchy) is discarded, because
  structure is a later step's job and the model is least trustworthy exactly when
  it guesses hierarchy.
- **Tag the lift.** Every LLM term is tagged `llm`. The orchestrator then checks
  each term against the source: terms found verbatim earn literal-span
  provenance; terms not in the text are kept anyway (recall-first) but stay
  `literal_span: false` and `llm_only: true`, marking them for a harder look
  downstream.

The LLM API key is read from `GOOGLE_API_KEY` in a `.env` file (an explicit
`--env` path, the environment, or the nearest `.env` walking up from the scripts
directory).

### Dedup: make the union a set

`scripts/dedup.py`. Lexical and morphological only: casefold, lemmatize, merge
string variants (`Whole Numbers` / `whole numbers` into one candidate keyed
`whole number`; `divides by` / `divided by` into `divide by`). Sources,
occurrences, and NER labels are pooled. This is **string-level** dedup;
meaning-level merging (`LCM` with `least common multiple`) is a harder operation
left to a later step.

## How to run

```bash
cd scripts
python surface_candidates.py <INPUT_FILE> [options]
```

The result is written next to the input as `<INPUT_FILE_stem>_candidates.json`.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--no-llm` | off | spaCy extractor only. No API key needed; fully offline and deterministic. |
| `--adjectives` | off | Also surface adjective qualities (`ADJ` in `amod`/`acomp`) as concepts, tagged `adj`. Adds qualities like `prime`/`even` at the cost of generic-adjective noise. |
| `--model` | `gemini-3.1-flash-lite` | Gemini model id for the LLM extractor. |
| `--spacy-model` | `en_core_web_sm` | spaCy model (must be installed). |
| `--llm-chunk-chars` | `8000` | Characters per LLM chunk. |
| `--llm-max-chunks` | `0` (all) | Cap LLM chunks, **evenly sampled** across the document, to bound cost/time. The spaCy extractor always covers the whole document. |
| `--env` | auto | Explicit path to a `.env` holding `GOOGLE_API_KEY`. |
| `--output` | auto | Override the output path. |

Examples:

```bash
# Full pass: spaCy over the whole document, LLM over every chunk
python surface_candidates.py INPUT.md

# Bounded LLM cost on a large file: 12 evenly-sampled chunks
python surface_candidates.py INPUT.md --llm-max-chunks 12

# Offline, deterministic, no API calls
python surface_candidates.py INPUT.md --no-llm

# Also surface adjective qualities (prime, even, rational) as concepts
python surface_candidates.py INPUT.md --adjectives
```

## Output schema

`<stem>_candidates.json`:

```jsonc
{
  "stage": "1-surface-candidates",
  "input_file": "....md",
  "generated_at": "ISO-8601 UTC",
  "config": { "spacy_model": "...", "adjectives": false, "llm_model": "...", "llm_chunk_chars": 8000, "llm_max_chunks": 0 },
  "stats": {
    "raw_spacy_mentions": 0, "raw_llm_mentions": 0, "llm_chunks_processed": 0,
    "candidates_total": 0, "concepts": 0, "relations": 0,
    "literal_span_candidates": 0, "llm_only_candidates": 0, "implicit_candidates": 0
  },
  "candidates": [
    {
      "canonical": "whole numbers",        // human-facing surface (most frequent variant)
      "key": "whole number",               // normalized dedup key (lemmatized, casefolded)
      "kind": "concept",                   // "concept" | "relation"  (grounding by kind)
      "sources": ["llm", "term"],          // subset of: ner, term, adj, verb, llm
      "variants": ["Whole Numbers", "whole numbers"],
      "occurrences": [                      // every literal appearance; text == source[start:end]
        { "start": 1423, "end": 1436, "text": "whole numbers" }
      ],
      "ner_labels": [],
      "mention_count": 13,
      "literal_span": true,                // copied verbatim from source at least once, so not a hallucination
      "llm_only": false                    // only the LLM surfaced it (no hallucination-free backing)
    }
  ]
}
```

Source tags on `sources`:

- `ner`: spaCy named entity (concept).
- `term`: spaCy noun-phrase term or gerund nominalization (concept).
- `adj`: spaCy adjective quality (concept); only present with `--adjectives`.
- `verb`: spaCy syntactic predicate (relation), covering bare verbs, verb
  phrases, and copular predicates.
- `llm`: language-model proposal (concept or relation).

Useful flags for downstream steps:

- **`literal_span: false`**: an implicit candidate (described by the LLM but not
  present in the text verbatim). Kept by recall-first; needs grounding later.
- **`llm_only: true`**: no spaCy, hallucination-free extractor backed it. Worth a
  harder look downstream.
- **`sources` with more than one extractor**: agreement, a stronger candidate.

## Files

```
stage1-surface-candidates/
├── SKILL.md                      # this file
├── requirements.txt
├── scripts/
│   ├── surface_candidates.py     # orchestrator + CLI (run this)
│   ├── spaCy_extractor.py        # spaCy NER + noun phrases + gerunds + verb/copular predicates
│   ├── llm_extractor.py          # Gemini terms-only extractor (chunked)
│   ├── dedup.py                  # lexical/morphological union -> set
│   ├── candidate.py              # Candidate / RawCandidate / Occurrence models
│   └── textnorm.py               # lemmatized, casefolded dedup keys
└── references/
    └── method.md                 # the surfacing method in detail
```

## Requirements

- Python 3.10+
- `spacy` with `en_core_web_sm` (`python -m spacy download en_core_web_sm`)
- `google-genai` (for the LLM extractor; not needed with `--no-llm`)
- `GOOGLE_API_KEY` in a `.env` file (for the LLM extractor)

## Output and what comes next

The artifact is a flat bag of typed, deduplicated, provenance-bearing candidate
mentions, and nothing else: no identity beyond string-level dedup, no structure.
A downstream naming step can take this bag and perform meaning-level grouping and
canonical labeling.

## Notes and limitations

- **Over-generation is intentional.** Document boilerplate (`Page`, `Step`,
  `EXAMPLE`, section titles) will appear as candidates. Recall-first forbids
  dropping them here. Do not "fix" this by tightening the filter past markup
  debris; pruning is a later step's job.
- **LLM failures degrade gracefully.** A network or parse error on a chunk skips
  that chunk (the spaCy floor still stands) rather than aborting the run.
- **Offsets are authoritative.** Occurrence offsets index the *original* input
  file; the recorded `text` is re-sliced from the source so it always equals
  `source[start:end]`.
- **Relation parsing depends on the spaCy model.** Verb phrases and copular
  predicates use the dependency parse; some predicates (for example certain
  comparatives) may not be caught by the spaCy extractor, but the LLM extractor
  covers the gap.
- **English by default.** The default spaCy model is English; pass
  `--spacy-model` for another language.
```
