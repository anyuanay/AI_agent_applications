---
name: stage1b-classify-candidates
description: The type/instance gate of ontology extraction. Sits between surfacing (Stage 1) and naming (Stage 2) and sorts every surfaced candidate into class (a universal that belongs in the ontology), individual (a particular such as a specific number, date, money amount, or named person, routed to an A-Box instances file for a later knowledge-graph pass), or non_concept (LaTeX/markup debris or numbered document labels, parked as feedback). Uses deterministic morphology (digits, math/markup symbols, document-structure words) plus an optional gemini-3.1-flash-lite tiebreak that uses the Stage 0 scope to catch named individuals NER cannot. Use after surfacing candidates and before naming, or when asked to "filter instances", "keep only concepts", "separate the T-Box from the A-Box", or "drop example values". Reads a <stem>_candidates.json and writes <stem>_candidates_gated.json next to it (its candidates key holds the class candidates for Stage 2).
---

# Classify Candidates (Type/Instance Gate)

This skill performs the **gating** step of ontology extraction: separate the
universals from the particulars. An ontology is a conceptualization of a domain,
so it should contain general **classes** (`Fraction`, `Equation`, `Operation`),
not **individuals** (a specific value `$\frac{1}{5}$`, a worked number `144`, a
person `Marissa`) and not document debris. Bottom-up surfacing cannot tell these
apart; this gate does, before naming commits them to vocabulary.

The single job of this step: **keep only what is a kind of thing.** Particulars
are not deleted, they are routed to an A-Box instances file (preserved for the
knowledge-graph pass, Article 5); debris is parked as feedback. Nothing is lost.

## When to use this skill

- You have a Stage 1 candidates file and the source is instance-dense (a textbook,
  a manual full of worked examples, a dataset description with sample values).
- You want the ontology to hold concepts, not the specific numbers and names the
  text happens to use.
- The user asks to "filter instances", "keep only concepts", "separate T-Box from
  A-Box", or "drop example values".

## Why a separate gate

The type/instance distinction is the single highest-leverage fix for an
over-large, instance-polluted ontology. Doing it here (before naming and
structuring) keeps the downstream stages small and on-topic, and it keeps the
particulars available for a later knowledge-graph extraction rather than throwing
them away.

## Mechanism

### Deterministic morphology gate (no LLM)

`scripts/gate.py`. The reliable signal on a real corpus is morphology, **not**
spaCy NER labels (which on this data tag `equation` as MONEY and `fraction` as
PERSON, so routing on them would discard core concepts):

- **non_concept**: LaTeX/markup/HTML debris (`\`, `$`, `{}`, `^`, `_`, `|`, `<`,
  `>`, `&`), pure symbols, or a numbered document-structure label
  (`Chapter 1 Foundations`).
- **individual**: contains a numeric value (`3 years`, `49-cent stamps`,
  `2nd month`), the clearest mark of a particular.
- **class**: a clean common-noun phrase with no digits or debris. Relations are
  object properties and pass through unless they are debris/numeric.

### Optional LLM tiebreak (`--llm`)

`scripts/llm_classify.py` (`gemini-3.1-flash-lite`). After the morphology gate,
the survivors are clean common nouns, most of which are classes but some of which
are named individuals with no numeric tell (`Marissa`, `Tuesday`, a place, a
brand). Given the Stage 0 **domain statement** and **out-of-scope** note, the LLM
re-sorts the class survivors into class / individual / non_concept. This is a
tiebreak over the survivors, not the primary filter, so its cost is bounded and a
failure degrades to the deterministic result.

## How to run

```bash
cd scripts
python classify_candidates.py <CANDIDATES_JSON> [--llm] [--scope PATH]
```

The result is written next to the candidates file as
`<CANDIDATES_JSON_stem>_gated.json`. Its `candidates` key holds the **class**
candidates, so Stage 2 (name the vocabulary) consumes it directly. The
`individuals` and `non_concepts` keys hold the routed-out records.

| Option | Default | Meaning |
|---|---|---|
| `--scope PATH` | auto-detect | Stage 0 scope JSON, for the LLM tiebreak's domain context. |
| `--llm` | off | LLM tiebreak over class survivors (`gemini-3.1-flash-lite`). |
| `--llm-model` | `gemini-3.1-flash-lite` | Gemini model id. |
| `--env PATH` | auto | Explicit `.env` holding `GOOGLE_API_KEY` (only with `--llm`). |
| `--output PATH` | auto | Override the output path. |

```bash
# Deterministic gate only (no API calls)
python classify_candidates.py INPUT_candidates.json

# With the LLM tiebreak (uses the auto-detected Stage 0 scope)
python classify_candidates.py INPUT_candidates.json --llm
```

## Output schema

`<stem>_candidates_gated.json`:

```jsonc
{
  "stage": "1b-classify-candidates",
  "candidates_file": "..._candidates.json", "input_file": "..._content.md",
  "scope_file": "..._scope.json",
  "generated_at": "ISO-8601 UTC",
  "config": { "llm": true, "llm_model": "gemini-3.1-flash-lite" },
  "stats": {
    "in_candidates": 0, "classes": 0, "class_concepts": 0, "class_relations": 0,
    "individuals": 0, "non_concepts": 0, "llm_moved": 0, "by_reason": { "...": 0 }
  },
  "candidates": [ /* class candidates, Stage 1 schema -> Stage 2 reads this */ ],
  "individuals": [ /* particulars (A-Box), each with a gate_reason */ ],
  "non_concepts": [ /* debris/boilerplate, each with a gate_reason */ ]
}
```

## Files

```
stage1b-classify-candidates/
├── SKILL.md                 # this file
├── requirements.txt
├── scripts/
│   ├── classify_candidates.py  # orchestrator + CLI (run this)
│   ├── gate.py                 # deterministic morphology gate
│   └── llm_classify.py         # optional gemini-3.1-flash-lite tiebreak
└── references/
    └── method.md               # the gating method in detail
```

## Requirements

- Python 3.10+ (deterministic gate is pure standard library)
- `google-genai` + `GOOGLE_API_KEY` only when `--llm` is used

## Output and what comes next

The `candidates` (classes) feed Stage 2 (name the vocabulary), now free of numeric
particulars and debris. The `individuals` are the seed of the domain's A-Box: a
later knowledge-graph extraction (Article 5) can attach them to the ontology's
classes as instances. The `non_concepts` are feedback, available if a later pass
wants to revisit them.

## Notes and limitations

- **NER is deliberately not used for routing.** It is too noisy on real text;
  morphology plus the LLM tiebreak is both safer and more accurate.
- **Morphology is conservative.** A digit-less named individual passes the
  deterministic gate and is only caught with `--llm`. Without the LLM, a few
  proper names survive into Stage 2, where salience (Stage 2b) and review (Stage 4)
  can still catch them.
- **Recall-first, relocated.** Particulars and debris are routed, not deleted, so
  the recall-first invariant holds: the ontology gets smaller while nothing is
  thrown away.
