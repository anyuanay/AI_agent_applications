---
name: stage0-scope
description: Frames an ontology before extracting it. Mines a source document's structure (table of contents, learning objectives, explicit "is/are called X" definitions, bold key terms) into a high-precision skeleton, then (optionally, with gemini-3.1-flash-lite) synthesizes a domain statement, cleaned topics, general key concepts and relations, competency questions, and an explicit out-of-scope note that names the instance/fact kinds to keep OUT of the ontology. This top-down scope bounds the bottom-up pipeline so it does not drown in worked-example instances. Use as the first step of ontology extraction (before surfacing candidates), or when asked to "scope the ontology", "summarize the domain", "find the key topics/concepts", or "define competency questions". Reads a source document and writes <input_stem>_scope.json next to it.
---

# Extract Scope

This skill performs the **scoping** step of ontology extraction: frame the
ontology before extracting it. An ontology is a conceptualization of a domain (its
general concepts and relationships), not a dump of every noun and number in a
text. The bottom-up pipeline (surface, name, structure, review) has no notion of
what the document is *about* or at what abstraction level the ontology should sit,
so on an instance-dense source it over-generates badly. Stage 0 supplies the
missing top-down spine.

The single job of this step: **say what the ontology is about, and at what level.**
The output is a scope artifact: a domain statement, the topics, the seed concepts
the author themselves defined, general relations, competency questions that bound
scope, and an out-of-scope note that names the instance/fact kinds to exclude.

## When to use this skill

- You are about to extract an ontology and want to frame and bound it first.
- You have a structured source (a textbook, a manual, a spec) whose headings,
  objectives, and definitions enumerate the domain's real concepts.
- The user asks to "scope the ontology", "summarize the domain", "find the key
  topics", or "write competency questions".

## Why scope first

- **Abstraction.** Topics and competency questions fix the level: general kinds
  (`Fraction`, `Operation`, `Equation`), not worked-example values.
- **Type/instance boundary.** The out-of-scope note tells the later type/instance
  gate exactly what to exclude (specific numbers, specific worked examples).
- **Grounding.** Seed concepts come from what the author marked important
  (definitions, objectives, the table of contents), not from a model's guess.

## Mechanism

### Deterministic structure mining (no LLM)

`scripts/structure.py` pulls the author's own scaffold out of the markup:

- **Topics** from the chapter outline (`1.1 Introduction to Whole Numbers`) and
  the non-boilerplate section headings, filtered for math/markup debris and
  document scaffolding (Example, Solution, Try It, Exercises, ...).
- **Objectives** from the "you will be able to" bullet lists.
- **Defined terms** from "is/are called X" and "X is defined as" cues and from
  bold `**Term:**` markers. These are the highest-precision concept seeds.

### LLM synthesis (optional, `--llm`)

`scripts/llm_scope.py` (`gemini-3.1-flash-lite`) reads only the compact skeleton
(never the raw text, never inventing a domain) and writes the parts that need
synthesis: the **domain statement**, cleaned **topics**, general **key concepts**
and **relations**, **competency questions**, and the **out-of-scope** note. It is
asked for general concepts, so it raises abstraction without fabricating content.

## How to run

```bash
cd scripts
python extract_scope.py <INPUT_FILE> [--llm]
```

The result is written next to the input as `<INPUT_FILE_stem>_scope.json`.

| Option | Default | Meaning |
|---|---|---|
| `--llm` | off | Use `gemini-3.1-flash-lite` to synthesize statement / concepts / questions / out-of-scope. |
| `--llm-model` | `gemini-3.1-flash-lite` | Gemini model id. |
| `--env PATH` | auto | Explicit `.env` holding `GOOGLE_API_KEY` (only with `--llm`). |
| `--output PATH` | auto | Override the output path. |

```bash
# Deterministic skeleton only (no API calls)
python extract_scope.py INPUT.md

# Full scope with domain statement, key concepts, and competency questions
python extract_scope.py INPUT.md --llm
```

## Output schema

`<stem>_scope.json`:

```jsonc
{
  "stage": "0-scope",
  "input_file": "....md",
  "generated_at": "ISO-8601 UTC",
  "config": { "llm": true, "llm_model": "gemini-3.1-flash-lite" },
  "domain_statement": "This ontology covers the fundamental concepts of elementary algebra ...",
  "topics": ["Introduction to Whole Numbers", "Prime Number and Composite Number", "..."],
  "key_terms": ["counting numbers", "prime factorization", "Number", "Variable", "Factor", "..."],
  "relations": ["is a factor of", "is a multiple of", "is a solution to", "simplifies to"],
  "competency_questions": ["What is the relationship between factors and multiples?", "..."],
  "out_of_scope": ["Specific numerical values like 5, 10, or 100", "Specific worked examples", "..."],
  "structure": { "topics": ["..."], "objectives": ["..."], "defined_terms": ["..."] },
  "stats": { "topics": 0, "key_terms": 0, "objectives": 0, "defined_terms": 0, "competency_questions": 0 }
}
```

## Files

```
stage0-scope/
├── SKILL.md                 # this file
├── requirements.txt
├── scripts/
│   ├── extract_scope.py     # orchestrator + CLI (run this)
│   ├── structure.py         # deterministic TOC / objectives / definitions mining
│   └── llm_scope.py         # optional gemini-3.1-flash-lite scope synthesis
└── references/
    └── method.md            # the scoping method in detail
```

## Requirements

- Python 3.10+ (deterministic mining is pure standard library)
- `google-genai` + `GOOGLE_API_KEY` only when `--llm` is used

## Output and what comes next

The scope artifact frames the rest of the pipeline. Stage 1 (surface candidates)
still casts a wide net, but Stage 1b (the type/instance gate) uses the domain
statement and out-of-scope note to route instances out of the ontology, Stage 2b
(salience) boosts the seed concepts, and a synthesis step is scoped by the topics
and competency questions. The scope is what turns a bottom-up, abstraction-blind
extractor into a focused one.

## Notes and limitations

- **Structure quality drives scope quality.** A well-marked source (TOC,
  objectives, definitions) yields a strong skeleton; an unstructured wall of text
  yields less, and the `--llm` synthesis carries more of the load.
- **The skeleton, not the corpus, goes to the LLM.** This keeps the call cheap and
  keeps the model grounded in the author's own structure rather than free-writing.
- **Scope is advisory, not a hard filter here.** Stage 0 never drops content; it
  produces guidance that later stages apply. Recall-first still holds.
