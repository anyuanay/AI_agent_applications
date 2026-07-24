"""Preprocessing for Stage 1: Markdown cleaning, exercise block removal, KG sentence filter.

Call order:
  1. remove_exercise_blocks(raw_text)   — runs on raw Markdown (needs # markers)
  2. clean_markdown(text)               — strips remaining Markdown syntax
  3. is_kg_sentence(spacy_span)         — called per sentence inside spacy_pass.run()
"""
from __future__ import annotations

import re


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXERCISE BLOCK REMOVAL   (operates on raw Markdown, before # is stripped)
# ══════════════════════════════════════════════════════════════════════════════

# Heading text that opens a block to remove (tested against the text AFTER the # prefix)
_REMOVE_HEADING_RE = re.compile(
    r"^(?:"
    r"EXAMPLE\s+\d"               # EXAMPLE 1.1 / EXAMPLE 1.7 HOW TO ...
    r"|Solution"                   # Solution / SOLUTION
    r"|HOW TO\b"                   # HOW TO :: ...
    r"|BE PREPARED\b"
    r"|MANIPULATIVE MATHEMATICS"
    r"|Learning Objectives?"
    r"|Practice Makes Perfect"
    r"|Writing Exercises?"
    r"|Everyday Math"
    r"|Self Check"
    r"|Practice\s+Test"
    r"|Review\s+Exercises?"
    r"|Key\s+(?:Concepts?|Terms?|Takeaways?)"
    r"|Chapter\s+(?:Review|Outline)"
    r"|TRY IT\b"
    r")",
    re.IGNORECASE,
)

# A numbered content section heading that always ends any skip
# e.g. "# 1.2 Use the Language of Algebra"
_SECTION_HEADING_RE = re.compile(
    r"^(#{1,6})\s+\d+\.\d+\s+(?!EXERCISES)\S",
    re.IGNORECASE,
)

# Bare exercise-section marker: "1.1 EXERCISES" with no # prefix
_BARE_EXERCISES_RE = re.compile(r"^\s*\d+\.\d+\s+EXERCISES\b", re.IGNORECASE)

# Inline blockquote triggers (> TRY IT, > HOW TO)
_BLOCKQUOTE_SKIP_RE = re.compile(r"^>+\s*(?:TRY IT|HOW TO)\b", re.IGNORECASE)


def remove_exercise_blocks(text: str) -> str:
    """Remove exercise sets, worked examples, TRY IT, HOW TO, and end-of-section blocks.

    Uses a three-mode state machine:
      'content'         — include lines normally
      'skip_section'    — skip until the next numbered section heading (e.g. # 1.2 ...)
      'skip_heading'    — skip until a heading at the same or shallower depth
      'skip_paragraph'  — skip until the next blank line (for inline blockquotes)
    """
    lines = text.split("\n")
    out: list[str] = []
    mode = "content"
    skip_depth = 0      # heading depth that triggered 'skip_heading'

    for line in lines:
        stripped = line.strip()

        # Parse Markdown heading
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        is_heading = bool(m)
        h_depth = len(m.group(1)) if m else 0
        h_text  = m.group(2).strip() if m else ""

        # ── Bare EXERCISES line ───────────────────────────────────────────────
        if _BARE_EXERCISES_RE.match(line):
            mode = "skip_section"
            continue

        # ── MODE: skip_section ────────────────────────────────────────────────
        if mode == "skip_section":
            if _SECTION_HEADING_RE.match(line):
                mode = "content"
                out.append(line)
            # any other line: stay in skip
            continue

        # ── MODE: skip_heading ────────────────────────────────────────────────
        if mode == "skip_heading":
            if is_heading and h_depth <= skip_depth:
                # This heading ends the skip — is it also a block to remove?
                if _REMOVE_HEADING_RE.match(h_text):
                    skip_depth = h_depth   # start a new remove block at same level
                else:
                    mode = "content"
                    out.append(line)
            # else: stay in skip
            continue

        # ── MODE: skip_paragraph ──────────────────────────────────────────────
        if mode == "skip_paragraph":
            if not stripped:               # blank line ends the paragraph skip
                mode = "content"
                out.append(line)
            continue

        # ── MODE: content — check for new skip triggers ───────────────────────
        if is_heading and _REMOVE_HEADING_RE.match(h_text):
            mode = "skip_heading"
            skip_depth = h_depth
            continue

        if _BLOCKQUOTE_SKIP_RE.match(stripped):
            mode = "skip_paragraph"
            continue

        out.append(line)

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# 2. MARKDOWN CLEANING   (runs after exercise removal)
# ══════════════════════════════════════════════════════════════════════════════

def clean_markdown(text: str) -> str:
    """Strip Markdown and LaTeX syntax, preserving plain-text content."""
    # YAML front matter
    text = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    # HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Fenced code blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`[^`\n]*`", " ", text)
    # Images: ![alt](url) → ''
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    # Links: [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Markdown table rows
    text = re.sub(r"^\|.*\|.*$", "", text, flags=re.MULTILINE)
    # Headings: ## Text → Text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Blockquote markers
    text = re.sub(r"^>+\s*", "", text, flags=re.MULTILINE)
    # Bold / italic
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_\n]+)_{1,3}", r"\1", text)
    # HTML entities
    text = (text.replace("&gt;", ">").replace("&lt;", "<")
                .replace("&amp;", "&").replace("&nbsp;", " "))
    # LaTeX display math $$...$$ (multi-line blocks)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    # LaTeX inline math $...$ (single-line formulas)
    text = re.sub(r"\$[^$\n]+\$", "", text)
    # Circled digit markers ①–⑩
    text = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", "", text)
    # Standalone horizontal rules
    text = re.sub(r"^\s*---\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to 1
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 3. SENTENCE-LEVEL KG FILTER   (called inside spacy_pass.run() per sentence)
# ══════════════════════════════════════════════════════════════════════════════

# Sentence-start patterns that never yield useful KG triples
_NOISE_PREFIX_RE = re.compile(
    r"^(?:"
    r"Figure\s+\d"              # figure captions
    r"|Step\s+\d"               # procedure steps
    r"|Chapter\s+\d"            # chapter header lines
    r"|Solution\b"              # bare "Solution" blocks (exercise artifacts)
    r"|This\s+OpenStax\b"       # boilerplate attribution
    r"|http\S+"                 # bare URLs
    r"|\d{1,3}\.\s"             # numbered exercise items (e.g. "1. In the following…")
    r"|[A-Z]\s+[A-Z]\b"        # abbreviated table entries (A B C D)
    r")",
    re.IGNORECASE,
)

# Sentences that contain no alphabetic characters at all
_NO_ALPHA_RE = re.compile(r"^[^a-zA-Z]*$")


def is_kg_sentence(sent) -> bool:  # sent: spaCy Span
    """Return True only if the sentence is likely to contain a KG-extractable triple.

    Criteria (all must hold):
      - At least 6 tokens
      - At least one VERB
      - At least one NOUN or PROPN
      - Does not match known noise-start patterns
      - Not purely numeric / symbolic
    """
    text = sent.text.strip()

    if _NOISE_PREFIX_RE.match(text):
        return False

    if _NO_ALPHA_RE.match(text):
        return False

    toks = list(sent)

    if len(toks) < 6:
        return False

    if not any(tok.pos_ == "VERB" for tok in toks):
        return False

    if not any(tok.pos_ in ("NOUN", "PROPN") for tok in toks):
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# 4. FULL TEXT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(text: str) -> tuple[str, dict]:
    """Run exercise removal then Markdown cleaning. Returns (cleaned_text, stats)."""
    original_chars = len(text)
    text = remove_exercise_blocks(text)
    after_exercise = len(text)
    text = clean_markdown(text)
    after_markdown = len(text)
    return text, {
        "original_chars":      original_chars,
        "after_exercise_removal": after_exercise,
        "after_markdown_clean":   after_markdown,
        "chars_removed":       original_chars - after_markdown,
        "pct_removed":         round(100 * (1 - after_markdown / original_chars), 1),
    }
