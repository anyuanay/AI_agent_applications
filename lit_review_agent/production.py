"""
Production engineering (Part 13).

What separates a demo from a product is managing variance and cost at scale, not
model intelligence. Three levers from the article, made concrete:

  - model routing : match the model to the step's difficulty. The cheap model
                    plans and searches; the frontier model only synthesizes (the
                    step the user sees). A misroute is a real quality failure, so
                    routing is gated by the Part 12 eval suite — "routing without
                    an eval suite is just hoping."
  - version pinning: reproducibility is manufactured because determinism is a
                    fiction (Part 2). Pin model id + prompt version + tool
                    versions so a silent model upgrade is not an overnight
                    regression.
  - cost rollup   : the span tree is a cost tree (Part 11). Roll the per-span
                    tokens up into a per-run bill, the raw material for the
                    enforce_budget cap and for deciding where routing pays off.
"""

from __future__ import annotations

from dataclasses import dataclass

from tracing import Run

# ---------------------------------------------------------------------------
# Model routing (Part 13, Section 3). Each stage gets the cheapest model that
# can hold its quality bar; the frontier model is reserved for synthesis.
# ---------------------------------------------------------------------------
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-8"

ROUTES: dict[str, tuple[str, str]] = {
    # stage        (model,  $ tier)
    "plan":        (HAIKU,  "$"),
    "worker":      (SONNET, "$$"),
    "synthesize":  (OPUS,   "$$$"),   # the step the user sees -> the frontier model
    "reflect":     (SONNET, "$$"),
    "judge":       (SONNET, "$$"),
}
_DEFAULT = (SONNET, "$$")


def route(stage: str) -> str:
    """The model to use for a pipeline stage. The orchestrator seams from Parts
    7/8 already support this — only the model id per call changes."""
    return ROUTES.get(stage, _DEFAULT)[0]


# ---------------------------------------------------------------------------
# Version pinning (Part 13, Section 6). A run is reproducible only if every
# moving part is pinned. A silent model upgrade is the classic overnight
# regression; the manifest is what a deploy gate diffs against.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VersionManifest:
    prompt_version: str
    tool_versions: dict[str, str]
    models: dict[str, str]


def pin_versions(prompt_version: str = "2026-06-17") -> VersionManifest:
    return VersionManifest(
        prompt_version=prompt_version,
        tool_versions={"search_papers": "v3", "fetch_paper": "v2", "save_to_file": "v1"},
        models={stage: model for stage, (model, _) in ROUTES.items()},
    )


# ---------------------------------------------------------------------------
# Cost rollup (Part 13, Section 2): the span tree is a bill. Approximate per-1k
# token prices, blended input+output, just to make the routing argument concrete.
# ---------------------------------------------------------------------------
PRICE_PER_1K = {HAIKU: 0.001, SONNET: 0.004, OPUS: 0.020}


def cost_rollup(run: Run, model: str = SONNET) -> dict:
    """Roll per-span tokens up into a per-run cost, grouped by span name. In a
    real deployment each span carries its own routed model; here we attribute a
    single rate to keep the stand-in dependency-free."""
    by_name: dict[str, int] = {}
    for s in run.spans:
        by_name[s.name] = by_name.get(s.name, 0) + s.tokens
    rate = PRICE_PER_1K.get(model, PRICE_PER_1K[SONNET])
    total_tokens = sum(by_name.values())
    lines = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "total_tokens": total_tokens,
        "est_cost_usd": round(total_tokens / 1000 * rate, 4),
        "by_span": [{"span": n, "tokens": t,
                     "usd": round(t / 1000 * rate, 4)} for n, t in lines],
    }


def routed_vs_flat(run: Run) -> dict:
    """Illustrate the routing win: one-model-everywhere (opus) vs the routed mix.
    The dominant synthesis span justifies the frontier model; the cheap stages
    should not pay for it."""
    flat = cost_rollup(run, model=OPUS)["est_cost_usd"]
    # Routed: attribute each span to its stage's model where we can name it.
    routed = 0.0
    for s in run.spans:
        stage = s.name.split(":")[0].replace("llm", "worker")
        model = route(stage)
        routed += s.tokens / 1000 * PRICE_PER_1K.get(model, PRICE_PER_1K[SONNET])
    return {"flat_opus_usd": round(flat, 4), "routed_usd": round(routed, 4)}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from tracing import TRACE_DIR

    paths = sorted(Path(TRACE_DIR).glob("run_*.json")) if TRACE_DIR.exists() else []
    if not paths:
        print("No traces yet. Run harness.py or orchestrator.py first.")
        sys.exit(0)
    run = Run.load(paths[-1])
    print("Version manifest:", pin_versions())
    print("\nCost rollup (span tree is a bill, Part 13):")
    roll = cost_rollup(run)
    print(f"  total: {roll['total_tokens']} tokens  ~${roll['est_cost_usd']}")
    for line in roll["by_span"][:6]:
        print(f"    {line['span']:<22} {line['tokens']:>7} tok  ${line['usd']}")
    print("\nRouted vs flat:", routed_vs_flat(run))
