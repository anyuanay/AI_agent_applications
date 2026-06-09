"""
Hooks — the harness enforcement layer (Part 8, hardened in Part 10).

A prompt rule is a wish; a hook is a guarantee. Hooks fire on lifecycle events
(pre/post tool use) regardless of whether the model cooperates, and a pre-tool
hook returns a *structured* Block (Part 5 error-contract discipline, now
enforced by the harness) so the model can reason about the refusal and correct.

  - guard_file_writes : containment — confine writes to output/ (works even when
                        the model is fully fooled by an injection, Part 10).
  - enforce_budget    : the runaway-loop / spend cap promised in Part 1.
  - log_call          : the post-tool seed that grows into tracing (Part 11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

OUTPUT_DIR = (Path(__file__).parent / "output").resolve()


# ---------------------------------------------------------------------------
# The hook result contract: Allow or a structured Block.
# ---------------------------------------------------------------------------
@dataclass
class HookResult:
    allowed: bool
    error: dict | None = None


def Allow() -> HookResult:
    return HookResult(allowed=True)


def Block(error: dict) -> HookResult:
    return HookResult(allowed=False, error=error)


# ---------------------------------------------------------------------------
# Pre-tool hooks
# ---------------------------------------------------------------------------
def guard_file_writes(tool: str, args: dict) -> HookResult:
    """Containment: writes are confined to output/, fully fooled or not."""
    if tool != "save_to_file":
        return Allow()
    filename = args.get("filename", "")
    # Resolve against output/ and reject anything that escapes it (.., abs paths).
    target = (OUTPUT_DIR / filename).resolve()
    if not str(target).startswith(str(OUTPUT_DIR)):
        return Block({
            "error": "path_denied",
            "reason": "writes are confined to output/",
        })
    return Allow()


class enforce_budget:
    """The Part 1 runaway-loop guard, made a hard limit in the harness.

    Stateful across a run: tracks cumulative tokens and blocks further tool
    calls once the cap is exceeded.
    """

    def __init__(self, max_tokens: int = 200_000) -> None:
        self.max_tokens = max_tokens
        self.spent = 0

    def add(self, tokens: int) -> None:
        self.spent += tokens

    def __call__(self, tool: str, args: dict) -> HookResult:
        if self.spent >= self.max_tokens:
            return Block({
                "error": "budget_exceeded",
                "reason": f"run exceeded {self.max_tokens} tokens; stopping",
            })
        return Allow()


# ---------------------------------------------------------------------------
# Post-tool hooks
# ---------------------------------------------------------------------------
def log_call(tool: str, args: dict, result: str) -> None:
    """The Part 11 seed: every call passes through here on the way out."""
    preview = result if len(result) < 120 else result[:120] + "…"
    print(f"  [hook] {tool}({list(args)}) -> {preview}")


# ---------------------------------------------------------------------------
# A small dispatcher the harness calls. Pre-hooks can block; post-hooks observe.
# ---------------------------------------------------------------------------
@dataclass
class HookSet:
    pre_tool_use: list = field(default_factory=list)
    post_tool_use: list = field(default_factory=list)

    def run_pre(self, tool: str, args: dict) -> HookResult:
        for hook in self.pre_tool_use:
            result = hook(tool, args)
            if not result.allowed:
                return result
        return Allow()

    def run_post(self, tool: str, args: dict, result: str) -> None:
        for hook in self.post_tool_use:
            hook(tool, args, result)
