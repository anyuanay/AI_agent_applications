"""Offline LLM client for `induce-schema`.

Re-exports the deterministic ``StubLLMClient`` from the backing package and shows
the interface a real client must satisfy. Any object with a
``complete(system, user) -> str`` method works as a drop-in.

The stub returns the structured JSON a capable model would induce from the SCIMA
procedure corpus, plus one planted hallucination
(``scima:UnicornEvacuationProtocol``) so the RITE guard in `refine-ontology` has
something to catch.

Usage:
    python scripts/stub_llm_client.py     # print the raw JSON the stub returns
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(_REPO_ROOT / "ontology_kg_for_agents"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scima.ontology_learning import SYSTEM_PROMPT, StubLLMClient  # noqa: E402


class RealClientTemplate:
    """Skeleton for a production client. Fill in `complete` with a real call.

    Example with the Anthropic SDK:

        import anthropic

        class AnthropicClient:
            def __init__(self, model="claude-opus-4-8"):
                self._client = anthropic.Anthropic()
                self._model = model

            def complete(self, system: str, user: str) -> str:
                msg = self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text
    """

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        raise NotImplementedError("Wire this to a real model client.")


def main() -> int:
    stub = StubLLMClient()
    print(stub.complete(system=SYSTEM_PROMPT, user="(demo)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
