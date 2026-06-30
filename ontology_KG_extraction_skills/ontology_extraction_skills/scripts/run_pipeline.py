#!/usr/bin/env python3
"""Drive the seven-stage ontology-extraction pipeline end to end.

Each stage is an independent skill with its own ``scripts/`` directory; this
driver chains them in order, computing the file handoff between stages so you do
not have to remember the cumulative filename suffixes. Every stage writes its
artifact *next to the source document*, which is also what lets Stages 1b, 2b and
3 auto-detect the Stage 0 scope as a sibling file.

The chain (source stem = SRC):

    0   extract_scope.py        SRC                          -> SRC_scope.json
    1   surface_candidates.py   SRC                          -> SRC_candidates.json
    1b  classify_candidates.py  SRC_candidates.json          -> SRC_candidates_gated.json
    2   name_vocabulary.py      SRC_candidates_gated.json    -> ..._gated_vocabulary.json
    2b  rank_salience.py        ..._gated_vocabulary.json    -> ..._vocabulary_salient.json
    3   synthesize_structure.py ..._vocabulary_salient.json  -> ..._salient_synth_structure.json
    4   review_ontology.py      ..._synth_structure.json     -> ..._reviewed.json  + ..._admitted.ttl

Each stage runs as a separate ``python`` subprocess, so the per-stage scripts'
bare sibling imports (and any same-named modules across stages) never collide.

LLM use: Stages 0, 1b and 4 take ``--llm`` (opt in); Stage 1 takes ``--no-llm``
and Stage 2 ``--no-embeddings`` (opt out); Stage 3 *requires* an LLM (Gemini).
Pass ``--no-llm`` to this driver to run every deterministic path; Stage 3 will
still try to call the model and needs a key. API keys come from a ``.env`` (see
``.env.example``); the driver passes ``--env`` to every stage that accepts it.

Usage:
    python run_pipeline.py SOURCE.md                 # full run, LLM on
    python run_pipeline.py SOURCE.md --no-llm        # deterministic stages only
    python run_pipeline.py SOURCE.md --from 1b --to 3
    python run_pipeline.py SOURCE.md --dry-run       # print commands, run nothing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# This driver lives in <skill-root>/scripts/; the stage directories and the
# shared .env sit one level up at the skill root.
_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Stage:
    sid: str                       # stage id, e.g. "1b"
    directory: str                 # stage skill folder
    script: str                    # entry script under <directory>/scripts/
    in_suffix: str | None          # input filename = <stem><in_suffix>; None = the source file
    out_name: str                  # human-facing description of what it writes
    llm_on: list[str] = field(default_factory=list)    # flags added when LLM is enabled
    llm_off: list[str] = field(default_factory=list)   # flags added when --no-llm
    env: bool = False              # accepts --env
    source_flag: bool = False      # accepts --source (pass the original document)
    needs_llm: bool = False        # cannot run deterministically
    extra: list[str] = field(default_factory=list)     # always-on flags

    def script_path(self) -> Path:
        return _ROOT / self.directory / "scripts" / self.script


# The pipeline, in order. Input suffixes are cumulative and match exactly what
# each stage script computes with `<stem>_<suffix>.json` (verified against the
# scripts), so the handoff is correct rather than assumed.
STAGES: list[Stage] = [
    Stage("0", "stage0-scope", "extract_scope.py",
          in_suffix=None, out_name="_scope.json",
          llm_on=["--llm"], env=True),
    Stage("1", "stage1-surface-candidates", "surface_candidates.py",
          in_suffix=None, out_name="_candidates.json",
          llm_off=["--no-llm"], env=True),
    Stage("1b", "stage1b-classify-candidates", "classify_candidates.py",
          in_suffix="_candidates.json", out_name="_candidates_gated.json",
          llm_on=["--llm"], env=True),
    Stage("2", "stage2-name-vocabulary", "name_vocabulary.py",
          in_suffix="_candidates_gated.json",
          out_name="_candidates_gated_vocabulary.json",
          llm_off=["--no-embeddings"], env=True, source_flag=True),
    Stage("2b", "stage2b-salience", "rank_salience.py",
          in_suffix="_candidates_gated_vocabulary.json",
          out_name="_candidates_gated_vocabulary_salient.json"),
    Stage("3", "stage3-synthesize-structure", "synthesize_structure.py",
          in_suffix="_candidates_gated_vocabulary_salient.json",
          out_name="_candidates_gated_vocabulary_salient_synth_structure.json",
          env=True, needs_llm=True),
    Stage("4", "stage4-review-ontology", "review_ontology.py",
          in_suffix="_candidates_gated_vocabulary_salient_synth_structure.json",
          out_name="_..._reviewed.json + _..._admitted.ttl",
          llm_on=["--llm"], env=True, source_flag=True, extra=["--emit-ttl"]),
]

_INDEX = {s.sid: i for i, s in enumerate(STAGES)}


def _resolve_env(source: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for cand in (_ROOT / ".env", source.parent / ".env"):
        if cand.exists():
            return cand
    return None


def _input_path(stage: Stage, source: Path) -> Path:
    if stage.in_suffix is None:
        return source
    return source.with_name(f"{source.stem}{stage.in_suffix}")


def _build_cmd(stage: Stage, source: Path, *, no_llm: bool, no_ttl: bool,
               env_path: Path | None) -> list[str]:
    cmd = [sys.executable, str(stage.script_path()), str(_input_path(stage, source))]
    cmd += stage.llm_off if no_llm else stage.llm_on
    for flag in stage.extra:
        if flag == "--emit-ttl" and no_ttl:
            continue
        cmd.append(flag)
    if stage.source_flag:
        cmd += ["--source", str(source)]
    if stage.env and env_path is not None:
        cmd += ["--env", str(env_path)]
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the seven-stage ontology-extraction pipeline.")
    ap.add_argument("source", type=Path, help="source document (text or markdown)")
    ap.add_argument("--from", dest="from_sid", default="0", metavar="STAGE",
                    help="first stage to run (0,1,1b,2,2b,3,4). Default 0.")
    ap.add_argument("--to", dest="to_sid", default="4", metavar="STAGE",
                    help="last stage to run. Default 4.")
    ap.add_argument("--no-llm", action="store_true",
                    help="use every deterministic path (Stage 3 still needs an LLM).")
    ap.add_argument("--no-ttl", action="store_true",
                    help="do not emit the admitted Turtle in Stage 4.")
    ap.add_argument("--env", default=None,
                    help="path to a .env with API keys (default: auto-detect).")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them.")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue to the next stage even if one fails.")
    args = ap.parse_args(argv)

    source: Path = args.source
    if not args.dry_run and not source.exists():
        ap.error(f"source not found: {source}")
    if args.from_sid not in _INDEX:
        ap.error(f"unknown --from stage {args.from_sid!r}; choose from {list(_INDEX)}")
    if args.to_sid not in _INDEX:
        ap.error(f"unknown --to stage {args.to_sid!r}; choose from {list(_INDEX)}")
    lo, hi = _INDEX[args.from_sid], _INDEX[args.to_sid]
    if lo > hi:
        ap.error(f"--from {args.from_sid} comes after --to {args.to_sid}")

    env_path = _resolve_env(source, args.env)
    selected = STAGES[lo:hi + 1]

    print(f"Ontology-extraction pipeline")
    print(f"  source : {source}")
    print(f"  stages : {' -> '.join(s.sid for s in selected)}")
    print(f"  mode   : {'deterministic (--no-llm)' if args.no_llm else 'LLM-assisted'}")
    print(f"  .env   : {env_path if env_path else '(none found; LLM stages may fail)'}")
    print("=" * 72)

    for stage in selected:
        in_path = _input_path(stage, source)
        if stage.in_suffix is not None and not args.dry_run and not in_path.exists():
            print(f"[stage {stage.sid}] MISSING INPUT: {in_path}")
            print(f"            run the earlier stages first (or --from a later stage "
                  f"once the artifact exists).")
            return 2
        if stage.needs_llm and args.no_llm:
            print(f"[stage {stage.sid}] note: this stage requires an LLM (Gemini); "
                  f"--no-llm cannot make it deterministic. Attempting anyway.")
        cmd = _build_cmd(stage, source, no_llm=args.no_llm, no_ttl=args.no_ttl,
                         env_path=env_path)
        print(f"[stage {stage.sid}] {stage.directory}  ->  writes {stage.out_name}")
        print("            " + " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[stage {stage.sid}] FAILED (exit {result.returncode}).")
            if not args.keep_going:
                return result.returncode
        print("-" * 72)

    if not args.dry_run:
        final = source.with_name(
            f"{source.stem}_candidates_gated_vocabulary_salient_synth_structure_admitted.ttl")
        if hi == _INDEX["4"] and not args.no_ttl and final.exists():
            print(f"Done. Admitted ontology: {final}")
        else:
            print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
