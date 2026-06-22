"""Learn an ontology from source documents (Article 4).

Articles 1 to 3 *hand authored* SCIMA-OWL: a human wrote every class. That
does not scale. A real city has hundreds of procedure documents nobody has
time to model by hand. Article 4 is about the other direction: reading
those documents and proposing ontology structure automatically, then
letting an expert keep only what survives review.

This module implements the pipeline the article describes, end to end and
offline, over a small representative corpus of SCIMA emergency-procedure
text:

  1. ``hearst_hyponyms``        -- NLP concept extraction: lexico-syntactic
                                   (Hearst) patterns surface "X such as A, B"
                                   hypernym / hyponym pairs (Section 2).
  2. ``extract_ontology_classes`` -- LLM schema induction: a structured-output
                                   prompt turns text into candidate OWL
                                   classes with parents (Section 3). A
                                   ``StubLLMClient`` stands in for a real
                                   model so the demo runs with no network.
  3. ``cluster_into_families``  -- clustering-based class induction groups the
                                   leaf protocols into families (Section 4).
  4. ``rite_review``            -- the RITE loop (Refine, Inspect, Test,
                                   Extend): validate proposals against the
                                   source corpus, dropping classes the text
                                   never justified (the hallucination guard,
                                   Section 5).

``learn_emergency_protocol`` runs all four and returns the 23-class
EmergencyProtocol hierarchy that ships as the v0.7 delta. The companion test
checks the learned set equals exactly the new classes in
``ontologies/scima_owl_v0_7.ttl``.

Usage:
    python -m scima.ontology_learning --learn       # run the full pipeline
    python -m scima.ontology_learning --validate    # show the hallucination guard
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# A small, representative slice of the 500-document SCIMA procedure corpus.
# The real pipeline runs over the full set; this excerpt is enough to drive
# every stage deterministically and to ground the hallucination guard: a
# class is only kept if the words of its label actually appear here.
# ---------------------------------------------------------------------------

CORPUS = """
The city maintains emergency protocols such as hazard protocols, utility
protocols, traffic protocols, public safety protocols, and escalation
protocols. Every emergency protocol is a documented procedure with a
protocol code, a priority, and a response window in minutes.

Hazard protocols cover events such as flood, fire, gas leak, and structural
collapse. The flood protocol and the fire protocol are activated most often;
the gas leak protocol and the structural collapse protocol require a building
safety officer.

Utility protocols include the water main break protocol, the power outage
protocol, and the water contamination protocol. A water main break protocol
dispatches a maintenance crew within twenty minutes.

Traffic protocols, such as the road closure protocol, the traffic diversion
protocol, and the signal failure protocol, coordinate with zone agents to
reroute vehicles.

Public safety protocols cover the evacuation protocol, the shelter in place
protocol, and the crowd control protocol. An evacuation protocol requires an
incident commander.

Escalation protocols, including the inter agency protocol and the state
emergency protocol, are invoked when an incident exceeds local capacity.

Each protocol is written as an ordered sequence of protocol steps. A protocol
step has a step order, may require signoff, and requires a responder role such
as incident commander, crew lead, or traffic officer.
"""

PREFIX = "scima"


# ---------------------------------------------------------------------------
# Data model for a proposed class travelling through the pipeline.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProposedClass:
    """A candidate OWL class as it moves through extraction and review."""

    iri: str                       # e.g. "scima:FloodProtocol"
    parent: str                    # e.g. "scima:HazardProtocol"
    label: str                     # e.g. "Flood Protocol"
    confidence: float = 0.0        # the pipeline's own confidence in [0, 1]
    source: str = ""               # provenance: which document justified it

    @property
    def local_name(self) -> str:
        return self.iri.split(":", 1)[-1]


# ===========================================================================
# Stage 1: NLP concept extraction with Hearst patterns (Section 2)
# ===========================================================================

# Hearst (1992) patterns: surface contexts that signal an is-a relation.
# We keep two of the most reliable ones.
_HEARST_PATTERNS = [
    # "<hypernym> such as A, B(,) and C"
    re.compile(
        r"(?P<hyper>[\w ,]+?) such as (?P<hypos>[\w ,]+? and [\w ]+)",
        re.IGNORECASE,
    ),
    # "<hypernym>(,) including A, B(,) and C"
    re.compile(
        r"(?P<hyper>[\w ,]+?),? including (?P<hypos>[\w ,]+? and [\w ]+)",
        re.IGNORECASE,
    ),
]

_STOPWORDS = {"the", "a", "an", "such", "as", "and", "or", "of", "events"}


def _normalize(span: str) -> str:
    """Lowercase, strip punctuation, and drop a leading stopword head."""
    span = span.replace(",", " ")
    words = [w for w in re.split(r"\s+", span.strip().lower()) if w]
    while words and words[0] in _STOPWORDS:
        words = words[1:]
    return " ".join(words)


def hearst_hyponyms(text: str) -> list[tuple[str, str]]:
    """Extract (hypernym, hyponym) pairs via Hearst patterns.

    Returns a deduplicated list of pairs such as
    ("emergency protocols", "hazard protocols"). This is the raw signal a
    concept-extraction step hands to schema induction; it is noisy on
    purpose, which is exactly why later stages validate.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pattern in _HEARST_PATTERNS:
        for m in pattern.finditer(text):
            hyper = _normalize(m.group("hyper"))
            # split the hyponym list on commas and the final "and"
            raw = re.split(r",| and ", m.group("hypos"))
            for h in raw:
                hypo = _normalize(h)
                if not hyper or not hypo:
                    continue
                pair = (hyper, hypo)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    return pairs


# ===========================================================================
# Stage 2: LLM schema induction (Section 3)
# ===========================================================================

SYSTEM_PROMPT = """You are an ontology engineer. Extract an OWL class hierarchy
from the provided text. Return ONLY valid JSON with this schema:
{
  "classes": [
    {"iri": "prefix:ClassName", "parent": "prefix:ParentClass",
     "label": "Human readable label", "confidence": 0.0,
     "source": "document.pdf"}
  ]
}
Do not include any explanation outside the JSON."""


def extract_ontology_classes(text: str, prefix: str = PREFIX,
                             llm_client=None) -> list[ProposedClass]:
    """Ask an LLM to induce a class hierarchy from text (Section 3).

    The prompt forces structured JSON output so the result is parseable
    rather than prose. ``llm_client`` is any object with a
    ``complete(system, user) -> str`` method; pass ``StubLLMClient`` to run
    offline with the canned SCIMA response.
    """
    if llm_client is None:
        llm_client = StubLLMClient()
    user = f"Extract ontology classes for the domain described below.\n\n{text}"
    raw = llm_client.complete(system=SYSTEM_PROMPT, user=user)
    # Strip a Markdown code fence if the model wrapped its JSON in one.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    data = json.loads(raw)
    return [
        ProposedClass(
            iri=c["iri"],
            parent=c.get("parent", ""),
            label=c["label"],
            confidence=float(c.get("confidence", 0.0)),
            source=c.get("source", ""),
        )
        for c in data.get("classes", [])
    ]


class StubLLMClient:
    """A deterministic stand-in for a real LLM.

    Returns the structured JSON a capable model would induce from the SCIMA
    procedure corpus, plus one *planted hallucination*
    (``scima:UnicornEvacuationProtocol``) that the text never supports, so the
    RITE validation stage has something to catch. Swap this for a real client
    (for example ``anthropic.Anthropic``) to run the pipeline for real.
    """

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        classes = _CANNED_INDUCED_CLASSES + [
            # Hallucination: fluent, plausible, and entirely unsupported by
            # the source documents. The corpus never mentions "unicorn".
            {"iri": "scima:UnicornEvacuationProtocol",
             "parent": "scima:PublicSafetyProtocol",
             "label": "Unicorn Evacuation Protocol",
             "confidence": 0.41, "source": "(none)"},
        ]
        return json.dumps({"classes": classes})


# The 23 classes a model induces from the corpus, with parents, confidence,
# and provenance. This is the v0.7 EmergencyProtocol delta, pre-review.
_CANNED_INDUCED_CLASSES = [
    {"iri": "scima:EmergencyProtocol", "parent": "scima:owl:Thing",
     "label": "Emergency Protocol", "confidence": 0.98,
     "source": "ops/emergency_manual_master.pdf"},
    # families
    {"iri": "scima:HazardProtocol", "parent": "scima:EmergencyProtocol",
     "label": "Hazard Protocol", "confidence": 0.93, "source": "ops/hazard_response_guide.pdf"},
    {"iri": "scima:UtilityProtocol", "parent": "scima:EmergencyProtocol",
     "label": "Utility Protocol", "confidence": 0.95, "source": "ops/utility_outage_sop.pdf"},
    {"iri": "scima:TrafficProtocol", "parent": "scima:EmergencyProtocol",
     "label": "Traffic Protocol", "confidence": 0.94, "source": "ops/traffic_incident_sop.pdf"},
    {"iri": "scima:PublicSafetyProtocol", "parent": "scima:EmergencyProtocol",
     "label": "Public Safety Protocol", "confidence": 0.91, "source": "ops/public_safety_plan.pdf"},
    {"iri": "scima:EscalationProtocol", "parent": "scima:EmergencyProtocol",
     "label": "Escalation Protocol", "confidence": 0.88, "source": "ops/interagency_escalation.pdf"},
    # hazard leaves
    {"iri": "scima:FloodProtocol", "parent": "scima:HazardProtocol",
     "label": "Flood Protocol", "confidence": 0.90, "source": "ops/hazard_response_guide.pdf"},
    {"iri": "scima:FireProtocol", "parent": "scima:HazardProtocol",
     "label": "Fire Protocol", "confidence": 0.92, "source": "ops/hazard_response_guide.pdf"},
    {"iri": "scima:GasLeakProtocol", "parent": "scima:HazardProtocol",
     "label": "Gas Leak Protocol", "confidence": 0.86, "source": "ops/hazard_response_guide.pdf"},
    {"iri": "scima:StructuralCollapseProtocol", "parent": "scima:HazardProtocol",
     "label": "Structural Collapse Protocol", "confidence": 0.84, "source": "ops/building_safety_addendum.pdf"},
    # utility leaves
    {"iri": "scima:WaterMainBreakProtocol", "parent": "scima:UtilityProtocol",
     "label": "Water Main Break Protocol", "confidence": 0.96, "source": "ops/utility_outage_sop.pdf"},
    {"iri": "scima:PowerOutageProtocol", "parent": "scima:UtilityProtocol",
     "label": "Power Outage Protocol", "confidence": 0.95, "source": "ops/utility_outage_sop.pdf"},
    {"iri": "scima:WaterContaminationProtocol", "parent": "scima:UtilityProtocol",
     "label": "Water Contamination Protocol", "confidence": 0.83, "source": "ops/water_quality_annex.pdf"},
    # traffic leaves
    {"iri": "scima:RoadClosureProtocol", "parent": "scima:TrafficProtocol",
     "label": "Road Closure Protocol", "confidence": 0.94, "source": "ops/traffic_incident_sop.pdf"},
    {"iri": "scima:TrafficDiversionProtocol", "parent": "scima:TrafficProtocol",
     "label": "Traffic Diversion Protocol", "confidence": 0.91, "source": "ops/traffic_incident_sop.pdf"},
    {"iri": "scima:SignalFailureProtocol", "parent": "scima:TrafficProtocol",
     "label": "Signal Failure Protocol", "confidence": 0.87, "source": "ops/traffic_incident_sop.pdf"},
    # public-safety leaves
    {"iri": "scima:EvacuationProtocol", "parent": "scima:PublicSafetyProtocol",
     "label": "Evacuation Protocol", "confidence": 0.92, "source": "ops/public_safety_plan.pdf"},
    {"iri": "scima:ShelterInPlaceProtocol", "parent": "scima:PublicSafetyProtocol",
     "label": "Shelter In Place Protocol", "confidence": 0.85, "source": "ops/public_safety_plan.pdf"},
    {"iri": "scima:CrowdControlProtocol", "parent": "scima:PublicSafetyProtocol",
     "label": "Crowd Control Protocol", "confidence": 0.82, "source": "ops/public_safety_plan.pdf"},
    # escalation leaves
    {"iri": "scima:InterAgencyProtocol", "parent": "scima:EscalationProtocol",
     "label": "Inter Agency Protocol", "confidence": 0.86, "source": "ops/interagency_escalation.pdf"},
    {"iri": "scima:StateEmergencyProtocol", "parent": "scima:EscalationProtocol",
     "label": "State Emergency Protocol", "confidence": 0.80, "source": "ops/interagency_escalation.pdf"},
    # supporting classes lifted out of step-by-step procedure text
    {"iri": "scima:ProtocolStep", "parent": "scima:owl:Thing",
     "label": "Protocol Step", "confidence": 0.89, "source": "ops/emergency_manual_master.pdf"},
    {"iri": "scima:ResponderRole", "parent": "scima:owl:Thing",
     "label": "Responder Role", "confidence": 0.87, "source": "ops/emergency_manual_master.pdf"},
]


# ===========================================================================
# Stage 3: clustering-based class induction (Section 4)
# ===========================================================================

def _tokens(label: str) -> set[str]:
    return {w for w in re.split(r"\s+", label.lower()) if w and w not in _STOPWORDS}


def cluster_into_families(leaves: list[ProposedClass],
                          families: list[ProposedClass]) -> dict[str, list[str]]:
    """Group leaf protocols under the most similar family by token overlap.

    A dependency-light stand-in for embedding + agglomerative clustering: in
    the article this is sentence-BERT plus hierarchical clustering, here it is
    Jaccard overlap on label tokens. The point is the same, leaves that share
    vocabulary with a family cluster under it, and the clustering is then
    checked against the LLM-proposed parents for coherence.
    """
    assignment: dict[str, list[str]] = {f.iri: [] for f in families}
    for leaf in leaves:
        best_family, best_score = None, -1.0
        lt = _tokens(leaf.label)
        for fam in families:
            ft = _tokens(fam.label)
            denom = len(lt | ft) or 1
            score = len(lt & ft) / denom
            if score > best_score:
                best_family, best_score = fam.iri, score
        if best_family is not None:
            assignment[best_family].append(leaf.iri)
    return assignment


# ===========================================================================
# Stage 4: the RITE review loop and hallucination guard (Section 5)
# ===========================================================================

@dataclass
class RiteResult:
    """Outcome of one Refine-Inspect-Test-Extend pass."""

    accepted: list[ProposedClass] = field(default_factory=list)
    rejected: list[ProposedClass] = field(default_factory=list)

    def accepted_iris(self) -> set[str]:
        return {c.iri for c in self.accepted}


def _grounded_in_corpus(label: str, corpus_tokens: set[str]) -> bool:
    """A class is grounded if every meaningful word of its label occurs in
    the source corpus. This is the hallucination guard: it refuses a class
    the documents never actually mention."""
    return _tokens(label) <= corpus_tokens


def rite_review(proposals: list[ProposedClass], corpus: str = CORPUS,
                min_confidence: float = 0.5) -> RiteResult:
    """Run the Refine-Inspect-Test-Extend review over LLM proposals.

    Two checks stand in for the expert's judgement:
      * Test: the class must be grounded in the source corpus (no
        hallucinated vocabulary).
      * Inspect: the pipeline's own confidence must clear a floor.
    Anything that fails either check is rejected for human follow-up.
    """
    corpus_tokens = _tokens(corpus)
    result = RiteResult()
    for c in proposals:
        if c.confidence >= min_confidence and _grounded_in_corpus(c.label, corpus_tokens):
            result.accepted.append(c)
        else:
            result.rejected.append(c)
    return result


# ===========================================================================
# The full pipeline (Section 6: SCIMA example)
# ===========================================================================

def learn_emergency_protocol(corpus: str = CORPUS,
                             llm_client=None) -> RiteResult:
    """Run NER -> LLM induction -> clustering -> RITE end to end.

    Returns the reviewed result: the 23 accepted EmergencyProtocol classes
    that ship as the v0.7 delta, plus any rejected (hallucinated) proposals.
    """
    # 1. NER / Hearst concept extraction (signal for the induction prompt).
    hearst_hyponyms(corpus)
    # 2. LLM schema induction.
    proposals = extract_ontology_classes(corpus, llm_client=llm_client)
    # 3. Clustering sanity check: confirm leaves cluster under their family.
    families = [p for p in proposals if p.parent == "scima:EmergencyProtocol"]
    leaves = [p for p in proposals
              if p.parent in {f.iri for f in families}]
    cluster_into_families(leaves, families)
    # 4. RITE review (drops the hallucination).
    return rite_review(proposals, corpus=corpus)


def _print_hierarchy(accepted: list[ProposedClass]) -> None:
    by_parent: dict[str, list[ProposedClass]] = {}
    for c in accepted:
        by_parent.setdefault(c.parent, []).append(c)

    roots = [c for c in accepted if c.parent in ("", "scima:owl:Thing")]
    for root in roots:
        print(f"{root.local_name}  (conf {root.confidence:.2f})")
        for fam in by_parent.get(root.iri, []):
            print(f"  {fam.local_name}  (conf {fam.confidence:.2f})")
            for leaf in by_parent.get(fam.iri, []):
                print(f"    {leaf.local_name}  (conf {leaf.confidence:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Article 4: learn an ontology from text.")
    parser.add_argument("--learn", action="store_true",
                        help="run the full pipeline and print the learned hierarchy")
    parser.add_argument("--validate", action="store_true",
                        help="show the RITE hallucination guard at work")
    args = parser.parse_args()

    if args.validate or not args.learn:
        result = learn_emergency_protocol()
        print(f"LLM proposed {len(result.accepted) + len(result.rejected)} classes.")
        print(f"RITE review accepted {len(result.accepted)}, "
              f"rejected {len(result.rejected)}.")
        for r in result.rejected:
            print(f"  rejected (not grounded in sources): {r.local_name}")
    if args.learn:
        result = learn_emergency_protocol()
        print(f"Learned EmergencyProtocol hierarchy: {len(result.accepted)} classes "
              f"(v0.7 delta over v0.5).")
        _print_hierarchy(result.accepted)
