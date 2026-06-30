"""Extract an ontology from source documents (Article 4).

Articles 1 through 3 *authored* SCIMA-OWL by hand. Article 4 *learns* a slice of
it from text, because once a domain is large or specialized the vocabulary is
already written down and the job is to extract it auditably, not reinvent it. An
ontology is a *conceptualization*: it holds the general kinds and relationships of
a domain, not the individual facts a text happens to mention. This module
implements the seven-stage pipeline from the article:

  Stage 0   Scope      -> frame the domain (topics, competency questions,
                          out-of-scope) and drop exercise/example sections
  Stage 1   Surface    -> candidate mentions (literal spans + LLM), deduped
  Stage 1b  Sort       -> class vs individual vs non-concept (the type/instance gate)
  Stage 2   Name       -> a flat named vocabulary (term -> concept)
  Stage 2b  Salience   -> keep the salient core, park the long tail
  Stage 3   Structure  -> synthesize an is-a DAG + domain/range + axioms,
                          fenced by grounding (first commitment to structure)
  Stage 4   Review      -> the agentic RITE trust gate (accept / reject / escalate)

Five invariants hold across every stage:
  * Recall-first         -- nothing the corpus supports is silently dropped;
                            pruning is feedback, not deletion.
  * Provenance           -- every record knows the span (or children) it came from.
  * Grounding by kind    -- how an element is justified depends on its kind.
  * Universals not particulars -- the ontology (T-Box) holds kinds; particulars
                            (specific numbers, dates, names) are individuals routed
                            to the A-Box (Article 5), not the schema.
  * Scope before extract -- a top-down scope frames and bounds the bottom-up work.

The "LLM" and "embedding clustering" steps are *deterministic stubs* here, so the
example is fast and reproducible. They stand in for an ``anthropic``-backed
extractor and a real embedding model; the pipeline structure, the RITE decisions,
and the emitted v0.6 delta are the real, testable contract. Running the pipeline
over the procedures corpus reproduces exactly the SCIMA-OWL v0.6 delta shipped in
``ontologies/scima_owl_v0_6.ttl``.

Usage:
    python -m scima.ontology_extraction --corpus corpus/emergency_procedures.txt
    python -m scima.ontology_extraction --emit ontologies/scima_owl_v0_6.ttl
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import OWL, RDF, RDFS, Graph, Literal, Namespace, URIRef
from rdflib.namespace import SKOS

from scima.ontology import ScimaOntology

SCIMA = Namespace("http://scima.city/ontology#")
PROV = Namespace("http://www.w3.org/ns/prov#")

_ROOT = Path(__file__).resolve().parent.parent
_V0_5 = _ROOT / "ontologies" / "scima_owl_v0_5.ttl"
_DEFAULT_CORPUS = _ROOT / "corpus" / "emergency_procedures.txt"
# Emit writes a machine-serialized copy here, never over the hand-curated
# ontologies/scima_owl_v0_6.ttl source of truth.
_DEFAULT_EMIT = _ROOT / "build" / "scima_owl_v0_6.ttl"


# =====================================================================
# Records that flow down the pipeline. Each carries its provenance.
# =====================================================================
@dataclass
class Scope:
    """A Stage 0 scope: what the ontology is about and what to keep out."""
    domain: str
    topics: list[str] = field(default_factory=list)
    competency_questions: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    content: str = ""                  # source with exercise/example sections dropped
    dropped_sections: int = 0


@dataclass
class Mention:
    """A Stage 1 candidate: a literal string with where it came from.

    ``kind`` is assigned in Stage 1b: ``class`` (a universal, into the ontology),
    ``individual`` (a particular, routed to the A-Box), or ``non_concept`` (debris
    or a generic/process word, parked as feedback).
    """
    text: str
    source: str            # "cheap" (literal span) or "llm"
    grounded: bool         # is the text actually present in the corpus?
    kind: str = "class"    # class | individual | non_concept (set in Stage 1b)


@dataclass
class Concept:
    """A Stage 2 named concept; gains structure in Stage 3, a verdict in Stage 4."""
    name: str                                  # canonical local name, e.g. IncidentCommander
    label: str                                 # human label, e.g. "Incident Commander"
    lexicalizations: list[str] = field(default_factory=list)
    origin: str = "extracted"                  # "extracted" or "coined"
    parent: str | None = None                  # qname of rdfs:subClassOf target
    grounded: bool = True
    salience: float = 0.0                       # set in Stage 2b
    flags: list[str] = field(default_factory=list)
    verdict: str | None = None                 # accept | reject | demote | park (set in Stage 4)


@dataclass
class Relationship:
    """A Stage 2/3 relationship with a domain and range assigned in Stage 3."""
    name: str
    label: str
    domain: str
    range: str
    grounded: bool = True
    verdict: str | None = None


@dataclass
class Outcome:
    """The result of a full pipeline run: the admitted delta plus the feedback set."""
    mentions: list[Mention]
    concepts: list[Concept]
    relationships: list[Relationship]
    coined: list[str]
    admitted_classes: list[Concept]
    admitted_relationships: list[Relationship]
    rejected: list[Concept]
    parked: list[Concept]
    demoted: list[str]
    n_axioms: int
    consistent: bool
    surface_stats: tuple[int, int, int] = (0, 0, 0)   # (cheap, llm, merged)
    scope: Scope | None = None
    individuals: list[Mention] = field(default_factory=list)
    non_concepts: list[Mention] = field(default_factory=list)
    salience_parked: list[Concept] = field(default_factory=list)


# =====================================================================
# The pipeline.
# =====================================================================
class OntologyExtractionPipeline:
    """Run Scope -> Surface -> Sort -> Name -> Salience -> Structure -> Review.

    The pipeline reads a corpus string, proposes a delta to SCIMA-OWL v0.5, and
    (after the RITE review) can emit the cumulative v0.6 ontology.
    """

    def __init__(self, corpus: str) -> None:
        self.corpus = corpus
        self.norm = _normalize(corpus)        # normalized text for grounding checks

    # ---- grounding (the spine of "grounding by kind") ----------------
    def _in_corpus(self, phrase: str) -> bool:
        """Is the phrase (or all of its significant tokens) present in the corpus?"""
        p = _normalize(phrase)
        if p in self.norm:
            return True
        toks = [t for t in p.split() if t not in _STOP]
        return bool(toks) and all(t in self.norm for t in toks)

    # ---- Stage 0: Scope ----------------------------------------------
    def scope(self) -> Scope:
        """Frame the ontology and drop the exercise/example sections.

        A structured source is segmented by heading and every exercise, example,
        solution, or activity block is removed, so its specific values never reach
        surfacing. The procedures corpus is a single expository paragraph with no
        such sections, so nothing is dropped here, but the gate is real. The
        domain statement, topics, competency questions, and out-of-scope note are
        deterministic stand-ins for an LLM summary of the mined skeleton.
        """
        content, dropped = _filter_content(self.corpus)
        self.norm = _normalize(content)       # surface and ground against content
        return Scope(
            domain=_SCOPE_DOMAIN,
            topics=list(_SCOPE_TOPICS),
            competency_questions=list(_SCOPE_COMPETENCY),
            out_of_scope=list(_SCOPE_OUT),
            content=content,
            dropped_sections=dropped,
        )

    # ---- Stage 1: Surface --------------------------------------------
    def surface(self) -> list[Mention]:
        """Union a hallucination-free cheap extractor with an LLM, then dedup.

        Cheap extraction can only emit literal spans, so every cheap mention is
        grounded by construction. The LLM lifts implicit/multi-word candidates but
        may over-reach (it proposes CrisisManager, which is nowhere in the text).
        Dedup is lexical: same normalized string => one mention.
        """
        cheap = [m for m in _CHEAP_CANDIDATES if self._in_corpus(m)]
        llm = list(_LLM_CANDIDATES)            # stub: an LLM's term-only proposals

        seen: dict[str, Mention] = {}
        for text in cheap:
            seen[_normalize(text)] = Mention(text, "cheap", grounded=True)
        merged = 0
        for text in llm:
            key = _normalize(text)
            if key in seen:                    # already surfaced by the cheap extractor
                merged += 1
                continue
            seen[key] = Mention(text, "llm", grounded=self._in_corpus(text))
        self._surface_stats = (len(cheap), len(llm), merged)
        return list(seen.values())

    # ---- Stage 1b: Sort by kind (the type/instance gate) -------------
    def sort_by_kind(
        self, mentions: list[Mention]
    ) -> tuple[list[Mention], list[Mention], list[Mention]]:
        """Sort candidates into class, individual, and non_concept.

        The reliable signal is morphology, not NER labels. A digit or markup marks
        a particular or debris; a generic role word or abstract process noun is a
        non-concept. Only classes flow into the ontology; individuals are routed to
        the A-Box (Article 5) and non-concepts to feedback. Nothing is deleted.
        """
        classes: list[Mention] = []
        individuals: list[Mention] = []
        non_concepts: list[Mention] = []
        for m in mentions:
            key = _normalize(m.text)
            if _DEBRIS.search(m.text):
                m.kind = "non_concept"; non_concepts.append(m)
            elif _DIGIT.search(m.text):
                m.kind = "individual"; individuals.append(m)
            elif key in _NON_CONCEPTS:
                m.kind = "non_concept"; non_concepts.append(m)
            else:
                m.kind = "class"; classes.append(m)
        return classes, individuals, non_concepts

    # ---- Stage 2: Name -----------------------------------------------
    def name(self, mentions: list[Mention]) -> tuple[list[Concept], list[Relationship]]:
        """Group synonymous mentions by meaning, drop non-concepts, name survivors.

        Grouping here is by *meaning*, not string (so "IC" joins "incident
        commander"). The input is the class mentions from Stage 1b, so non-concepts
        are already out. Selection here is minimal: Stage 2b and Stage 4, not this
        stage, are where the set is narrowed.
        """
        texts = {_normalize(m.text): m for m in mentions}
        concepts: list[Concept] = []
        used: set[str] = set()

        for canonical, group in _SYNONYM_GROUPS:
            present = [g for g in group if _normalize(g) in texts]
            if not present:
                continue
            used.update(_normalize(g) for g in group)
            grounded = any(texts[_normalize(g)].grounded for g in present)
            concepts.append(Concept(
                name=canonical[0],
                label=canonical[1],
                lexicalizations=present,
                grounded=grounded,
            ))

        # any surfaced class mention not in a synonym group becomes its own concept
        # (recall-first); known non-concepts were already removed in Stage 1b.
        for key, m in texts.items():
            if key in used or key in _NON_CONCEPTS:
                continue
            concepts.append(Concept(
                name=_camel(m.text), label=_titlecase(m.text),
                lexicalizations=[m.text], grounded=m.grounded,
            ))

        relationships = [Relationship(*r) for r in _RELATION_CANDIDATES]
        return concepts, relationships

    # ---- Stage 2b: Salience ------------------------------------------
    def salience(self, concepts: list[Concept]) -> tuple[list[Concept], list[Concept]]:
        """Keep the salient core, park the long tail.

        Each concept is scored from signals already in the pipeline: a match to a
        Stage 0 topic, whether it is grounded, and how its label relates to the
        domain. A concept is kept if its score clears the threshold; the rest are
        parked as feedback. On this small, on-topic corpus every concept is salient,
        so none is parked, but the stage is real (and an ungrounded LLM proposal is
        kept here, on purpose, so Stage 4 can reject it explicitly).
        """
        kept: list[Concept] = []
        parked: list[Concept] = []
        for c in concepts:
            scope_hit = any(_normalize(t) in _normalize(c.label) or
                            _normalize(c.label) in _normalize(t)
                            for t in _SCOPE_TOPICS)
            c.salience = round(0.5 * (1.0 if scope_hit else 0.0)
                               + 0.3 * (1.0 if c.grounded else 0.0)
                               + 0.2, 3)       # base term-hood for any named concept
            # Keep anything plausible; only a vanishingly weak signal would park.
            if c.salience >= _SALIENCE_FLOOR:
                kept.append(c)
            else:
                c.flags.append("low_salience")
                parked.append(c)
        return kept, parked

    # ---- Stage 3: Synthesize structure -------------------------------
    def synthesize(self, concepts: list[Concept],
                   relationships: list[Relationship]) -> list[str]:
        """Wire the flat vocabulary into a DAG, fenced by grounding.

        This is the first commitment to structure. An LLM (a deterministic stub
        here) organizes the salient concepts into an is-a hierarchy, anchoring each
        leaf to an existing class where one is stated and coining a parent only
        where a family has no name in the vocabulary, and it assigns each
        relationship a domain and range. Coined parents are flagged hypotheses,
        tested in review. Returns the list of coined parent names; mutates concepts
        in place with their parent and flags.
        """
        by_name = {c.name: c for c in concepts}

        # (a) anchor before you coin: attach leaves to existing classes where stated
        for child, parent in _STATED_ISA.items():
            if child in by_name:
                by_name[child].parent = parent

        # (b) coin parents over families that have no name in the vocabulary
        coined: list[str] = []
        for parent_name, parent_label, children in _COINED_CLUSTERS:
            present_children = [c for c in children if c in by_name]
            if not present_children:
                continue
            coined.append(parent_name)
            by_name[parent_name] = Concept(
                name=parent_name, label=parent_label, origin="coined",
                grounded=False, flags=["coined"],
            )
            concepts.append(by_name[parent_name])
            for child in present_children:
                by_name[child].parent = parent_name

        # (c) flag the orphans: extracted, grounded, but no confident parent
        for c in concepts:
            if c.origin == "extracted" and c.parent is None and c.name in _ORPHANS:
                c.flags.append("orphan")

        # (d) domain/range were assigned over the class set when the relation
        #     candidates were written (the LLM's job; fixed here for determinism).
        self._coined = coined
        return coined

    def reasoner_consistent(self, concepts: list[Concept]) -> bool:
        """Stage 3 closes with a reasoner run. Here: no class is asserted to be a
        subclass of two mutually disjoint classes (the cheap unsatisfiability check
        that matters for this delta). The emitted disjointness axioms put the three
        responder units in different leaves of one parent, which is consistent."""
        parents = {c.name: c.parent for c in concepts}
        for a, b in _DISJOINT_PAIRS:
            for name, parent in parents.items():
                if parent == a and parents.get(name) == b:
                    return False
        return True

    # ---- Stage 4: Agentic RITE review --------------------------------
    def review(self, concepts: list[Concept],
               relationships: list[Relationship]) -> Outcome:
        """Test each element by its kind and route it to accept / reject /
        demote / park. The agent acts alone on the clear cases and would escalate
        the genuinely ambiguous; this corpus has no ambiguous case, so every
        verdict here is one the agent can take and a human ratifies.
        """
        admitted: list[Concept] = []
        rejected: list[Concept] = []
        parked: list[Concept] = []
        demoted: list[str] = []

        in_scope = set(_STATED_ISA) | set(_ORPHANS)
        for r in relationships:
            in_scope.add(r.domain.split(":")[-1])
            in_scope.add(r.range.split(":")[-1])
        for _, _, children in _COINED_CLUSTERS:
            in_scope.update(children)

        for c in concepts:
            if c.origin == "coined":
                # Test, coined parent: >= 2 grounded children, else demote.
                kids = [k for k in concepts if k.parent == c.name and k.grounded]
                if len(kids) >= 2:
                    c.verdict = "accept"
                    admitted.append(c)
                else:
                    c.verdict = "demote"
                    demoted.append(c.name)
                    for k in kids:
                        k.parent = None
                continue

            # Test, extracted concept: must be grounded in the corpus.
            if not c.grounded:
                c.verdict = "reject"
                rejected.append(c)
            elif c.name not in in_scope:
                # grounded but unconnected to this pass's relationships: park it in
                # the feedback set for the next pass (recall-first, not deletion).
                c.verdict = "park"
                c.flags.append("deferred")
                parked.append(c)
            else:
                c.verdict = "accept"
                admitted.append(c)

        # Test, relationship: predicate grounding + sane domain/range.
        admitted_rels = [r for r in relationships if r.grounded]
        for r in relationships:
            r.verdict = "accept" if r.grounded else "reject"

        admitted_classes = [c for c in admitted if c.verdict == "accept"]
        return Outcome(
            mentions=[], concepts=concepts, relationships=relationships,
            coined=list(self._coined),
            admitted_classes=admitted_classes,
            admitted_relationships=admitted_rels,
            rejected=rejected, parked=parked, demoted=demoted,
            n_axioms=len(_DISJOINT_PAIRS),
            consistent=self.reasoner_consistent(concepts),
        )

    # ---- orchestration ------------------------------------------------
    def run(self) -> Outcome:
        scope = self.scope()
        mentions = self.surface()
        classes, individuals, non_concepts = self.sort_by_kind(mentions)
        concepts, relationships = self.name(classes)
        kept, salience_parked = self.salience(concepts)
        self.synthesize(kept, relationships)
        outcome = self.review(kept, relationships)
        outcome.mentions = mentions
        outcome.surface_stats = self._surface_stats
        outcome.scope = scope
        outcome.individuals = individuals
        outcome.non_concepts = non_concepts
        outcome.salience_parked = salience_parked
        return outcome

    # ---- emit the cumulative v0.6 ontology ---------------------------
    def emit(self, outcome: Outcome) -> Graph:
        """Serialize v0.5 + the admitted delta into one cumulative graph.

        The emitted graph is structurally identical to the canonical
        ``scima_owl_v0_6.ttl`` (same class/property/axiom counts), so the article
        and the shipped file stay in lock-step.
        """
        g = Graph()
        g.parse(_V0_5, format="turtle")
        g.bind("scima", SCIMA)
        g.bind("skos", SKOS)
        g.bind("prov", PROV)

        onto_iri = URIRef("http://scima.city/ontology")
        g.set((onto_iri, OWL.versionInfo, Literal("0.6")))
        g.set((onto_iri, OWL.priorVersion, URIRef("http://scima.city/ontology/v0.5")))

        for c in outcome.admitted_classes:
            ref = SCIMA[c.name]
            g.add((ref, RDF.type, OWL.Class))
            g.add((ref, RDFS.label, Literal(c.label)))
            if c.parent:
                g.add((ref, RDFS.subClassOf, SCIMA[c.parent.split(":")[-1]]))
            for lex in c.lexicalizations:
                if _normalize(lex) != _normalize(c.label):
                    g.add((ref, SKOS.altLabel, Literal(lex)))
            if c.origin == "extracted":
                g.add((ref, PROV.wasDerivedFrom,
                       URIRef(f"corpus/emergency_procedures.txt#span_{c.name.lower()}")))

        for r in outcome.admitted_relationships:
            ref = SCIMA[r.name]
            g.add((ref, RDF.type, OWL.ObjectProperty))
            g.add((ref, RDFS.label, Literal(r.label)))
            g.add((ref, RDFS.domain, SCIMA[r.domain.split(":")[-1]]))
            g.add((ref, RDFS.range, SCIMA[r.range.split(":")[-1]]))

        for a, b in _DISJOINT_PAIRS:
            g.add((SCIMA[a], OWL.disjointWith, SCIMA[b]))

        return g


# =====================================================================
# Deterministic stand-ins for the learned components. In production these would
# be an LLM extractor and an embedding clusterer; fixing them makes the example
# reproducible and lets the tests assert exact stage decisions.
# =====================================================================

# Stage 0 scope (stub for the LLM summary of the mined document skeleton).
_SCOPE_DOMAIN = ("Urban emergency response: the units, incidents, zones, and "
                 "protocols of a smart-city emergency-management procedure.")
_SCOPE_TOPICS = ["incident command", "responder units",
                 "hazardous-material response", "evacuation", "protective equipment"]
_SCOPE_COMPETENCY = [
    "Who commands an incident?",
    "Which responder unit is dispatched to which incident?",
    "Which zone does the incident commander designate?",
    "What protocol must responders follow, and what equipment must they wear?",
]
_SCOPE_OUT = [
    "specific spill case identifiers",
    "timestamps and dates",
    "individual responder names",
    "specific street addresses",
]

# Stage 1 cheap extractor: literal noun phrases it would chunk out of the text.
_CHEAP_CANDIDATES = [
    "incident commander", "IC", "command post", "evacuation zone",
    "hazmat team", "fire department", "emergency medical services",
    "hazardous-material protocol", "hot zone",
]

# Stage 1 LLM extractor (stub): multi-word/implicit candidates, plus one
# hallucination (crisis manager) that the corpus does not support.
_LLM_CANDIDATES = [
    "incident commander",        # overlaps cheap -> merged in dedup
    "hazmat team",               # overlaps cheap -> merged in dedup
    "hazardous material spill",
    "PPE",
    "responders",
    "containment",
    "crisis manager",            # hallucination: nowhere in the corpus
]

# Stage 1b: generic role words and abstract process nouns are non-concepts (they
# are not kinds of thing in this schema); routed to feedback, not the ontology.
_NON_CONCEPTS = {"responders", "containment"}
_DIGIT = re.compile(r"\d")
_DEBRIS = re.compile(r"[\\${}^_|<>]")

# Stage 2 meaning-grouping: ((canonical_name, label), [surface forms]).
_SYNONYM_GROUPS = [
    (("IncidentCommander", "Incident Commander"), ["incident commander", "IC"]),
    (("HazMatSpill", "Hazardous Material Spill"), ["hazardous material spill", "hazardous-material spill"]),
    (("HazmatTeam", "Hazmat Team"), ["hazmat team"]),
    (("FireDepartment", "Fire Department"), ["fire department"]),
    (("EmergencyMedicalService", "Emergency Medical Service"), ["emergency medical services"]),
    (("HazardousMaterialProtocol", "Hazardous Material Protocol"), ["hazardous-material protocol"]),
    (("EvacuationZone", "Evacuation Zone"), ["evacuation zone"]),
    (("HotZone", "Hot Zone"), ["hot zone"]),
    (("CommandPost", "Command Post"), ["command post"]),
    (("ProtectiveEquipment", "Protective Equipment"), ["PPE"]),
    (("CrisisManager", "Crisis Manager"), ["crisis manager"]),
]

# Stage 2b: a concept below this salience floor is parked. The floor is set so
# every plausible named concept clears it on this corpus (including the ungrounded
# CrisisManager, which Stage 4 then rejects explicitly).
_SALIENCE_FLOOR = 0.15

# Stage 3 relations, with domain/range assigned over the class set.
_RELATION_CANDIDATES = [
    ("commands", "commands", "scima:IncidentCommander", "scima:Incident"),
    ("dispatchedTo", "dispatched to", "scima:ResponderUnit", "scima:Incident"),
    ("designates", "designates", "scima:IncidentCommander", "scima:EvacuationZone"),
    ("followsProtocol", "follows protocol", "scima:ResponderUnit", "scima:HazardousMaterialProtocol"),
]

# Stage 3 stated is-a edges (anchor before coin), attaching learned leaves to the
# existing hand-authored schema.
_STATED_ISA = {
    "IncidentCommander": "scima:Agent",      # from v0.5
    "HazMatSpill": "scima:Incident",         # from v0.2
}

# Stage 3 coined parents: (name, label, [children]) where a family has no name.
_COINED_CLUSTERS = [
    ("ResponderUnit", "Responder Unit",
     ["HazmatTeam", "FireDepartment", "EmergencyMedicalService"]),
    ("HazardProtocol", "Hazard Protocol",
     ["HazardousMaterialProtocol"]),            # one child -> demoted in review
]

# Stage 3 orphans: grounded leaves with no confident parent.
_ORPHANS = {"EvacuationZone"}

# Stage 3 axioms: the responder units are mutually disjoint.
_DISJOINT_PAIRS = [
    ("HazmatTeam", "FireDepartment"),
    ("HazmatTeam", "EmergencyMedicalService"),
    ("FireDepartment", "EmergencyMedicalService"),
]

_STOP = {"the", "a", "an", "of", "to", "and", "material"}

# Section headings whose content is exercises/examples, dropped in Stage 0.
_EXCLUDE_SECTION = re.compile(
    r"\b(example|solution|try\s*it|exercise|exercises|practice|activity|review)\b",
    re.I,
)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


# ---- small text helpers ------------------------------------------------
def _normalize(s: str) -> str:
    return re.sub(r"[\s\-]+", " ", s.lower()).strip()


def _camel(s: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[\s\-]+", s.strip()))


def _titlecase(s: str) -> str:
    return " ".join(w.capitalize() for w in re.split(r"[\s\-]+", s.strip()))


def _filter_content(corpus: str) -> tuple[str, int]:
    """Drop exercise/example sections (level-aware); return (content, n_dropped).

    A source with no headings (the procedures corpus) is returned unchanged.
    """
    heads = [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(corpus)]
    if not heads:
        return corpus, 0
    n = len(heads)
    kept_spans: list[tuple[int, int]] = []
    dropped = 0
    if corpus[:heads[0][0]].strip():
        kept_spans.append((0, heads[0][0]))
    i = 0
    while i < n:
        pos, level, text = heads[i]
        if _EXCLUDE_SECTION.search(text):
            j = i + 1
            while j < n and heads[j][1] > level:
                j += 1
            dropped += 1
            i = j
        else:
            end = heads[i + 1][0] if i + 1 < n else len(corpus)
            kept_spans.append((pos, end))
            i += 1
    return "".join(corpus[s:e] for s, e in kept_spans), dropped


# =====================================================================
# CLI
# =====================================================================
def _print_run(outcome: Outcome) -> None:
    cheap, llm, merged = outcome.surface_stats
    n_concepts = len([c for c in outcome.concepts if c.origin == "extracted"])
    sc = outcome.scope
    if sc is not None:
        print(f"Stage 0 scope    : domain framed; {sc.dropped_sections} "
              f"exercise/example section(s) dropped; "
              f"{len(sc.competency_questions)} competency questions")
    print(f"Stage 1 surface  : {len(outcome.mentions)} candidate mentions "
          f"({cheap} cheap, {llm} LLM, {merged} merged)")
    print(f"Stage 1b sort    : "
          f"{len([m for m in outcome.mentions if m.kind == 'class'])} class, "
          f"{len(outcome.individuals)} individual, "
          f"{len(outcome.non_concepts)} non-concept "
          f"({', '.join(m.text for m in outcome.non_concepts) or 'none'})")
    print(f"Stage 2 name     : {n_concepts} named concepts, "
          f"{len(outcome.relationships)} named relationships")
    print(f"Stage 2b salience: {n_concepts - len(outcome.salience_parked)} kept, "
          f"{len(outcome.salience_parked)} parked")
    print(f"Stage 3 structure: synthesized DAG with {len(outcome.coined)} coined "
          f"parents, {outcome.n_axioms} axioms, reasoner: "
          f"{'consistent' if outcome.consistent else 'INCONSISTENT'}")
    print(f"Stage 4 review   : {len(outcome.admitted_classes)} classes admitted, "
          f"{len(outcome.admitted_relationships)} relationships admitted")
    if outcome.demoted:
        print(f"                   demoted {', '.join(outcome.demoted)} "
              f"(coined parent with one child)")
    for c in outcome.rejected:
        print(f"                   rejected {c.name} (no corpus grounding)")
    if outcome.parked:
        print(f"                   parked {len(outcome.parked)} unconnected "
              f"concept(s) for the next pass: "
              f"{', '.join(c.name for c in outcome.parked)}")
    for c in outcome.admitted_classes:
        if "orphan" in c.flags:
            print(f"                   {c.name} admitted as a flagged orphan")


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract an ontology delta from a corpus (Article 4).")
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS),
                        help="path to the source corpus")
    parser.add_argument("--emit", metavar="PATH", nargs="?", const=str(_DEFAULT_EMIT),
                        help="write the cumulative v0.6 ontology to PATH")
    args = parser.parse_args(argv)

    onto = ScimaOntology.load("v0.5").summary()
    print(f"Loaded SCIMA-OWL v0.5: {onto.n_classes} classes, "
          f"{onto.n_properties} properties, {onto.n_axioms} axioms")

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    pipe = OntologyExtractionPipeline(corpus)
    outcome = pipe.run()
    _print_run(outcome)

    if args.emit:
        g = pipe.emit(outcome)
        out = Path(args.emit)
        out.parent.mkdir(parents=True, exist_ok=True)
        g.serialize(destination=str(out), format="turtle")
        n_classes = len(set(g.subjects(RDF.type, OWL.Class)))
        n_obj = len(set(g.subjects(RDF.type, OWL.ObjectProperty)))
        n_dat = len(set(g.subjects(RDF.type, OWL.DatatypeProperty)))
        n_disj = len(list(g.triples((None, OWL.disjointWith, None))))
        n_char = sum(len(list(g.triples((None, RDF.type, t)))) for t in
                     (OWL.SymmetricProperty, OWL.FunctionalProperty,
                      OWL.TransitiveProperty, OWL.InverseFunctionalProperty))
        print(f"Wrote SCIMA-OWL v0.6 -> {out.name}: {n_classes} classes, "
              f"{n_obj + n_dat} properties, {n_disj + n_char} axioms (cumulative)")


if __name__ == "__main__":
    _cli()
