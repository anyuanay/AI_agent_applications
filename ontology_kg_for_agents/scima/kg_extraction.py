"""Extract an ontology-compliant knowledge graph from source text (Article 5).

Article 4 produced SCIMA-OWL v0.8: a T-Box of classes, properties, and
axioms. This module fills the A-Box. The pipeline takes source text and a
fixed ontology, and runs three sequential stages:

  Stage 1  Extract  -> spaCy NER + dep parse (stub) and LLM pass (stub),
                       entity mention merge, triple merge, negation gate.
  Stage 2  Map      -> build an ontology index once; then for each candidate
                       triple: subject typing, predicate mapping, object
                       mapping via an embedding-then-LLM cascade (stub).
  Stage 3  Verify   -> sort by confidence, five sequential checks (domain,
                       range, datatype repair, disjointness, cardinality),
                       running state updated on each admit.

The spaCy/embedding/LLM steps are deterministic stubs: they reproduce the
exact Incident Report I-204 example from the article. The ontology-index
builder, IRI minting, and all five Stage 3 checks are real, operating
against the v0.8 TTL. The pipeline structure and the emitted A-Box are the
testable contract.

Usage:
    python -m scima.kg_extraction
    python -m scima.kg_extraction --corpus corpus/incident_report_I204.txt
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

from scima.ontology import ScimaOntology

SCIMA = Namespace("http://scima.city/ontology#")

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CORPUS = _ROOT / "corpus" / "incident_report_I204.txt"
_DEFAULT_ONTOLOGY_VERSION = "v0.8"


# =====================================================================
# Intermediate representations flowing through the pipeline.
# =====================================================================

@dataclass
class Sentence:
    index: int
    text: str


@dataclass
class EntityMention:
    surface_form: str
    mention_type: str           # "entity" or "value"
    sources: list[str]          # "ner", "chunk", "llm"
    sentences: list[int]


@dataclass
class RawTriple:
    subject: str
    predicate: str
    object: str
    sentence_index: int
    source: str                 # "dep_parse" or "llm"
    negated: bool = False


@dataclass
class CandidateTriple:
    subject: str
    predicate: str
    object: str
    object_type: Optional[str]  # "entity", "value", or None
    sentence_index: int
    sentence_indices: list[int]
    sources: list[str]
    negated: bool = False


@dataclass
class OntologyIndex:
    """Pre-computed structural index over the ontology; Stage 3 reads this."""
    classes: dict[str, dict]            # local_name -> {iri, label}
    properties: dict[str, dict]         # local_name -> {iri, label, kind, domain, range, functional, inverse_functional}
    subclass_closure: dict[str, set[str]]  # class_name -> ancestors including self (SCIMA only)
    disjoint_pairs: list[tuple[str, str]]
    functional_props: set[str]
    inverse_functional_props: set[str]


@dataclass
class MappedTriple:
    subject_iri: str
    subject_type: str
    predicate_iri: str
    predicate_kind: str              # "object_property" or "datatype_property"
    object_iri: Optional[str]        # set when predicate_kind == object_property
    object_literal: Optional[str]    # set when predicate_kind == datatype_property
    object_datatype: Optional[str]   # e.g. "xsd:string", "xsd:integer"
    object_type: Optional[str]       # set when predicate_kind == object_property
    sentence_index: int
    sentence_indices: list[int]
    sources: list[str]
    mapping_confidence: str          # "high" or "low"
    inverted: bool = False


@dataclass
class AdmittedTriple:
    subject_iri: str
    predicate_iri: str
    object_iri: Optional[str]
    object_literal: Optional[str]
    object_datatype: Optional[str]
    verdict: str                     # "admit" or "repaired"


@dataclass
class RejectedTriple:
    subject_iri: str
    predicate_iri: str
    object_iri: Optional[str]
    reason: str


@dataclass
class TypeAssertion:
    subject_iri: str
    predicate_iri: str               # always "rdf:type"
    object_iri: str                  # the class IRI


@dataclass
class PipelineResult:
    sentences: list[Sentence]
    entity_mentions: list[EntityMention]
    dep_triples: list[RawTriple]
    llm_triples: list[RawTriple]
    candidate_triples: list[CandidateTriple]
    negated_triples: list[CandidateTriple]
    ontology_index: OntologyIndex
    mapped_triples: list[MappedTriple]
    admitted_triples: list[AdmittedTriple]
    type_assertions: list[TypeAssertion]
    rejected_triples: list[RejectedTriple]


# =====================================================================
# Ontology index builder (runs over the rdflib graph; not a stub).
# =====================================================================

def _build_ontology_index(graph: Graph) -> OntologyIndex:
    """Parse the rdflib graph once and build the structured OntologyIndex."""
    scima_str = str(SCIMA)

    def _local(uri: URIRef) -> str:
        return str(uri).split("#")[-1]

    def _is_scima(ref) -> bool:
        return isinstance(ref, URIRef) and str(ref).startswith(scima_str)

    # Classes
    classes: dict[str, dict] = {}
    for cls in graph.subjects(RDF.type, OWL.Class):
        if _is_scima(cls):
            local = _local(cls)
            label = str(next(graph.objects(cls, RDFS.label), Literal(local)))
            classes[local] = {"iri": str(cls), "label": label}

    # Subclass closure within SCIMA namespace
    direct_parent: dict[str, str] = {}
    for s, _, o in graph.triples((None, RDFS.subClassOf, None)):
        if _is_scima(s) and _is_scima(o):
            direct_parent[_local(s)] = _local(o)

    subclass_closure: dict[str, set[str]] = {}

    def _closure(name: str) -> set[str]:
        if name in subclass_closure:
            return subclass_closure[name]
        result: set[str] = {name}
        if name in direct_parent:
            result |= _closure(direct_parent[name])
        subclass_closure[name] = result
        return result

    for cls_name in classes:
        _closure(cls_name)

    # Functional and inverse-functional property sets
    functional: set[str] = set()
    for p in graph.subjects(RDF.type, OWL.FunctionalProperty):
        if _is_scima(p):
            functional.add(_local(p))

    inverse_functional: set[str] = set()
    for p in graph.subjects(RDF.type, OWL.InverseFunctionalProperty):
        if _is_scima(p):
            inverse_functional.add(_local(p))

    # Properties
    properties: dict[str, dict] = {}
    for prop_type, kind in [
        (OWL.ObjectProperty, "object_property"),
        (OWL.DatatypeProperty, "datatype_property"),
    ]:
        for prop in graph.subjects(RDF.type, prop_type):
            if not _is_scima(prop):
                continue
            local = _local(prop)
            label = str(next(graph.objects(prop, RDFS.label), Literal(local)))

            domain_ref = next(graph.objects(prop, RDFS.domain), None)
            range_ref = next(graph.objects(prop, RDFS.range), None)

            domain = None
            if domain_ref and isinstance(domain_ref, URIRef):
                if str(domain_ref).startswith(scima_str):
                    domain = _local(domain_ref)

            range_val = None
            if range_ref and isinstance(range_ref, URIRef):
                if str(range_ref).startswith(scima_str):
                    range_val = _local(range_ref)
                elif "XMLSchema#" in str(range_ref):
                    range_val = "xsd:" + str(range_ref).split("#")[-1]

            properties[local] = {
                "iri": str(prop),
                "label": label,
                "kind": kind,
                "domain": domain,
                "range": range_val,
                "functional": local in functional,
                "inverse_functional": local in inverse_functional,
            }

    # Disjoint pairs within SCIMA
    disjoint_pairs: list[tuple[str, str]] = []
    for a, _, b in graph.triples((None, OWL.disjointWith, None)):
        if _is_scima(a) and _is_scima(b):
            disjoint_pairs.append((_local(a), _local(b)))

    return OntologyIndex(
        classes=classes,
        properties=properties,
        subclass_closure=subclass_closure,
        disjoint_pairs=disjoint_pairs,
        functional_props=functional,
        inverse_functional_props=inverse_functional,
    )


# =====================================================================
# Deterministic stubs for Stage 1 and Stage 2 learned components.
# These reproduce the Incident Report I-204 example from the article.
# In production these would be spaCy, an embedding model, and an LLM.
# =====================================================================

_STUB_SENTENCES = [
    Sentence(1, "Commander Diaz commands HazmatTeam Alpha at the scene."),
    Sentence(2, "HazmatTeam Alpha was dispatched to Incident I-204."),
    Sentence(3, "Sensor Reading R1 recorded a value of 47 psi."),
    Sentence(4, "WaterMain 7B was dispatched to Incident I-204."),
    Sentence(5, "Commander Diaz commands HazmatTeam Gamma as reserve."),
    Sentence(6, "Commander Diaz did not command HazmatTeam Bravo."),
]

_STUB_DEP_TRIPLES = [
    RawTriple("Commander Diaz", "commands",      "HazmatTeam Alpha",  1, "dep_parse"),
    RawTriple("HazmatTeam Alpha", "dispatchedTo", "Incident I-204",    2, "dep_parse"),
    RawTriple("Reading R1",       "observedValue", "47 psi",           3, "dep_parse"),
    RawTriple("WaterMain 7B",     "dispatchedTo", "Incident I-204",    4, "dep_parse"),
    RawTriple("Commander Diaz", "commands",      "HazmatTeam Gamma",  5, "dep_parse"),
    RawTriple("Commander Diaz", "command",       "HazmatTeam Bravo",  6, "dep_parse", negated=True),
]

_STUB_LLM_TRIPLES = [
    RawTriple("Commander Diaz", "commands",      "HazmatTeam Alpha",  1, "llm"),
    RawTriple("HazmatTeam Alpha", "dispatchedTo", "Incident I-204",    2, "llm"),
    RawTriple("Reading R1",       "observedValue", "47 psi",           3, "llm"),
    RawTriple("WaterMain 7B",     "dispatchedTo", "Incident I-204",    4, "llm"),
    RawTriple("Commander Diaz", "commands",      "HazmatTeam Gamma",  5, "llm"),
    RawTriple("Commander Diaz", "commands",      "HazmatTeam Bravo",  6, "llm", negated=True),
]

_STUB_ENTITY_MENTIONS = [
    EntityMention("Commander Diaz",   "entity", ["ner", "llm"], [1, 5, 6]),
    EntityMention("HazmatTeam Alpha", "entity", ["ner", "llm"], [1, 2]),
    EntityMention("Incident I-204",   "entity", ["ner", "llm"], [2, 4]),
    EntityMention("Reading R1",       "entity", ["ner", "llm"], [3]),
    EntityMention("47 psi",           "value",  ["ner", "llm"], [3]),
    EntityMention("WaterMain 7B",     "entity", ["ner", "llm"], [4]),
    EntityMention("HazmatTeam Gamma", "entity", ["ner", "llm"], [5]),
    EntityMention("HazmatTeam Bravo", "entity", ["ner", "llm"], [6]),
]

# Stub: surface form (lowercase) -> (class_local_name, component_confidence)
_CLASS_LOOKUP: dict[str, tuple[str, str]] = {
    "commander diaz":   ("IncidentCommander", "high"),
    "hazmatteam alpha": ("HazmatTeam",        "high"),
    "hazmatteam gamma": ("HazmatTeam",        "high"),
    "hazmatteam bravo": ("HazmatTeam",        "high"),
    "incident i-204":   ("HazMatSpill",       "high"),
    "reading r1":       ("SensorReading",     "high"),
    "watermain 7b":     ("WaterMain",         "high"),
}

# Stub: predicate (lowercase, optionally lemmatized) -> (prop_local, inverted, confidence)
_PREDICATE_LOOKUP: dict[str, tuple[str, bool, str]] = {
    "commands":      ("commands",      False, "high"),
    "command":       ("commands",      False, "high"),
    "dispatchedto":  ("dispatchedTo",  False, "high"),
    "observedvalue": ("observedValue", False, "high"),
}


# =====================================================================
# IRI minting helpers.
# =====================================================================

def _slug(surface: str) -> str:
    """Lowercase, strip non-alphanumeric (except spaces), replace spaces with underscores."""
    s = surface.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s


# =====================================================================
# The three-stage pipeline.
# =====================================================================

class KGExtractionPipeline:
    """Extract an ontology-compliant A-Box from source text.

    Stage 1 is recall-first and ontology-agnostic. Stage 2 reads the
    ontology once, builds an index, and maps each candidate onto IRIs.
    Stage 3 verifies each mapped triple against structural constraints.
    """

    def __init__(self, corpus: str, ontology_version: str = _DEFAULT_ONTOLOGY_VERSION) -> None:
        self.corpus = corpus
        self.ontology = ScimaOntology.load(ontology_version)
        self._iri_registry: dict[tuple[str, str], str] = {}
        self._slug_counters: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Stage 1: Extract                                                     #
    # ------------------------------------------------------------------ #

    def stage1_extract(self) -> tuple[
        list[Sentence],
        list[EntityMention],
        list[RawTriple],
        list[RawTriple],
        list[CandidateTriple],
        list[CandidateTriple],
    ]:
        """Run spaCy pass (stub) + LLM pass (stub), then merge.

        Returns: sentences, entity_mentions, dep_triples, llm_triples,
                 candidate_triples, negated_triples.
        """
        sentences = list(_STUB_SENTENCES)
        entity_mentions = list(_STUB_ENTITY_MENTIONS)
        dep_triples = list(_STUB_DEP_TRIPLES)
        llm_triples = list(_STUB_LLM_TRIPLES)

        # Build object_type lookup from entity mentions
        obj_type_map: dict[str, str] = {
            m.surface_form.lower(): m.mention_type for m in entity_mentions
        }

        # Merge dep_parse and LLM triples on normalized (subj, predicate_lemma, obj)
        # Predicate normalization: lowercase + simple lemma (strip trailing 's')
        def _norm_pred(p: str) -> str:
            p = p.lower()
            return p.rstrip("s") if p.endswith("s") else p

        merged: dict[tuple[str, str, str], dict] = {}
        for t in dep_triples + llm_triples:
            key = (t.subject.lower(), _norm_pred(t.predicate), t.object.lower())
            if key not in merged:
                merged[key] = {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "sentence_index": t.sentence_index,
                    "sentence_indices": [t.sentence_index],
                    "sources": [t.source],
                    "negated": t.negated,
                }
            else:
                rec = merged[key]
                if t.source not in rec["sources"]:
                    rec["sources"].append(t.source)
                if t.sentence_index not in rec["sentence_indices"]:
                    rec["sentence_indices"].append(t.sentence_index)
                    rec["sentence_indices"].sort()
                if t.negated:
                    rec["negated"] = True

        candidate_triples: list[CandidateTriple] = []
        negated_triples: list[CandidateTriple] = []

        for rec in merged.values():
            ct = CandidateTriple(
                subject=rec["subject"],
                predicate=rec["predicate"],
                object=rec["object"],
                object_type=obj_type_map.get(rec["object"].lower()),
                sentence_index=rec["sentence_index"],
                sentence_indices=rec["sentence_indices"],
                sources=sorted(rec["sources"]),
                negated=rec["negated"],
            )
            if ct.negated:
                negated_triples.append(ct)
            else:
                candidate_triples.append(ct)

        # Sort by sentence_index for determinism
        candidate_triples.sort(key=lambda t: t.sentence_index)
        negated_triples.sort(key=lambda t: t.sentence_index)

        return sentences, entity_mentions, dep_triples, llm_triples, candidate_triples, negated_triples

    # ------------------------------------------------------------------ #
    # Stage 2: Map onto the ontology                                       #
    # ------------------------------------------------------------------ #

    def _mint_iri(self, class_name: str, surface: str) -> str:
        """Return a stable IRI for (class, surface); mint on first encounter."""
        s = _slug(surface)
        key = (class_name, s)
        if key in self._iri_registry:
            return self._iri_registry[key]
        if s in self._slug_counters:
            self._slug_counters[s] += 1
            final = f"{s}_{self._slug_counters[s]}"
        else:
            self._slug_counters[s] = 0
            final = s
        iri = f"scima:{class_name}_{final}"
        self._iri_registry[key] = iri
        return iri

    def _type_entity(self, surface: str) -> Optional[tuple[str, str]]:
        """Stub: embedding cosine vs class labels -> (class_local_name, confidence)."""
        return _CLASS_LOOKUP.get(surface.lower())

    def _map_predicate(self, surface: str) -> Optional[tuple[str, bool, str]]:
        """Stub: lemma/string/embedding match -> (prop_local_name, inverted, confidence)."""
        return _PREDICATE_LOOKUP.get(surface.lower())

    def _parse_literal(self, surface: str) -> tuple[str, str]:
        """Split a value surface form; return (literal_str, xsd_datatype)."""
        surface = surface.strip()
        # unit compound: leading digits optionally followed by non-numeric unit
        m = re.match(r"^(\d+(?:\.\d+)?)\s*\w+$", surface)
        if m:
            return m.group(1), "xsd:string"
        if re.match(r"^\d+$", surface):
            return surface, "xsd:integer"
        if re.match(r"^\d+\.\d+$", surface):
            return surface, "xsd:decimal"
        return surface, "xsd:string"

    def stage2_map(
        self,
        candidates: list[CandidateTriple],
        index: OntologyIndex,
        sentences: list[Sentence],
    ) -> list[MappedTriple]:
        """Map each candidate triple onto ontology IRIs using the index."""
        sent_map = {s.index: s.text for s in sentences}
        mapped: list[MappedTriple] = []

        for ct in candidates:
            sent_text = sent_map.get(ct.sentence_index, "")

            # -- Subject typing --
            subj_result = self._type_entity(ct.subject)
            if subj_result is None:
                continue
            subj_class, subj_conf = subj_result
            subject_iri = self._mint_iri(subj_class, ct.subject)

            # -- Predicate mapping --
            pred_result = self._map_predicate(ct.predicate)
            if pred_result is None:
                continue
            pred_local, inverted, pred_conf = pred_result
            if pred_local not in index.properties:
                continue
            prop_info = index.properties[pred_local]
            pred_iri = f"scima:{pred_local}"
            pred_kind = prop_info["kind"]

            # -- Object mapping --
            obj_iri = None
            obj_lit = None
            obj_dt = None
            obj_type = None
            obj_conf = "high"

            if pred_kind == "object_property":
                obj_result = self._type_entity(ct.object)
                if obj_result is None:
                    continue
                obj_class, obj_conf = obj_result
                obj_iri = self._mint_iri(obj_class, ct.object)
                obj_type = f"scima:{obj_class}"
            else:
                obj_lit, obj_dt = self._parse_literal(ct.object)

            # -- Inversion: swap grammatical subject/object if inverse match --
            if inverted:
                subject_iri, obj_iri = obj_iri, subject_iri
                subj_class, obj_type_local = (
                    (obj_type.split(":")[-1] if obj_type else subj_class),
                    subj_class,
                )
                obj_type = f"scima:{obj_type_local}"
                subj_type_str = f"scima:{subj_class}"
            else:
                subj_type_str = f"scima:{subj_class}"

            # -- Overall confidence: min of three component confidences --
            conf_rank = {"high": 1, "low": 0}
            overall = "high" if all(
                conf_rank.get(c, 0) == 1
                for c in [subj_conf, pred_conf, obj_conf]
            ) else "low"

            mapped.append(MappedTriple(
                subject_iri=subject_iri,
                subject_type=subj_type_str,
                predicate_iri=pred_iri,
                predicate_kind=pred_kind,
                object_iri=obj_iri,
                object_literal=obj_lit,
                object_datatype=obj_dt,
                object_type=obj_type,
                sentence_index=ct.sentence_index,
                sentence_indices=list(ct.sentence_indices),
                sources=list(ct.sources),
                mapping_confidence=overall,
            ))

        return mapped

    # ------------------------------------------------------------------ #
    # Stage 3: Verify and admit                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_subclass(child: str, ancestor: str, index: OntologyIndex) -> bool:
        """True if child is a subclass of ancestor (or equal) in the index."""
        child_local = child.split(":")[-1]
        ancestor_local = ancestor.split(":")[-1]
        closure = index.subclass_closure.get(child_local, {child_local})
        return ancestor_local in closure

    @staticmethod
    def _sort_key(t: MappedTriple) -> tuple:
        """Three-level sort: cross-sentence corroboration, source tier, confidence, index."""
        n_sents = len(t.sentence_indices)
        cross = 0 if n_sents >= 2 else 1       # 0 = higher priority
        sources_set = set(t.sources)
        if sources_set == {"dep_parse", "llm"}:
            src_tier = 0
        elif "dep_parse" in sources_set:
            src_tier = 1
        else:
            src_tier = 2
        conf_tier = 0 if t.mapping_confidence == "high" else 1
        return (cross, src_tier, conf_tier, t.sentence_index)

    def stage3_verify(
        self,
        mapped: list[MappedTriple],
        index: OntologyIndex,
    ) -> tuple[list[AdmittedTriple], list[TypeAssertion], list[RejectedTriple]]:
        """Sort mapped triples by confidence and run five sequential checks."""
        sorted_triples = sorted(mapped, key=self._sort_key)

        admitted: list[AdmittedTriple] = []
        rejected: list[RejectedTriple] = []

        # Running state
        admitted_types: dict[str, set[str]] = {}   # individual_iri -> set of class IRIs
        cardinality_counter: dict[tuple[str, str], int] = {}
        inv_func_tracker: dict[tuple[str, str], str] = {}  # (pred_iri, obj_iri) -> subj_iri

        for t in sorted_triples:
            pred_local = t.predicate_iri.split(":")[-1]
            prop = index.properties.get(pred_local)
            subj_type_local = t.subject_type.split(":")[-1]
            obj_type_local = t.object_type.split(":")[-1] if t.object_type else None

            # Check 1: Domain
            if prop and prop["domain"]:
                if not self._is_subclass(t.subject_type, prop["domain"], index):
                    rejected.append(RejectedTriple(
                        subject_iri=t.subject_iri,
                        predicate_iri=t.predicate_iri,
                        object_iri=t.object_iri,
                        reason="domain_violation",
                    ))
                    continue

            # Check 2 / 3: Range
            if t.predicate_kind == "object_property":
                if prop and prop["range"] and t.object_type:
                    if not self._is_subclass(t.object_type, prop["range"], index):
                        rejected.append(RejectedTriple(
                            subject_iri=t.subject_iri,
                            predicate_iri=t.predicate_iri,
                            object_iri=t.object_iri,
                            reason="range_violation",
                        ))
                        continue
                verdict = "admit"
                final_lit = None
                final_dt = None

            else:  # datatype_property
                declared_range = prop["range"] if prop else None
                lit = t.object_literal
                dt = t.object_datatype
                verdict = "admit"

                if declared_range and dt and declared_range != dt:
                    # Check 3: attempt repair (cast to declared type)
                    repaired_lit, repaired_dt = self._try_cast(lit, declared_range)
                    if repaired_lit is not None:
                        lit, dt, verdict = repaired_lit, repaired_dt, "repaired"
                    else:
                        rejected.append(RejectedTriple(
                            subject_iri=t.subject_iri,
                            predicate_iri=t.predicate_iri,
                            object_iri=None,
                            reason="datatype_mismatch",
                        ))
                        continue
                final_lit = lit
                final_dt = dt

            # Check 4: Disjointness
            # Would admitting this triple's type assignments conflict with existing types?
            proposed_types: dict[str, str] = {t.subject_iri: subj_type_local}
            if obj_type_local:
                proposed_types[t.object_iri] = obj_type_local

            disj_conflict = False
            for ind_iri, new_type in proposed_types.items():
                existing = admitted_types.get(ind_iri, set())
                for existing_type in existing:
                    for a, b in index.disjoint_pairs:
                        if (a == new_type and b == existing_type) or (b == new_type and a == existing_type):
                            disj_conflict = True
                            break
                if disj_conflict:
                    break

            if disj_conflict:
                rejected.append(RejectedTriple(
                    subject_iri=t.subject_iri,
                    predicate_iri=t.predicate_iri,
                    object_iri=t.object_iri,
                    reason="disjointness_violation",
                ))
                continue

            # Check 5a: Functional / max-cardinality (subject side)
            if prop and prop["functional"]:
                count = cardinality_counter.get((t.subject_iri, t.predicate_iri), 0)
                if count >= 1:
                    rejected.append(RejectedTriple(
                        subject_iri=t.subject_iri,
                        predicate_iri=t.predicate_iri,
                        object_iri=t.object_iri,
                        reason="cardinality_violation",
                    ))
                    continue

            # Check 5b: Inverse-functional (object side)
            if prop and prop["inverse_functional"] and t.object_iri:
                tracker_key = (t.predicate_iri, t.object_iri)
                existing_subj = inv_func_tracker.get(tracker_key)
                if existing_subj and existing_subj != t.subject_iri:
                    rejected.append(RejectedTriple(
                        subject_iri=t.subject_iri,
                        predicate_iri=t.predicate_iri,
                        object_iri=t.object_iri,
                        reason="cardinality_violation",
                    ))
                    continue

            # Admit: update running state
            admitted.append(AdmittedTriple(
                subject_iri=t.subject_iri,
                predicate_iri=t.predicate_iri,
                object_iri=t.object_iri,
                object_literal=final_lit if t.predicate_kind == "datatype_property" else None,
                object_datatype=final_dt if t.predicate_kind == "datatype_property" else None,
                verdict=verdict,
            ))

            # Update admitted type set
            admitted_types.setdefault(t.subject_iri, set()).add(subj_type_local)
            if obj_type_local and t.object_iri:
                admitted_types.setdefault(t.object_iri, set()).add(obj_type_local)

            # Update cardinality counter
            key = (t.subject_iri, t.predicate_iri)
            cardinality_counter[key] = cardinality_counter.get(key, 0) + 1

            # Update inverse-functional tracker
            if t.object_iri:
                inv_func_tracker[(t.predicate_iri, t.object_iri)] = t.subject_iri

        # Emit deduplicated rdf:type assertions
        seen_types: set[tuple[str, str]] = set()
        type_assertions: list[TypeAssertion] = []
        for ind_iri, type_set in admitted_types.items():
            for type_local in sorted(type_set):
                class_iri = f"scima:{type_local}"
                pair = (ind_iri, class_iri)
                if pair not in seen_types:
                    seen_types.add(pair)
                    type_assertions.append(TypeAssertion(
                        subject_iri=ind_iri,
                        predicate_iri="rdf:type",
                        object_iri=class_iri,
                    ))

        type_assertions.sort(key=lambda ta: (ta.subject_iri, ta.object_iri))
        return admitted, type_assertions, rejected

    @staticmethod
    def _try_cast(literal: Optional[str], target_xsd: str) -> tuple[Optional[str], Optional[str]]:
        """Attempt to coerce a literal to target_xsd; return (value, type) or (None, None)."""
        if literal is None:
            return None, None
        try:
            if target_xsd == "xsd:integer":
                int(literal)
                return literal, "xsd:integer"
            if target_xsd == "xsd:decimal":
                float(literal)
                return literal, "xsd:decimal"
            if target_xsd == "xsd:boolean":
                if literal.lower() in ("true", "false", "1", "0"):
                    return literal.lower() in ("true", "1") and "true" or "false", "xsd:boolean"
            if target_xsd == "xsd:string":
                return literal, "xsd:string"
        except (ValueError, TypeError):
            pass
        return None, None

    # ------------------------------------------------------------------ #
    # Orchestration                                                        #
    # ------------------------------------------------------------------ #

    def run(self) -> PipelineResult:
        """Run all three stages and return the full result."""
        # Stage 1
        (sentences, entity_mentions, dep_triples, llm_triples,
         candidates, negated) = self.stage1_extract()

        # Build ontology index (Stage 2 entry)
        index = _build_ontology_index(self.ontology.graph)

        # Stage 2
        mapped = self.stage2_map(candidates, index, sentences)

        # Stage 3
        admitted, type_assertions, rejected_triples = self.stage3_verify(mapped, index)

        return PipelineResult(
            sentences=sentences,
            entity_mentions=entity_mentions,
            dep_triples=dep_triples,
            llm_triples=llm_triples,
            candidate_triples=candidates,
            negated_triples=negated,
            ontology_index=index,
            mapped_triples=mapped,
            admitted_triples=admitted,
            type_assertions=type_assertions,
            rejected_triples=rejected_triples,
        )


# =====================================================================
# CLI
# =====================================================================

def _print_result(result: PipelineResult) -> None:
    print(f"Stage 1: {len(result.sentences)} sentences, "
          f"{len(result.entity_mentions)} entity mentions")
    print(f"         dep triples: {len(result.dep_triples)}, "
          f"llm triples: {len(result.llm_triples)}")
    print(f"         candidates: {len(result.candidate_triples)}, "
          f"negated (quarantined): {len(result.negated_triples)}")
    n_classes = len(result.ontology_index.classes)
    n_props = len(result.ontology_index.properties)
    print(f"Stage 2: ontology index: {n_classes} classes, {n_props} properties")
    print(f"         mapped triples: {len(result.mapped_triples)}")
    n_admit = sum(1 for t in result.admitted_triples if t.verdict == "admit")
    n_repair = sum(1 for t in result.admitted_triples if t.verdict == "repaired")
    print(f"Stage 3: admitted: {n_admit}, repaired: {n_repair}, "
          f"rejected: {len(result.rejected_triples)}")
    print(f"         type assertions: {len(result.type_assertions)}")
    for t in result.admitted_triples:
        obj = t.object_iri or f'"{t.object_literal}"^^{t.object_datatype}'
        print(f"  [{t.verdict}] {t.subject_iri} {t.predicate_iri} {obj}")
    for t in result.rejected_triples:
        print(f"  [reject:{t.reason}] {t.subject_iri} {t.predicate_iri} {t.object_iri or '?'}")
    for ta in result.type_assertions:
        print(f"  [type] {ta.subject_iri} rdf:type {ta.object_iri}")


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract an ontology-compliant KG from source text (Article 5).")
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS),
                        help="path to the source text")
    parser.add_argument("--ontology", default=_DEFAULT_ONTOLOGY_VERSION,
                        help="ontology version to use (default: v0.8)")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    pipe = KGExtractionPipeline(corpus, args.ontology)
    result = pipe.run()
    _print_result(result)


if __name__ == "__main__":
    _cli()
