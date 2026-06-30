"""Tests that keep Article 4 honest.

Two contracts:
  1. SCIMA-OWL v0.6 matches the Growth Tracker (cumulative): 26 classes,
     34 properties (19 object + 15 datatype), 15 axioms, and stays backward
     compatible with v0.5.
  2. The seven-stage extraction pipeline behaves as the article describes:
     Stage 0 scopes the domain and drops exercise/example sections; Stage 1
     surfaces recall-first; Stage 1b sorts candidates by kind (class vs
     individual vs non-concept); Stage 2 names by meaning; Stage 2b ranks by
     salience; Stage 3 synthesizes a structured DAG with coined parents and
     domain/range; and Stage 4 runs the agentic RITE review, which accepts the
     grounded, rejects the hallucination (CrisisManager), demotes the one-child
     coined parent (HazardProtocol), and parks the orphan/unconnected concepts
     instead of deleting them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdflib import OWL, RDF, RDFS, Namespace

from scima.ontology import ScimaOntology
from scima.ontology_extraction import (
    OntologyExtractionPipeline,
    _DEFAULT_CORPUS,
    _SALIENCE_FLOOR,
    _filter_content,
)

SCIMA = Namespace("http://scima.city/ontology#")

# ---- Growth Tracker targets for v0.6 (cumulative) ----
EXPECTED_CLASSES = 26
EXPECTED_PROPERTIES = 34
EXPECTED_AXIOMS = 15


# ===================================================================
# Schema: SCIMA-OWL v0.6
# ===================================================================

@pytest.fixture(scope="module")
def onto():
    return ScimaOntology.load("v0.6")


def test_v0_6_parses(onto):
    assert len(onto.graph) > 0


def test_v0_6_class_count(onto):
    assert len(onto.classes()) == EXPECTED_CLASSES


def test_v0_6_property_split(onto):
    s = onto.summary()
    assert s.n_object_properties == 19
    assert s.n_datatype_properties == 15
    assert s.n_properties == EXPECTED_PROPERTIES


def test_v0_6_axiom_count(onto):
    assert onto.axiom_count() == EXPECTED_AXIOMS


def test_v0_6_summary_matches_growth_tracker(onto):
    s = onto.summary()
    assert (s.n_classes, s.n_properties, s.n_axioms) == (
        EXPECTED_CLASSES, EXPECTED_PROPERTIES, EXPECTED_AXIOMS,
    )


def test_v0_6_is_backward_compatible_with_v0_5():
    """Every v0.5 class survives into v0.6 (extraction extends, never removes)."""
    v05 = set(ScimaOntology.load("v0.5").classes())
    v06 = set(ScimaOntology.load("v0.6").classes())
    assert v05 <= v06


def test_v0_6_reuses_existing_parents(onto):
    """Learned leaves attach under the hand-authored schema, not duplicates."""
    assert "scima:IncidentCommander" in onto.subclasses_of("scima:Agent")
    assert "scima:HazMatSpill" in onto.subclasses_of("scima:Incident")


def test_v0_6_coined_responder_unit_has_three_children(onto):
    kids = set(onto.subclasses_of("scima:ResponderUnit"))
    assert kids == {
        "scima:HazmatTeam", "scima:FireDepartment", "scima:EmergencyMedicalService",
    }


def test_v0_6_rejected_class_is_absent(onto):
    """CrisisManager was rejected in review, so it must not be in the schema."""
    assert "scima:CrisisManager" not in onto.classes()


def test_v0_6_demoted_parent_is_absent_but_child_remains(onto):
    """HazardProtocol was demoted; only the lone child survives, parentless."""
    assert "scima:HazardProtocol" not in onto.classes()
    assert "scima:HazardousMaterialProtocol" in onto.classes()
    parents = list(onto.graph.objects(SCIMA.HazardousMaterialProtocol, RDFS.subClassOf))
    assert parents == []


def test_v0_6_responder_units_are_disjoint(onto):
    pairs = {
        frozenset((str(a).split("#")[-1], str(b).split("#")[-1]))
        for a, _, b in onto.graph.triples((None, OWL.disjointWith, None))
    }
    assert frozenset(("HazmatTeam", "FireDepartment")) in pairs
    assert frozenset(("HazmatTeam", "EmergencyMedicalService")) in pairs
    assert frozenset(("FireDepartment", "EmergencyMedicalService")) in pairs


# ===================================================================
# Pipeline: the seven stages over the procedures corpus
# ===================================================================

@pytest.fixture(scope="module")
def outcome():
    corpus = _DEFAULT_CORPUS.read_text(encoding="utf-8")
    return OntologyExtractionPipeline(corpus).run()


# ---- Stage 0: Scope ----

def test_stage0_scope_frames_the_domain(outcome):
    sc = outcome.scope
    assert sc is not None
    assert sc.domain
    assert len(sc.competency_questions) >= 3
    assert sc.out_of_scope                      # names instance kinds to keep out


def test_stage0_clean_corpus_drops_no_sections(outcome):
    """The procedures paragraph has no exercise/example headings to drop."""
    assert outcome.scope.dropped_sections == 0


def test_stage0_content_filter_removes_exercise_sections():
    """A source with exercises drops those sections (and their subsections)."""
    md = (
        "# Whole Numbers\nNumbers used to count.\n\n"
        "# EXAMPLE 1.1\n## Step\n144 divided by 12.\n\n"
        "# Fractions\nA part of a whole.\n\n"
        "# 1.1 EXERCISES\n## Use Place Value\nProblem 1.\n"
    )
    content, dropped = _filter_content(md)
    assert dropped == 2                          # EXAMPLE and EXERCISES blocks
    assert "Whole Numbers" in content and "Fractions" in content
    assert "144 divided by 12" not in content    # example body gone
    assert "Problem 1" not in content            # exercise subsection gone


# ---- Stage 1b: Sort by kind ----

def test_stage1b_sorts_non_concepts_out(outcome):
    non = {m.text for m in outcome.non_concepts}
    assert non == {"responders", "containment"}  # generic / process words
    assert outcome.individuals == []             # no numeric particulars in this corpus
    assert all(m.kind in {"class", "individual", "non_concept"}
               for m in outcome.mentions)


def test_stage1b_classes_feed_naming(outcome):
    """Only class mentions reach naming; sorted-out terms are not named concepts."""
    named = {lex.lower() for c in outcome.concepts for lex in c.lexicalizations}
    assert "responders" not in named
    assert "containment" not in named


# ---- Stage 2b: Salience ----

def test_stage2b_keeps_the_core(outcome):
    """On this small on-topic corpus every named concept is salient (none parked)."""
    assert outcome.salience_parked == []
    extracted = [c for c in outcome.concepts if c.origin == "extracted"]
    assert extracted and all(c.salience >= _SALIENCE_FLOOR for c in extracted)


def test_stage2b_keeps_ungrounded_for_review(outcome):
    """Salience does not prune the ungrounded hallucination; Stage 4 rejects it."""
    cm = next(c for c in outcome.concepts if c.name == "CrisisManager")
    assert cm not in outcome.salience_parked
    assert cm.name in {c.name for c in outcome.rejected}


def test_stage1_surface_unions_and_dedups(outcome):
    cheap, llm, merged = outcome.surface_stats
    assert (cheap, llm, merged) == (9, 7, 2)
    assert len(outcome.mentions) == cheap + llm - merged == 14


def test_stage1_cheap_mentions_are_all_grounded(outcome):
    """The cheap extractor only emits literal spans, so it cannot hallucinate."""
    for m in outcome.mentions:
        if m.source == "cheap":
            assert m.grounded


def test_stage2_names_eleven_concepts_and_four_relationships(outcome):
    named = [c for c in outcome.concepts if c.origin == "extracted"]
    assert len(named) == 11
    assert len(outcome.relationships) == 4


def test_stage2_merges_synonyms_into_one_concept(outcome):
    ic = next(c for c in outcome.concepts if c.name == "IncidentCommander")
    assert "IC" in ic.lexicalizations
    assert "incident commander" in ic.lexicalizations


def test_stage3_coins_two_parents(outcome):
    assert set(outcome.coined) == {"ResponderUnit", "HazardProtocol"}


def test_stage3_reasoner_is_consistent(outcome):
    assert outcome.consistent is True
    assert outcome.n_axioms == 3


def test_stage4_admits_eight_classes_and_four_relationships(outcome):
    assert len(outcome.admitted_classes) == 8
    assert len(outcome.admitted_relationships) == 4
    names = {c.name for c in outcome.admitted_classes}
    assert names == {
        "IncidentCommander", "HazMatSpill", "ResponderUnit", "HazmatTeam",
        "FireDepartment", "EmergencyMedicalService", "HazardousMaterialProtocol",
        "EvacuationZone",
    }


def test_stage4_rejects_ungrounded_hallucination(outcome):
    rejected = {c.name for c in outcome.rejected}
    assert rejected == {"CrisisManager"}
    assert "CrisisManager" not in {c.name for c in outcome.admitted_classes}


def test_stage4_demotes_one_child_coined_parent(outcome):
    assert outcome.demoted == ["HazardProtocol"]
    # the lone child is left parentless after demotion
    proto = next(c for c in outcome.concepts if c.name == "HazardousMaterialProtocol")
    assert proto.parent is None


def test_stage4_parks_unconnected_concepts_not_deletes(outcome):
    """Recall-first: grounded-but-unconnected concepts go to the feedback set."""
    parked = {c.name for c in outcome.parked}
    assert parked == {"HotZone", "CommandPost", "ProtectiveEquipment"}
    for c in outcome.parked:
        assert "deferred" in c.flags


def test_stage4_evacuation_zone_admitted_as_flagged_orphan(outcome):
    ez = next(c for c in outcome.admitted_classes if c.name == "EvacuationZone")
    assert ez.parent is None
    assert "orphan" in ez.flags


# ===================================================================
# Emit: the pipeline reproduces the canonical v0.6 file
# ===================================================================

def test_emit_matches_canonical_counts(outcome):
    """The pipeline's emitted graph has the same counts as the shipped file."""
    corpus = _DEFAULT_CORPUS.read_text(encoding="utf-8")
    pipe = OntologyExtractionPipeline(corpus)
    g = pipe.emit(outcome)
    n_classes = len(set(g.subjects(RDF.type, OWL.Class)))
    n_obj = len(set(g.subjects(RDF.type, OWL.ObjectProperty)))
    n_dat = len(set(g.subjects(RDF.type, OWL.DatatypeProperty)))
    n_disj = len(list(g.triples((None, OWL.disjointWith, None))))
    assert n_classes == EXPECTED_CLASSES
    assert n_obj + n_dat == EXPECTED_PROPERTIES
    # disjointness: 8 carried from v0.5 (5 from v0.1/0.2 + 3 from v0.5) plus the
    # 3 new responder-unit axioms emitted by the pipeline.
    assert n_disj == 11


def test_emit_class_set_equals_canonical():
    """Emitted class set is identical to the canonical scima_owl_v0_6.ttl."""
    corpus = _DEFAULT_CORPUS.read_text(encoding="utf-8")
    pipe = OntologyExtractionPipeline(corpus)
    emitted = pipe.emit(pipe.run())
    emitted_classes = {
        str(s).split("#")[-1] for s in emitted.subjects(RDF.type, OWL.Class)
    }
    canonical_classes = {c.split(":")[-1] for c in ScimaOntology.load("v0.6").classes()}
    assert emitted_classes == canonical_classes
