# ontology_kg_for_agents

Code that backs the article series **"Ontology and Knowledge Graphs for
Intelligent Agents"** (see `../ontology_KG_agents/ontology_KG_agent_series_plan.md`).

Rather than scatter 18 disconnected snippets across 18 articles, this repo
holds one coherent implementation of **SCIMA** (the SmartCity Infrastructure
Management Agent) that grows alongside the series. Each article draws from,
and extends, the modules here. Tests keep the published code honest as the
ontology evolves from v0.1 to v2.1.

## Layout

```
ontology_kg_for_agents/
├── README.md
├── requirements.txt
├── scima/                     # the SCIMA package (grows across the series)
│   ├── __init__.py
│   ├── ontology.py            # load + inspect versioned SCIMA-OWL files
│   ├── building_blocks.py     # Article 1: classes, individuals, properties, axioms
│   ├── knowledge_graph.py     # Article 2: populate triples, named graphs, SPARQL, geo query
│   ├── context_graph.py       # Article 3: k-hop projection, relevance scoring, turns, eviction
│   ├── ontology_extraction.py # Article 4: seven-stage Scope/Surface/Sort/Name/Salience/Structure/Review + RITE
│   └── kg_extraction.py       # Article 5: schema-first KG extraction + four-verdict compliance gate
├── ontologies/                # canonical SCIMA-OWL, one file per version
│   ├── scima_owl_v0_1.ttl     # Article 1 (8 classes, 12 properties, 5 axioms)
│   ├── scima_owl_v0_2.ttl     # Article 2 (12 classes, 20 properties, 8 axioms)
│   ├── scima_owl_v0_5.ttl     # Article 3 (18 classes, 30 properties, 12 axioms)
│   └── scima_owl_v0_6.ttl     # Article 4 (26 classes, 34 properties, 15 axioms)
├── shapes/
│   └── scima_shacl_v0_6.ttl   # Article 5: SHACL shapes generated from SCIMA-OWL v0.6
└── tests/
    ├── test_article_01.py     # asserts v0.1 matches the Growth Tracker
    ├── test_article_02.py     # asserts v0.2 + SCIMA-KG population and queries
    ├── test_article_03.py     # asserts v0.5 + context-graph projection and turns
    ├── test_article_04.py     # asserts v0.6 + the learned emergency-response delta
    └── test_article_05.py     # asserts the compliance gate verdicts + a conformant A-Box
```

As the series proceeds, new modules join `scima/` (context graphs, belief
updates, particle filters, KG merging, GraphRAG) and new `scima_owl_vX_Y.ttl`
files join `ontologies/`, each with a matching test file.

## Article to code map

| Article | Concept | Code |
|---------|---------|------|
| 1 | What is an ontology? | `scima/building_blocks.py`, `ontologies/scima_owl_v0_1.ttl` |
| 2 | Knowledge graphs, triples, SPARQL | `scima/knowledge_graph.py`, `ontologies/scima_owl_v0_2.ttl` |
| 3 | Context graphs, dynamic working memory | `scima/context_graph.py`, `ontologies/scima_owl_v0_5.ttl` |
| 4 | Extracting ontologies from sources | `scima/ontology_extraction.py`, `ontologies/scima_owl_v0_6.ttl` |
| 5 | Extracting ontology-compliant KGs from sources | `scima/kg_extraction.py`, `shapes/scima_shacl_v0_6.ttl` |
| ... | ... | ... |

## Setup

```bash
cd ontology_kg_for_agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Inspect the v0.1 ontology:

```bash
python -m scima.ontology
# SCIMA-OWL v0.1: 8 classes, 12 properties (6 object + 6 datatype), 5 axioms
```

Populate the v0.2 knowledge graph and run the Article 2 geo query:

```bash
python -m scima.knowledge_graph --populate sample --query lights-near I-204
# Loaded SCIMA-OWL v0.2: 12 classes, 20 properties, 8 axioms
# Populated SCIMA-KG sample: 50,000 sensor nodes, ~203,156 triples, 1 named graph
# 3 traffic lights within 500m of I-204, ordered by distance:
#   scima:TL_90  on scima:RoadSegment_Main_St_NB   60 m
#   scima:TL_88  on scima:RoadSegment_Main_St_NB  210 m
#   scima:TL_91  on scima:RoadSegment_Main_St_NB  430 m
```

Build a context graph and trace its evolution across agent turns (Article 3):

```bash
python -m scima.context_graph --build --focal Incident_I204 --goal resolve
# Loaded SCIMA-OWL v0.5: 18 classes, 30 properties, 12 axioms
# Built context graph: focal Incident_I204, 3-hop, 142 nodes (budget 150)

python -m scima.context_graph --trace I-204
# Turn 0 (report): 2 nodes   +Incident_I204, WaterMain_7B
# Turn 1 (expand): 140 nodes  +<infrastructure within 3 hops>
# Turn 2 (refresh): 140 nodes +Vehicle_Ambulance_3  -FlowSensor_A42 (stale)
# Turn 3 (act): 140 nodes     dispatch grounded, assignedTo written back
```

Extract an ontology from source documents through the seven-stage pipeline and
watch the RITE review accept, reject, demote, and park by kind (Article 4):

```bash
python -m scima.ontology_extraction --corpus corpus/emergency_procedures.txt
# Loaded SCIMA-OWL v0.5: 18 classes, 30 properties, 12 axioms
# Stage 0 scope    : domain framed; 0 exercise/example section(s) dropped; 4 competency questions
# Stage 1 surface  : 14 candidate mentions (9 cheap, 7 LLM, 2 merged)
# Stage 1b sort    : 12 class, 0 individual, 2 non-concept (responders, containment)
# Stage 2 name     : 11 named concepts, 4 named relationships
# Stage 2b salience: 11 kept, 0 parked
# Stage 3 structure: synthesized DAG with 2 coined parents, 3 axioms, reasoner: consistent
# Stage 4 review   : 8 classes admitted, 4 relationships admitted
#                    demoted HazardProtocol (coined parent with one child)
#                    rejected CrisisManager (no corpus grounding)
#                    parked 3 unconnected concept(s) for the next pass

python -m scima.ontology_extraction --emit   # machine copy under build/
# Wrote SCIMA-OWL v0.6 -> scima_owl_v0_6.ttl: 26 classes, 34 properties, 15 axioms (cumulative)
```

Extract an **ontology-compliant** knowledge graph from a source feed, gating
every candidate triple against the fixed SCIMA-OWL v0.6 schema (Article 5):

```bash
python -m scima.kg_extraction --feed corpus/incident_report_I204.txt
# Loaded SCIMA-OWL v0.6: 26 classes, 34 properties (19 object + 15 datatype), 15 axioms (fixed target schema)
# Extracted 25 candidate triples from 1 feed (incident_report).
# Compliance gate: 20 admitted, 1 repaired, 3 rejected, 1 quarantined.
#   [REJECT] scima:IC_Diaz scima:authorizes scima:HMP_1   relation scima:authorizes not in ontology
#   [REJECT] scima:FD_12 rdf:type scima:HazmatTeam        scima:HazmatTeam disjoint with already-asserted scima:FireDepartment
#   [REJECT] scima:WM_7B scima:dispatchedTo scima:Incident_I204  domain violation: subject is not a scima:ResponderUnit
#   [REPAIR] scima:Reading_R1 scima:observedValue 47      cast xsd:string->xsd:integer
#   [QUARN ] scima:Incident_I205 rdf:type scima:HazMatSpill  confidence 0.40 < 0.5
# Admitted A-Box: 21 triples, 10 typed individuals.
# SHACL validation over admitted graph: conforms = True (0 violations).
# Reasoner consistency: consistent = True.
```

The SHACL shapes in `shapes/scima_shacl_v0_6.ttl` are the standards-track form
of the gate; the module runs the equivalent checks natively (pyshacl is
optional, the same way `knowledge_graph.py` computes haversine instead of
GeoSPARQL). Conformance is True by construction: the gate never admits a triple
that would fail a shape.

Run the tests:

```bash
pytest
```

## Design notes

- **The Turtle file is the source of truth for tooling.** `building_blocks.py`
  is the source of truth for the *prose*: a readable Python mirror of the same
  content, so the article's vocabulary can be imported and tested.
- **The Growth Tracker is the contract.** `tests/test_article_01.py` asserts
  v0.1 has exactly 8 classes / 12 properties / 5 axioms, matching the table in
  the series plan. Future versions get their own assertions.
- **OWL DL throughout.** SCIMA-OWL stays inside the decidable OWL DL profile so
  reasoning (Articles 8, 12) is guaranteed to terminate.
