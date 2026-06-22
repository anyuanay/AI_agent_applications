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
│   └── context_graph.py       # Article 3: k-hop projection, relevance scoring, turns, eviction
├── ontologies/                # canonical SCIMA-OWL, one file per version
│   ├── scima_owl_v0_1.ttl     # Article 1 (8 classes, 12 properties, 5 axioms)
│   ├── scima_owl_v0_2.ttl     # Article 2 (12 classes, 20 properties, 8 axioms)
│   └── scima_owl_v0_5.ttl     # Article 3 (18 classes, 30 properties, 12 axioms)
└── tests/
    ├── test_article_01.py     # asserts v0.1 matches the Growth Tracker
    ├── test_article_02.py     # asserts v0.2 + SCIMA-KG population and queries
    └── test_article_03.py     # asserts v0.5 + context-graph projection and turns
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
