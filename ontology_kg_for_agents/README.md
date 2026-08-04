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
│   ├── kg_extraction.py       # Article 5: three-stage Extract/Map/Verify pipeline + five-check compliance gate
│   └── agent_search.py        # Article 10: schema-derived predicate sets, guided beam walk, belief-weighted ranking, non-stationary cache TTLs, cost model
├── ontologies/                # canonical SCIMA-OWL, one file per version
│   ├── scima_owl_v0_1.ttl     # Article 1 (8 classes, 12 properties, 5 axioms)
│   ├── scima_owl_v0_2.ttl     # Article 2 (12 classes, 20 properties, 8 axioms)
│   ├── scima_owl_v0_5.ttl     # Article 3 (18 classes, 30 properties, 12 axioms)
│   ├── scima_owl_v0_6.ttl     # Article 4 (26 classes, 34 properties, 15 axioms)
│   ├── scima_owl_v0_8.ttl     # Article 5 (30 classes, 41 properties, 18 axioms)
│   └── scima_owl_v1_0.ttl     # Articles 8 + 10 (43 classes, 53 properties, 21 axioms)
├── corpus/
│   ├── emergency_procedures.txt  # Article 4 source corpus
│   └── incident_report_I204.txt  # Article 5 source: 6-sentence incident narrative
└── tests/
    ├── test_article_01.py     # asserts v0.1 matches the Growth Tracker
    ├── test_article_02.py     # asserts v0.2 + SCIMA-KG population and queries
    ├── test_article_03.py     # asserts v0.5 + context-graph projection and turns
    ├── test_article_04.py     # asserts v0.6 + the learned emergency-response delta
    ├── test_article_05.py     # asserts v0.8 + the three-stage pipeline and compliance gate
    └── test_article_10.py     # asserts v1.0 + the derived predicate set, the walk, the caches, the cost table
```

As the series proceeds, new modules join `scima/` (belief updates, particle
filters, KG merging, GraphRAG) and new `scima_owl_vX_Y.ttl` files join
`ontologies/`, each with a matching test file. Not every article adds code.
Articles 6 through 9 are article-only installments; the map below says which
is which.

## Article to code map

| Article | Concept | Code |
|---------|---------|------|
| 1 | What is an ontology? | `scima/building_blocks.py`, `ontologies/scima_owl_v0_1.ttl` |
| 2 | Knowledge graphs, triples, SPARQL | `scima/knowledge_graph.py`, `ontologies/scima_owl_v0_2.ttl` |
| 3 | Context graphs, dynamic working memory | `scima/context_graph.py`, `ontologies/scima_owl_v0_5.ttl` |
| 4 | Extracting ontologies from sources | `scima/ontology_extraction.py`, `ontologies/scima_owl_v0_6.ttl` |
| 5 | Extracting ontology-compliant KGs from sources | `scima/kg_extraction.py`, `ontologies/scima_owl_v0_8.ttl` |
| 6 | Evaluating ontologies and knowledge graphs | article only, no new code: a synthesis/evaluation pass over the v0.8 artifacts of Articles 4-5 |
| 7 | Stochastic ontology and uncertainty in concept space | article only, no new code: time-indexed belief snapshots plus a transition law; the article's `BeliefProcess` snippet (predict = matrix power, correct = Bayes) is execution-verified inline |
| 8 | Ontology update rules, Bayesian and belief revision | article only, no new code: the `correct` operator in full (Bayesian updates, conjugate Beta prior), expansion/contraction/revision, update-vs-revise, AGM postulates, and OWL-Time + named-graph temporal versioning (introduces SCIMA-OWL v1.0); inline `bayesian_update` and TriG snippets only, runnable belief layer still pending |
| 9 | Age of Information, staleness, and forgetting in KGs | article only, no new code: AoI as a sawtooth freshness clock, entropy-based expiry, a forgetting factor derived from the transition law, per-predicate staleness functions, Ebbinghaus retention/compaction, and AoI-aware querying. The forgetting factor now has a home in code, since `agent_search.belief_weight` is that factor and `test_article_10.py` asserts it is the transition law in closed form |
| 10 | Knowledge graphs for cost-efficient agent search | `scima/agent_search.py`, `ontologies/scima_owl_v1_0.ttl` |
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

Extract an ontology-compliant KG from Incident Report I-204 through the
three-stage Extract/Map/Verify pipeline and watch the five-check compliance
gate admit, repair, reject, and quarantine (Article 5):

```bash
python -m scima.kg_extraction
# Stage 1: 6 sentences, 8 entity mentions
#          dep triples: 6, llm triples: 6
#          candidates: 5, negated (quarantined): 1
# Stage 2: ontology index: 30 classes, 41 properties
#          mapped triples: 5
# Stage 3: admitted: 2, repaired: 1, rejected: 2
#          type assertions: 4
#   [admit] scima:IncidentCommander_commander_diaz scima:commands scima:HazmatTeam_hazmatteam_alpha
#   [admit] scima:HazmatTeam_hazmatteam_alpha scima:dispatchedTo scima:HazMatSpill_incident_i204
#   [repaired] scima:SensorReading_reading_r1 scima:observedValue "47"^^xsd:integer
#   [reject:domain_violation] scima:WaterMain_watermain_7b scima:dispatchedTo scima:HazMatSpill_incident_i204
#   [reject:cardinality_violation] scima:IncidentCommander_commander_diaz scima:commands scima:HazmatTeam_hazmatteam_gamma

python -m scima.kg_extraction --corpus corpus/incident_report_I204.txt   # same run, explicit corpus
```

Articles 6 through 9 add no modules here. Article 6 evaluates the v0.8
artifacts produced above; the Uncertainty pillar then runs across three
article-only installments, the stochastic layer in Article 7 (time-indexed
belief snapshots, transition laws, the predict-correct tracing recursion),
Bayesian updates and belief revision with SCIMA-OWL v1.0 temporal versioning
in Article 8, and Age of Information, staleness, and transition-law-derived
forgetting in Article 9, all as worked, execution-verified examples in the
articles themselves. The runnable belief machinery is still pending. Article 8
does contribute its temporal-versioning vocabulary to `scima_owl_v1_0.ttl`,
which Article 10 then extends and uses.

Derive the predicates a dispatch walk may follow, rather than typing them out
by hand (Article 10, Section 4):

```bash
python -m scima.agent_search --predicates
# Loaded SCIMA-OWL v1.0: 43 classes, 53 properties, 21 axioms
#
# Schema search: paths from scima:Incident to scima:EmergencyVehicle within 4 hops
#   8 of 43 classes lie on a route
#   10 of 53 properties can lie on such a route (19 percent of the vocabulary)
#     scima:adjacentTo  <- carries data in this scene
#     scima:affects  <- carries data in this scene
#     scima:assignedTo
#     scima:connectedTo
#     scima:controlledBy
#     scima:poweredBy
#     scima:respondsTo
#     scima:servedByStation  <- carries data in this scene
#     scima:stationedAt  <- carries data in this scene
#     scima:withinZone  <- carries data in this scene
```

Run the relation-guided beam walk for incident I-204 and watch the admissible
straight-line bound prune a branch without risking the answer:

```bash
python -m scima.agent_search --dispatch I-204
# Dispatch I-204: nearest available ambulance
#   full scan            : 800 records
#   blind 3-hop BFS      : 8 + 56 + 392 = 456 nodes
#   guided walk          : scima:AMB_17 at 4.2 min, 12 nodes visited
#   guided + bound (A*)  : scima:AMB_17 at 4.2 min, 11 nodes visited
#   frontier by hop      :
#     hop 1: scima:ControlZone_Z7, scima:RoadSegment_Main_St_NB
#     hop 2: scima:Station_Crosstown, scima:Station_Riverside, scima:Station_Hilltop
#     hop 3: scima:FIRE_12
```

Read each predicate's cache lifetime off its measured half-life, and price the
stale answers the cache will occasionally serve:

```bash
python -m scima.agent_search --cache
# Cache lifetime per predicate, at a 0.80 confidence floor:
#   scima:hasStatus          29 s
#   scima:hasLocation        97 s
#   scima:servedByStation    9.7 days
#   scima:stationedAt        29.0 days
#   scima:hasSpeedLimit      no expiry (invalidate on write)
#
#   floor 0.80: status TTL 29.0 s, hit rate 74.1%, 0.80 fresh hops of 12
#               (15x less graph work), stale-answer penalty 11 s
#   floor 0.50: status TTL 90.0 s, hit rate 91.7%, 0.27 fresh hops of 12
#               (45x less graph work), stale-answer penalty 31 s
```

A fleet-wide average penalty cannot rank one vehicle against another, so weight
each candidate by the age of its own status evidence and rank by expected time,
`E[T] = eta + (1 - e^(-lambda * age)) * 3.5 min` (Article 10, Section 5):

```bash
python -m scima.agent_search --rank
# Candidates ranked by expected time, E[T] = eta + (1 - w) x 3.5 min:
#   scima:AMB_17   eta  4.2 min   status  25.0 s old   w 0.825   E[T]  4.81 min
#   scima:AMB_31   eta  5.0 min   status   3.0 s old   w 0.977   E[T]  5.08 min
#   scima:AMB_09   eta  6.1 min   status  31.0 s old   w 0.788   E[T]  6.84 min
#
#   Now let AMB-17's status evidence age to 60 seconds:
#   scima:AMB_31   eta  5.0 min   status   3.0 s old   w 0.977   E[T]  5.08 min
#   scima:AMB_17   eta  4.2 min   status  60.0 s old   w 0.630   E[T]  5.50 min
#   scima:AMB_09   eta  6.1 min   status  31.0 s old   w 0.788   E[T]  6.84 min
#   The ranking flips, and nothing about the road network moved.
#
#   Joint confidence over the live hops: 0.825 x 0.944 = 0.779
#   So the answer is AMB-17 at about 78 percent, not AMB-17 as a fact.
```

The measured half-life is itself an estimate of a moving quantity. Show what a
frozen lifetime costs when the flip rate triples, and why the confidence floor
rather than the lifetime is the knob to hold fixed:

```bash
python -m scima.agent_search --surge
# A surge multiplies the status flip rate by 3, so the half-life falls 90 s -> 30 s.
#   lifetime frozen at 29.0 s : 14.2% stale, 30 s penalty
#   lifetime re-derived      : 9.7 s -> 5.3% stale, 11 s penalty
#   the floor is the invariant: (1 - sqrt(0.80)) / 2 = 5.3%, with lambda cancelled out
#   normal: status hit rate 74.1%, 0.80 fresh hops of 12 (15.1x less graph work)
#   surge : status hit rate 22.3%, 2.35 fresh hops of 12 (5.1x less graph work)
```

Price every search design in tokens, dollars, and seconds:

```bash
python -m scima.agent_search --costs
# scan every vehicle, one call each      800  800 call(s)  39,952,000 tok  $120.576  720.00 s
# pull all 800 records in one call       800    1 call(s)      98,000 tok  $  0.295    0.90 s
# vector search, top 40                   40    1 call(s)       6,800 tok  $  0.022    0.94 s
# blind three-hop BFS                    456    1 call(s)      56,720 tok  $  0.171    3.18 s
# relation-guided walk, agent-side        12   12 call(s)      31,920 tok  $  0.107   10.80 s
# relation-guided walk, in the engine     12    1 call(s)       2,930 tok  $  0.010    0.96 s
# relation-guided walk, warm cache      0.80    1 call(s)       2,930 tok  $  0.010    0.90 s
#
#   loop vs engine: 13,635x tokens, 12,070x dollars, 750x latency
#   decision budget 2 s: 4 of 7 designs meet it
```

Run the tests:

```bash
pytest
# 160 passed
```

## Design notes

- **The Turtle file is the source of truth for tooling.** `building_blocks.py`
  is the source of truth for the *prose*: a readable Python mirror of the same
  content, so the article's vocabulary can be imported and tested.
- **The Growth Tracker is the contract.** `tests/test_article_01.py` asserts
  v0.1 has exactly 8 classes / 12 properties / 5 axioms, matching the table in
  the series plan. Every later version gets its own assertions, up to
  `tests/test_article_10.py` on v1.0's 43 / 53 / 21.
- **OWL DL throughout.** SCIMA-OWL stays inside the decidable OWL DL profile so
  reasoning (Articles 8, 12) is guaranteed to terminate.
- **Every published number is a test.** The articles quote worked figures, so
  the figures are assertions rather than prose. `test_article_10.py` pins the
  39,952,000-token loop, the $120.58 bill, the 12-node walk, the 29-second
  cache lifetime, the 15x reduction in graph work, the 4.81-against-5.08
  expected-time ranking and the flip to 5.50, the 0.779 path confidence, and
  the surge that drives stale dispatches from 5.3 to 14.2 percent. An edit that
  changes the behaviour fails the suite instead of quietly making an article
  wrong.
- **Nothing that describes a rate is a constant.** Every decay rate takes a
  `rate_multiplier`, because the logged half-life is an estimate of a quantity
  that itself moves and holding it fixed is an unstated stationarity claim.
  Where a lifetime is derived from a rate, the derivation is a function rather
  than a literal, so a regime shift moves it. `stale_dispatch_rate` carries the
  result that makes this concrete: with the lifetime re-derived, the error rate
  is `(1 - sqrt(floor)) / 2` with lambda cancelled out, so the confidence floor
  is the knob to hold fixed and the lifetime is a consequence.
- **Beliefs age, so queries weight rather than threshold.** `belief_weight` is
  continuous, and `rank_candidates` ranks by expected time instead of filtering
  on a freshness cutoff. A hard cutoff would let a 0.02 difference in belief
  decide everything, which is the failure the articles criticise vector search
  for. `path_confidence` reports what a multi-hop answer is actually worth.
- **Small deterministic scenes, real logic.** `DispatchScene` holds one
  incident, three stations, and six vehicles rather than a 250K-triple city, so
  the demo is fast and reproducible. The walk, the schema derivation, and the
  cache logic are the same code that would run against the full graph; where a
  figure comes from the city-scale graph instead (the 456-node blind frontier),
  it is computed analytically and labelled as an estimate.
