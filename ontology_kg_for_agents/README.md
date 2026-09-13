# ontology_kg_for_agents

Code that backs the article series **"Ontology and Knowledge Graphs for
Intelligent Agents"** (see `../ontology_KG_agents/ontology_KG_agent_series_plan.md`).

SCIMA is the running city-service example for the series. This package contains versioned ontologies and teaching modules for extraction, search, planning, evolution, and belief utilities. The shipped ontology files extend through v1.8. Later schema versions remain planning targets.

## LLM agents and stochastic memory

[Article 13](../ontology_KG_agents/article_13_agent_belief_kgs.html) presents **LLM Agent Beliefs, Stochastic Memory, and Source-Actor Architecture**. Each agent has a foundation model, role, tools, and persistent memory. A stochastic knowledge graph records its task beliefs, uncertainty, evidence, and dependencies.

The managing LLM agent maintains shared ontology definitions, city structure, identifiers, responsibilities, and reviewed assessments. It seeds role-specific actor memories from a versioned snapshot. Actors update memory through accepted evidence and send proposed changes for review. Model weights remain unchanged during these memory updates.

Retrieved context is the part of memory supplied to the model for a task. Memory storage, retrieval, action selection, and model training are separate operations. Evidence identifiers survive copying between actors and the manager.

The architecture adapts SOLIR’s Source-Actor Knowledge Diffusion work. It includes source probing, targeted transmissions, actor updates, and peer exchange under a communication budget. The [SOLIR reference note](../../SOLIR_stochastic_ontology/agent_memory_reference.md) records the source definitions and their scope.

### Implementation status

The article contains a worked Bayesian update and a JavaScript memory-exchange walkthrough. `scima/source_actor_memory.py` implements its six stages in Python. The stages cover initialization, a local report, manager review, probing, a targeted cast, and duplicate peer evidence.

The Python module stores role-specific graph triples, claim scope, source snapshot identifiers, evidence, and assessment history. JSON snapshots support saving and loading memory. Evidence exchange applies missing reports to the recipient’s belief and preserves its local reports. A probe selects missing evidence under a report-count budget.

This teaching model assumes a fixed bridge state within one interval. Distinct reports require conditional independence given that state. Report error rates are supplied assumptions. Validation checks fields, scope, timing, and identity collisions. It does not authenticate sources or establish report truth.

The module includes a toy route policy and records its supporting memory version. It has no LLM calls, model training, world-state transitions, structural reconciliation, or complete SAKD runtime. The budget counts transmitted reports and does not price probes or bytes. JSON loading expects snapshots written by this module.

`scima/belief_decisions.py` and `scima/agent_beliefs.py` remain supplemental utilities from the earlier treatment. The v1.8 schema and its tests remain supplemental. The new memory example adds no ontology version.

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
│   ├── agent_search.py        # Article 10: schema-derived predicate sets, guided beam walk, belief-weighted ranking, non-stationary cache TTLs, cost model
│   ├── agent_planning.py      # Article 11: goals as ontology content, OWL-typed grounding, method decomposition, preconditions as beliefs at future times, plan shelf life, axiom-derived mutex
│   ├── ontology_evolution.py  # Article 12: derived promotion and retirement thresholds, axiom weights as seasonal processes, safe vs backward-reaching additions, the compatibility gate
│   ├── source_actor_memory.py # Article 13: saved memory, Bayes, probes, casts, and evidence identity
│   ├── agent_beliefs.py       # belief utilities and supplemental examples
│   └── belief_decisions.py    # Supplemental bridge assessments and pending-plan checks
├── ontologies/                # canonical SCIMA-OWL, one file per version
│   ├── scima_owl_v0_1.ttl     # Article 1 (8 classes, 12 properties, 5 axioms)
│   ├── scima_owl_v0_2.ttl     # Article 2 (12 classes, 20 properties, 8 axioms)
│   ├── scima_owl_v0_5.ttl     # Article 3 (18 classes, 30 properties, 12 axioms)
│   ├── scima_owl_v0_6.ttl     # Article 4 (26 classes, 34 properties, 15 axioms)
│   ├── scima_owl_v0_8.ttl     # Article 5 (30 classes, 41 properties, 18 axioms)
│   ├── scima_owl_v1_0.ttl     # Articles 8 + 10 (43 classes, 53 properties, 21 axioms)
│   ├── scima_owl_v1_1.ttl     # Article 11 (62 classes, 85 properties, 32 axioms)
│   ├── scima_owl_v1_5.ttl     # Article 12 (74 classes, 100 properties, 41 axioms)
│   └── scima_owl_v1_8.ttl     # Supplemental v1.8 (80 classes, 110 properties, 46 axioms)
├── corpus/
│   ├── emergency_procedures.txt  # Article 4 source corpus
│   └── incident_report_I204.txt  # Article 5 source: 6-sentence incident narrative
└── tests/
    ├── test_article_01.py     # asserts v0.1 matches the Growth Tracker
    ├── test_article_02.py     # asserts v0.2 + SCIMA-KG population and queries
    ├── test_article_03.py     # asserts v0.5 + context-graph projection and turns
    ├── test_article_04.py     # asserts v0.6 + the learned emergency-response delta
    ├── test_article_05.py     # asserts v0.8 + the three-stage pipeline and compliance gate
    ├── test_article_10.py     # asserts v1.0 + the derived predicate set, the walk, the caches, the cost table
    ├── test_article_11.py     # asserts v1.1 + the grounding cascade, the candidate funnel, the plan ranking reversal, shelf life, mutex, quarantine
    ├── test_article_12.py     # asserts v1.5 + the derived thresholds, the pooled-weight error, the stale type list, the compatibility gate
    ├── test_source_actor_memory.py # Article 13 exchange, persistence, validation, and local evidence
    ├── test_article_13.py     # v1.8 schema and supplemental belief utility checks
    └── test_belief_decisions.py # River Bridge arithmetic, evidence copies, and plan dependencies
```

As the series proceeds, new modules join `scima/` (particle filters, KG
merging, GraphRAG) and new `scima_owl_vX_Y.ttl` files join
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
| 8 | Ontology update rules, Bayesian and belief revision | article only, no new code: the `correct` operator in full (Bayesian updates, conjugate Beta prior), expansion/contraction/revision, update-vs-revise, AGM postulates, and OWL-Time + named-graph temporal versioning (introduces SCIMA-OWL v1.0); inline `bayesian_update` and TriG snippets. Article 13 adds a fixed-state Bayesian memory example |
| 9 | Age of Information, staleness, and forgetting in KGs | article only, no new code: AoI as a sawtooth freshness clock, entropy-based expiry, a forgetting factor derived from the transition law, per-predicate staleness functions, Ebbinghaus retention/compaction, and AoI-aware querying. The forgetting factor now has a home in code, since `agent_search.belief_weight` is that factor and `test_article_10.py` asserts it is the transition law in closed form |
| 10 | Knowledge graphs for cost-efficient agent search | `scima/agent_search.py`, `ontologies/scima_owl_v1_0.ttl` |
| 11 | Ontology for agent goal achievement and planning with class hierarchy | `scima/agent_planning.py`, `ontologies/scima_owl_v1_1.ttl` |
| 12 | Continual learning, ontology and knowledge graph evolution | `scima/ontology_evolution.py`, `ontologies/scima_owl_v1_5.ttl` |
| 13 | [LLM Agent Beliefs, Stochastic Memory, and Source-Actor Architecture](../ontology_KG_agents/article_13_agent_belief_kgs.html) | `scima/source_actor_memory.py`, `tests/test_source_actor_memory.py`. Six-stage teaching example with JSON snapshots. The integrated LLM runtime remains planned. |
| Supplemental | Earlier bridge and intersection examples | `scima/belief_decisions.py`, `scima/agent_beliefs.py`, `ontologies/scima_owl_v1_8.ttl` |
| 14, planned | Stochastic memory prediction and correction | Exact small-case update and particle filtering. No module shipped. |
| 15, planned | Source-actor memory coordination | Versioned initialization, evidence exchange, duplicate-safe reconciliation. Article 13 supplies a small exchange example. General coordination remains planned. |
| 16, planned | GraphRAG over stochastic memory | Context construction, output checks, and validated memory writes. No module shipped. |
| 17, planned | Cross-domain memory alignment | Versioned mappings with uncertainty and evidence identity. No module shipped. |
| 18, planned | Integrated LLM agent case | Matched baselines and trace-based evaluation. No integrated runtime shipped. |

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
articles themselves. Article 13 adds a fixed-state Bayesian memory update. General prediction and correction remain planned. Article 8
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

Ground an action schema through OWL classes rather than a flat type list, and
watch the three filters people constantly conflate do three different jobs
(Article 11, Sections 2 and 3):

```bash
python -m scima.agent_planning --ground
# Loaded SCIMA-OWL v1.1: 62 classes, 85 properties, 32 axioms
# Transport deadline: 20 min protocol - 4 min already spent = 16.0 min from tasking
#
# Grounding DispatchVehicle(?v, ?i), three ways:
#   untyped, every node x every node : 3,969,000,000
#   parameters typed by OWL class    :         4,800   (826,875x fewer)
#   precondition run as a query first:             7   (686x fewer)
#   total reduction                  : 567,000,000x
#
# The three filters, which answer three different questions:
#   type-correct (class hierarchy) : 7  AMB_17, AMB_22, FIRE_12, AMB_04, AMB_09, AMB_31, AIR_3
#   applicable   (precondition)    : 5  AMB_17, FIRE_12, AMB_09, AMB_31, AIR_3
#   useful       (axiom on goal)   : 4  AMB_17, AMB_09, AMB_31, AIR_3
#   same query, hard-coded 6-type list: 3  AMB_17, AMB_09, AMB_31
#   silently dropped: AIR_3   <- the only plan that meets the deadline
```

Decompose the goal, inherit methods down the class hierarchy, and derive the
mutual exclusions from axioms instead of tabulating them:

```bash
python -m scima.agent_planning --decompose
# Methods attach to a goal class, so subclasses inherit them:
#   IncidentResolutionGoal             decomposable  M1
#   PatientTransportGoal               decomposable  M2
#   WaterServiceRestorationGoal        decomposable  M3
#   PowerServiceRestorationGoal        decomposable  M3
#   TrafficClearanceGoal               decomposable  M4
#   PublicNotificationGoal             leaf          -
#   4 methods cover 6 goal classes, because M3 sits at the parent
#
# Schedule (partial order, earliest start):
#   Broadcast      start   0.0  finish   1.5   after: nothing
#   Dispatch       start   0.0  finish   4.2   after: nothing
#   DivertTraffic  start   0.0  finish   4.5   after: nothing
#   IsolateValve   start   0.0  finish   6.0   after: nothing
#   Load           start   4.2  finish   7.2   after: Dispatch
#   Transport      start   7.2  finish  12.6   after: Load
#   PumpOut        start   6.0  finish  18.0   after: IsolateValve, DivertTraffic
#   critical path 18.0 min, slack 2.0 against the 20.0 deadline
#
# Mutex over 8 proposed actions (28 candidate pairs), derived not tabulated:
#   Dispatch(AMB-17) x Dispatch(AIR-3)
#       because maxQualifiedCardinality 1 on scima:assignedResource
#   IsolateValve(V-12) x OpenHydrant(V-12)
#       because functional property scima:hasValveState
```

Price the three candidate plans two ways and watch the ranking reverse, which
is the whole article in one table (Article 11, Section 5):

```bash
python -m scima.agent_planning --plans
# Three plans for the transport goal, priced two ways:
#   plan                                            nominal  P(succ)     E[T]  verdict
#   A: AMB-17, checked once at plan time              12.60    0.374    16.71  misses by 0.71
#   B: AIR-3, checked once at plan time               13.00    0.632    14.99  meets, slack 1.01
#   C: AMB-17 with four verification steps            13.13    0.932    13.54  meets, slack 2.46
#
#   nominal order : A, B, C
#   expected order: C, B, A
#   The ranking reverses. The fastest plan on paper is the one that misses.
#
#   Same 0.20 min camera read, refreshed at plan time : 0.955 min saved
#   Same 0.20 min camera read, refreshed at execution : 2.511 min saved  (2.63x better)
```

Find out how long a finished plan stays committable, and which single
precondition is worth confirming:

```bash
python -m scima.agent_planning --shelf
# Plan B joint confidence 0.632, commit floor 0.55
#   shelf life as written        : 16.9 s
#   dominant term                : AIR-3 available at 93.0% of the total decay rate
#   after one radio call         : 375 s (6.25 min), 22.2x longer
```

Handle a precondition that is neither true nor false, because the fact it needs
failed the admission gate and is sitting in quarantine (Article 11, Section 7):

```bash
python -m scima.agent_planning --quarantine
# AIR-3's stationedAt failed the admission gate (two values, and the property is functional).
#   extraction confidence 0.88, observed 400 s ago, half-life 90 days
#   precondition belief = 0.880  (UNKNOWN, not FALSE)
#   breakeven belief    = 1 - 0.4 / 6.0 = 0.9333
#   decision            = RESOLVE
#   carry the risk : 15.71 min
#   resolve first  : 15.39 min   (wins by 0.32, and removes the variance)
#
#   A planner treating absence as falsehood never gets this choice.
#   It drops AIR-3 and reports a plan that misses the deadline.
```

Show what a surge does to a plan built on stored availability, and why a regime
is a per-predicate map rather than one multiplier:

```bash
python -m scima.agent_planning --surge
# A surge multiplies the status flip rate by 3, so the half-life falls 90 s -> 30 s.
#   lambda 0.007702 -> 0.023105 per second
#   The regime applies per predicate, since a surge does not change
#   how fast obstructions appear on a helipad:
#     AIR-3 available            scima:hasStatus              w 0.926 -> 0.794
#     landing zone clear         scima:hasLandingClearance    w 0.822 -> 0.822
#     Mercy General accepting    scima:hasDivertStatus        w 0.831 -> 0.831
#   plan B joint confidence 0.632 -> 0.542, floor 0.55
#   verdict: BELOW THE FLOOR at the moment it is written
#   A system holding the peacetime rate reports 0.632, commits, and raises no alarm.
```

Move the schema instead of the facts. Concept birth, where the floor is derived
from the cost of a homeless mention and the cost of retiring a class that does
not last (Article 12, Section 2):

```bash
python -m scima.ontology_evolution --promote
# promotion floor m* = (1 - q) x C_retire / (N x C_late) = 0.0844
#   month  9  m = 0.061   waiting costs $     976
#   month 12  m = 0.087   waiting costs $   1,392  ADD THE CLASS
#   cost of being wrong = (1 - 0.85) x $9,000 = $1,350
```

Concept death, the same argument backward, where a rare class earns a long wait
(Article 12, Section 2):

```bash
python -m scima.ontology_evolution --retire
# healthy rate 0.50/month -> wait   9.2 months; after 11 silent months: retire
# healthy rate 0.05/month -> wait  92.1 months; after 11 silent months: keep watching
```

Read an axiom weight as a process rather than a number, and see what pooling the
year costs (Article 12, Section 3):

```bash
python -m scima.ontology_evolution --weights
# pooled weight over the year = 0.7525, action floor = 0.7
#   winter  weight 0.87   pooled says act = True   season says act = True
#   summer  weight 0.61   pooled says act = True   season says act = False  <-- acts on a rule the season does not support
# retire the axiom after 4 straight quarters under the floor
```

Watch a hand-typed list of type names lose the new class without reporting
anything (Article 12, Section 4):

```bash
python -m scima.ontology_evolution --stale-list
# hand-typed list  -> 4 types, no error reported
# subclass walk    -> 12 types
# missed by the list: scima:ChargingPoint, scima:DroneCorridorSegment, scima:EVChargingStation, ...
```

Earn the right to claim `owl:backwardCompatibleWith` (Article 12, Sections 4
and 6):

```bash
python -m scima.ontology_evolution --gate
# repair=False FAIL: 147 queries replayed, 0 rows lost, 3 triples flagged, 0 repaired
#    flagged: scima:EVCS_02, scima:EVCS_04, scima:EVCS_05
# repair=True  PASS: 147 queries replayed, 0 rows lost, 3 triples flagged, 3 repaired
```

Run the Article 13 memory exchange and save each stage.

```bash
python -m scima.source_actor_memory --output build/article_13_memory
# 0. initialization: source=0.800 north=0.800 dispatch=0.800 dispatch=cross
# 1. local report: source=0.800 north=0.308 dispatch=0.800 dispatch=cross
# 2. manager review: source=0.308 north=0.308 dispatch=0.800 dispatch=cross
# 3. probe dispatch: source=0.308 north=0.308 dispatch=0.800 dispatch=cross
# 4. targeted cast: source=0.308 north=0.308 dispatch=0.308 dispatch=detour
# 5. duplicate peer report: source=0.308 north=0.308 dispatch=0.308 dispatch=detour
```

Each numbered directory contains source, north, and dispatch snapshots plus a decision record.
Omit `--output` to print the stages without saving files.

```python
from pathlib import Path
from scima.source_actor_memory import Memory, route_decision

memory = Memory.load(Path("build/article_13_memory/05/dispatch.json"))
assert route_decision(memory)["action"] == "detour"
assert memory.version == 1
```

The toy policy permits crossing at an open probability of at least 0.7.
This threshold is an added teaching assumption. The article specifies no numerical action threshold.
The decision record retains the claim dependency and retrieved context. Choosing a detour adds no bridge evidence.

Run the supplemental River Bridge decision demo from the earlier Article 13 treatment.

```bash
python -m scima.belief_decisions
# Fresh: p=0.900, publish loss=8.00, withhold loss=18.00, publish
# Base: p=0.700, publish loss=24.00, withhold loss=14.00, withhold
# Disruption: p=0.600, publish loss=32.00, withhold loss=12.00, withhold
# Observation groups: 1
# update: recheck ['ambulance across River Bridge']
# revision: recheck ['ambulance across River Bridge']
```

The rates and losses are teaching assumptions. The demo groups copied messages by observation ID and follows claim dependencies to pending plans.
Both update and revision require a check of the pending bridge route.

The `agent_beliefs.py` command-line scenes remain supplemental examples. Their intersection scene is separate from the rewritten article’s memory walkthrough.
Its `reopened_by` helper covers revision history, with no check of pending plans under world updates.
Use `belief_decisions.affected_plans` for that check. The demo has no access-control server or evidence-verification service.

Run the tests:

```bash
pytest
# 302 passed on 2026-09-12
```

## Design notes

- **Foundation models and graph memory have distinct roles.** The model interprets requests and supports action proposals. The graph stores task beliefs.
- **Learning requires a committed memory change.** A report in a conversation is not proof of persistent storage. Retrieval must expose the changed belief to later requests.
- **The manager supplies shared structure.** Role-specific initial memories preserve the source version, claim scope, and evidence links.
- **Shared evidence creates dependence.** Copying an inherited belief or forwarding a report adds no independent observation. Reconciliation must track original evidence.
- **Manager authority does not establish certainty.** Shared assessments need uncertainty and source records. Exact source knowledge is a stated teaching assumption.
- **Documentation distinguishes the target system from teaching utilities.** The integrated source-actor runtime and remaining article modules are planned.

- **The Turtle file is the source of truth for tooling.** `building_blocks.py`
  is the source of truth for the *prose*: a readable Python mirror of the same
  content, so the article's vocabulary can be imported and tested.
- **The Growth Tracker is the contract.** `tests/test_article_01.py` asserts
  v0.1 has exactly 8 classes / 12 properties / 5 axioms, matching the table in
  the series plan. Every later version gets its own assertions, up to
  `tests/test_article_13.py` on v1.8's 80 / 110 / 46. `axiom_count` grew in
  v1.1 to include class-expression axioms (restrictions and defined classes),
  since that is the first version to use them; a test pins every earlier
  version's total to prove the change moved nothing.
- **OWL DL throughout.** SCIMA-OWL stays inside the decidable OWL DL profile so
  reasoning (Articles 8, 12) is guaranteed to terminate.
- **Package examples have numerical checks.** Existing tests cover the Python demonstrations.
  `test_source_actor_memory.py` checks the revised Article 13 calculation, six exchange stages, saved snapshots, scope validation, and preservation of local evidence. `test_article_10.py` pins the
  39,952,000-token loop, the $120.58 bill, the 12-node walk, the 29-second
  cache lifetime, the 15x reduction in graph work, the 4.81-against-5.08
  expected-time ranking and the flip to 5.50, the 0.779 path confidence, and
  the surge that drives stale dispatches from 5.3 to 14.2 percent.
  `test_belief_decisions.py` checks the bridge probabilities, publication losses,
  grouping of copied evidence, and plan dependencies under both operators.
  `test_article_13.py` retains the schema and supplemental utility checks.
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
- **A rate belongs to a (class, predicate) pair, and a regime is a map.**
  `CHANGE_RATE_PROFILES` is keyed on both, because a maintenance crew's
  availability does not flip on an ambulance's clock even though both facts use
  `hasStatus`. `half_life_for` walks up the class hierarchy, so `AirAmbulance`
  inherits `Ambulance`'s 90 seconds without declaring anything. For the same
  reason `SURGE_REGIME` maps a predicate to its multiplier rather than being one
  number, since a surge changes how fast vehicles flip and does nothing to how
  fast obstructions appear on a helipad. An earlier draft applied one multiplier to
  every precondition, and `test_the_surge_leaves_the_slow_preconditions_alone`
  is what caught it.
- **A precondition is a claim about the future, so it is priced there.**
  `Precondition.confidence` uses `evidence_age + until_exec`, not just the
  evidence age, which is the correction the whole of Article 11 turns on. It is
  also why `plan_shelf_life` exists, because a finished plan decays while it
  waits to be committed and `CandidatePlan.shelf_life_s` says how long it has.
- **Absent is not false.** `QuarantinedFact` carries an extraction confidence
  and an age, so a precondition whose fact failed the admission gate returns
  UNKNOWN with a belief instead of FALSE. `breakeven_belief` then decides
  whether to resolve it or carry the risk. Treating absence as falsehood is what
  silently drops the only plan that meets the deadline.
- **The schema is append-only, and a retirement is a deprecation.**
  `scima_owl_v1_5.ttl` never renames a class and never deletes one.
  `scima:EvacuationZone` stays an `owl:Class` with `owl:deprecated true` and a
  `scima:replacedBy` pointer, because thousands of older triples name it and
  have to keep resolving. `test_nothing_v1_1_declared_was_removed` asserts the
  rule directly. Renaming or re-parenting in place is the ontology version of
  catastrophic forgetting, since it changes the meaning of every past triple
  with nothing reporting a loss.
- **A schema change reaches backward, so re-reasoning is not optional.**
  `classify_addition` sorts every edit into one that only grows the schema and
  one that forbids something. The `owl:maxCardinality 1` on `scima:hasOperator`
  condemns three stations whose triples passed the admission gate on the day
  they arrived, which is why `compatibility_gate` refuses to pass until they are
  repaired. Checking facts once, at ingestion, is not enough when the rules can
  move afterward. Widening a range is safe and narrowing one is not, which is
  why `scima:designates` could be widened to `scima:ControlZone` in the same
  release.
- **Promotion and retirement thresholds are outputs, not settings.**
  `promotion_floor` is `(1 - q) x C_retire / (N x C_late)`, so changing the cost
  of retiring a class moves the floor with no edit anywhere.
  `silence_before_retiring` is `ln(1/alpha) / r`, so a rare-but-healthy class
  earns a long wait on its own, and `quarters_before_retiring_axiom` is
  `ln(alpha) / ln(p)`, so a noisier estimate waits longer. Tests assert the
  direction of each response rather than only the value.
- **Belief holders and graph permissions are separate.** A named graph can
  identify an assessment. The application records its holder and enforces access permissions.
  Graph naming supplies no access control or belief semantics.
- **The transition model states its assumptions.** The bridge example uses a
  symmetric binary process that approaches a probability of one half.
  Its half-life measures decay of the difference from that midpoint.
  It does not measure time between closures. The disruption regime changes that half-life.
- **Copies preserve their source observation.** `source_groups` groups messages
  by observation ID. It does not treat message count as independent evidence.
  Different IDs do not prove independence without a source history.
- **Pending plans need checks under both operators.** `affected_plans` follows
  direct and derived dependencies under updates and revisions.
  Revision affects the historical explanation. A world update can make a pending route unusable.
- **OWL restrictions do not validate missing fields.** An unnamed value can
  satisfy an existential requirement under open-world semantics.
  The application must validate explicit frame links and graph-holder records.
  The current tests count schema axioms. They do not implement a complete OWL reasoner or admission validator.
- **Divergence needs an action model for decision costs.** The supplemental
  `chance_they_act_apart` helper computes total variation distance between distributions.
  That distance is not a general probability of incompatible actions.
  Its `worth_a_message` result is a heuristic without a specified action model.
  The supplemental bridge demo compares expected losses for a stated publication policy.

- **Small deterministic scenes, real logic.** `DispatchScene` holds one
  incident, three stations, and six vehicles rather than a 250K-triple city, so
  the demo is fast and reproducible. The walk, the schema derivation, and the
  cache logic are the same code that would run against the full graph; where a
  figure comes from the city-scale graph instead (the 456-node blind frontier),
  it is computed analytically and labelled as an estimate.
