"""SCIMA: the SmartCity Infrastructure Management Agent.

This package is the code that backs the article series
"Ontology and Knowledge Graphs for Intelligent Agents". One coherent
SCIMA implementation grows across the series; each article draws from
(and extends) the modules here rather than carrying disconnected
snippets.

Module map (grows with the series):
    ontology        loading and inspecting versioned SCIMA-OWL files (Article 1+)
    building_blocks the four building blocks: classes, individuals,
                    properties, axioms (Article 1)
    knowledge_graph populate SCIMA-OWL into a triple store, named graphs,
                    SPARQL helpers, geo queries (Article 2)
    context_graph   k-hop projection, relevance scoring, turn-by-turn
                    eviction and write-back (Article 3)
    ontology_extraction  the seven-stage Scope/Surface/Sort/Name/Salience/
                    Structure/Review pipeline that learns new classes from
                    text, with the type/instance gate and the agentic RITE
                    review loop (Article 4)
    kg_extraction   the three-stage Extract/Map/Verify-Admit pipeline that
                    populates an A-Box from source text against a fixed
                    ontology contract (Article 5)
"""

__all__ = [
    "ontology",
    "building_blocks",
    "knowledge_graph",
    "context_graph",
    "ontology_extraction",
    "kg_extraction",
]

ONTOLOGY_VERSIONS = {
    "v0.1": "scima_owl_v0_1.ttl",  # Article 1: core class hierarchy
    "v0.2": "scima_owl_v0_2.ttl",  # Article 2: sensing vocabulary for KG population
    "v0.5": "scima_owl_v0_5.ttl",  # Article 3: agent, goal, context-graph vocabulary
    "v0.6": "scima_owl_v0_6.ttl",  # Article 4: emergency-response vocabulary extracted from sources
    "v0.8": "scima_owl_v0_8.ttl",  # Article 5: KG extraction vocabulary with functional constraints
}
