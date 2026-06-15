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
"""

__all__ = ["ontology", "building_blocks", "knowledge_graph"]

ONTOLOGY_VERSIONS = {
    "v0.1": "scima_owl_v0_1.ttl",  # Article 1: core class hierarchy
    "v0.2": "scima_owl_v0_2.ttl",  # Article 2: sensing vocabulary for KG population
}
