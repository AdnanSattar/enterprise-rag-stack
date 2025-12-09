"""
Graph-RAG Integration Module (Advanced).

For multi-hop and entity-centric queries, combine knowledge graph with dense retrieval.

Pattern:
1. Entity extraction and linking populates a graph
2. Graph traversal finds connected evidence 2-3 hops away
3. Convert graph subgraphs to natural-language summaries for context

Example query: "Which products are impacted by delayed contracts in APAC?"
Requires:
1. Finding contracts with status="Delayed"
2. Filtering by region="APAC" (via customer relationship)
3. Following PERTAINS_TO relation to get products

Usage:
    from graph_rag import GraphRAGRetriever, SimpleKnowledgeGraph

    graph = SimpleKnowledgeGraph()
    retriever = GraphRAGRetriever(graph, vector_retriever)
    results = retriever.retrieve(query)
"""

from .entity_linking import EntityExtractor, EntityLinker
from .graph_builder import Entity, Relation, SimpleKnowledgeGraph
from .graph_traversal import GraphTraverser, TraversalResult
from .kg_summarizer import subgraph_to_text, summarize_path

__all__ = [
    "EntityExtractor",
    "EntityLinker",
    "SimpleKnowledgeGraph",
    "Entity",
    "Relation",
    "GraphTraverser",
    "TraversalResult",
    "subgraph_to_text",
    "summarize_path",
]
