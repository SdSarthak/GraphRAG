"""
GraphRAG - Graph-based Retrieval Augmented Generation
A sophisticated RAG system using knowledge graphs for enhanced information retrieval
"""

import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphRAG:
    """Main GraphRAG system combining knowledge graphs with RAG"""
    
    def __init__(self, 
                 llm_model: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize GraphRAG system
        
        Args:
            llm_model: Language model for generation
            embedding_model: Model for creating embeddings
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.graph = None
        self.vector_store = None
        logger.info(f"Initialized GraphRAG with LLM: {llm_model}")
    
    def build_knowledge_graph(self, documents: List[str]) -> None:
        """
        Build knowledge graph from documents
        
        Args:
            documents: List of text documents
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        # Extract entities and relationships
        entities = []
        relationships = []
        
        for doc in documents:
            # Extract entities (nouns, named entities)
            doc_entities = self._extract_entities(doc)
            entities.extend(doc_entities)
            
            # Extract relationships between entities
            doc_relationships = self._extract_relationships(doc, doc_entities)
            relationships.extend(doc_relationships)
        
        # Build graph structure
        self.graph = {
            'entities': list(set(entities)),
            'relationships': relationships,
            'documents': documents
        }
        logger.info(f"Built graph with {len(self.graph['entities'])} entities and {len(relationships)} relationships")
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simplified implementation)"""
        # Simple word-based extraction (in production use NER models)
        words = text.lower().split()
        # Filter common words and extract potential entities
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        entities = [word for word in words if word not in stopwords and len(word) > 3]
        return entities[:10]  # Limit entities per document
    
    def _extract_relationships(self, text: str, entities: List[str]) -> List[Dict[str, str]]:
        """Extract relationships between entities (simplified)"""
        relationships = []
        # Simple co-occurrence based relationships
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1 in text and entity2 in text:
                    relationships.append({
                        'source': entity1,
                        'target': entity2,
                        'type': 'co-occurs',
                        'context': text[:100]
                    })
        return relationships
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the GraphRAG system
        
        Args:
            question: User question
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: {question}")
        
        if not self.graph:
            return {
                'answer': "No knowledge graph available. Please build graph first.",
                'sources': [],
                'entities': []
            }
        
        # Find relevant entities in question
        question_entities = self._extract_entities(question)
        
        # Find relevant documents using graph traversal
        relevant_docs = self._graph_based_retrieval(question_entities, top_k)
        
        # Generate answer using retrieved context
        context = "\n\n".join(relevant_docs[:3])
        answer = self._generate_answer(question, context)
        
        return {
            'answer': answer,
            'sources': relevant_docs,
            'entities': question_entities
        }
    
    def _graph_based_retrieval(self, entities: List[str], top_k: int) -> List[str]:
        """Retrieve relevant documents using graph structure"""
        if not self.graph:
            return []
        
        # Score documents based on entity matches and relationships
        doc_scores = []
        for doc in self.graph['documents']:
            score = sum(1 for entity in entities if entity in doc.lower())
            doc_scores.append((doc, score))
        
        # Sort by score and return top-k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:top_k] if score > 0]
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer from context (simplified)"""
        # In production, use actual LLM API
        if not context:
            return "I don't have enough information to answer that question."
        
        return f"Based on the available information: {context[:200]}..."


def main():
    """Main execution function"""
    print("=" * 60)
    print("GraphRAG - Graph-based Retrieval Augmented Generation")
    print("=" * 60)
    
    # Initialize GraphRAG
    graph_rag = GraphRAG()
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing allows computers to understand and generate human language.",
        "Computer vision enables machines to interpret and analyze visual information from images.",
        "Reinforcement learning trains agents through rewards and penalties in an environment."
    ]
    
    # Build knowledge graph
    print("\nBuilding knowledge graph...")
    graph_rag.build_knowledge_graph(documents)
    
    # Example queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is natural language processing?"
    ]
    
    print("\nProcessing queries...")
    for query in queries:
        print(f"\nQuery: {query}")
        result = graph_rag.query(query)
        print(f"Answer: {result['answer']}")
        print(f"Relevant entities: {result['entities'][:5]}")
        print(f"Sources found: {len(result['sources'])}")
    
    print("\n" + "=" * 60)
    print("GraphRAG Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
