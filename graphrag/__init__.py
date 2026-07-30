"""
GraphRAG - Graph-based Retrieval Augmented Generation.

A RAG system that builds a knowledge graph over a corpus and fuses graph
traversal with dense and lexical retrieval before generating a grounded answer.
"""

from .config import GraphRAGConfig
from .documents import Chunk, Document, chunk_text, load_documents
from .embeddings import HashingEmbedder
from .extraction import (
    Entity,
    ExtractionResult,
    LLMExtractor,
    Relationship,
    RuleBasedExtractor,
)
from .graph import KnowledgeGraph
from .llm import AnthropicLLM, ExtractiveLLM, build_llm
from .pipeline import GraphRAG
from .retrieval import BM25Index, HybridRetriever, RetrievedChunk
from .vectorstore import VectorStore

__version__ = "0.2.0"

__all__ = [
    "AnthropicLLM",
    "BM25Index",
    "Chunk",
    "Document",
    "Entity",
    "ExtractionResult",
    "ExtractiveLLM",
    "GraphRAG",
    "GraphRAGConfig",
    "HashingEmbedder",
    "HybridRetriever",
    "KnowledgeGraph",
    "LLMExtractor",
    "Relationship",
    "RetrievedChunk",
    "RuleBasedExtractor",
    "VectorStore",
    "build_llm",
    "chunk_text",
    "load_documents",
    "__version__",
]
