"""The GraphRAG pipeline: index, retrieve, generate."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .config import GraphRAGConfig
from .documents import Chunk, Document, chunk_document, coerce_documents, load_documents
from .embeddings import HashingEmbedder, build_embedder
from .extraction import ExtractionResult, build_extractor
from .graph import KnowledgeGraph
from .llm import build_llm, generate_answer
from .retrieval import BM25Index, HybridRetriever, RetrievedChunk, vectors_for_chunks
from .storage import atomic_write_text
from .text import content_tokens
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)

GRAPH_FILE = "graph.json"
VECTORS_FILE = "vectors.npz"
BM25_FILE = "bm25.json"
CHUNKS_FILE = "chunks.json"
CONFIG_FILE = "config.json"


class GraphRAG:
    """Graph-based retrieval augmented generation.

    Indexing pipeline:
        documents -> chunks -> embeddings + BM25 + entity/relation extraction
        -> knowledge graph.

    Query pipeline:
        question -> seed entities -> personalised PageRank over the graph,
        fused with dense and lexical retrieval -> grounded generation.
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        config: Optional[GraphRAGConfig] = None,
        llm: Any = None,
        embedder: Optional[HashingEmbedder] = None,
        extractor: Any = None,
    ) -> None:
        self.config = config or GraphRAGConfig.from_env(
            llm_model=llm_model, embedding_model=embedding_model
        )
        self.embedder = embedder or build_embedder(self.config)
        self.llm = llm if llm is not None else build_llm(self.config)
        self.extractor = extractor or build_extractor(self.config, self.llm)

        self.graph = KnowledgeGraph()
        self.vector_store = VectorStore(dim=self.config.embedding_dim)
        self.bm25 = BM25Index()
        self.chunks: Dict[str, Chunk] = {}
        self.documents: Dict[str, Document] = {}

        logger.info(
            "Initialized GraphRAG (llm=%s, embeddings=%s, extractor=%s)",
            getattr(self.llm, "name", type(self.llm).__name__),
            self.embedder.name,
            type(self.extractor).__name__,
        )

    # -- indexing --------------------------------------------------------
    def add_documents(
        self,
        documents: Iterable[Union[str, Document, Dict[str, Any]]],
        source: str = "inline",
    ) -> int:
        """Index documents. Returns the number of new chunks added."""
        docs = coerce_documents(documents, source=source)
        new_documents: List[Document] = []
        new_chunks: List[Chunk] = []
        seen: set = set()
        for document in docs:
            if document.id in self.documents or document.id in seen:
                continue
            seen.add(document.id)
            new_documents.append(document)
            for chunk in chunk_document(
                document, self.config.chunk_size, self.config.chunk_overlap
            ):
                if chunk.id in self.chunks or chunk.id in seen:
                    continue
                seen.add(chunk.id)
                new_chunks.append(chunk)

        if not new_chunks:
            logger.info("No new chunks to index")
            return 0

        # Everything that can fail happens before any state is mutated. The
        # previous order registered the chunks first, so a failure part way
        # through left them recorded as indexed while carrying no vector, no
        # entities and no keyword posting - and a retry skipped them as
        # duplicates, making them permanently unretrievable.
        ids, matrix = vectors_for_chunks(new_chunks, self.embedder)
        extractions = [
            (chunk.id, self._extract(chunk)) for chunk in new_chunks
        ]

        self.vector_store.add(ids, matrix)
        for chunk_id, result in extractions:
            self.graph.add_extraction(chunk_id, result)
        for document in new_documents:
            self.documents[document.id] = document
        for chunk in new_chunks:
            self.chunks[chunk.id] = chunk

        # Incremental: re-fitting the whole corpus on every call made indexing
        # a corpus document by document quadratic in its size.
        self.bm25.add(
            [chunk.id for chunk in new_chunks],
            [content_tokens(chunk.text) for chunk in new_chunks],
        )
        logger.info(
            "Indexed %d chunks from %d documents (graph: %d entities, %d relations)",
            len(new_chunks),
            len(docs),
            self.graph.graph.number_of_nodes(),
            self.graph.graph.number_of_edges(),
        )
        return len(new_chunks)

    def add_path(self, path: Union[str, Path], **kwargs: Any) -> int:
        """Index every text file under ``path``."""
        return self.add_documents(load_documents(path, **kwargs))

    def build_knowledge_graph(
        self, documents: Iterable[Union[str, Document, Dict[str, Any]]]
    ) -> KnowledgeGraph:
        """Index documents and return the resulting knowledge graph."""
        self.add_documents(documents)
        return self.graph

    def _extract(self, chunk: Chunk):
        """Extract from one chunk, never letting it abort the whole batch."""
        try:
            return self.extractor.extract(chunk.text)
        except Exception:  # noqa: BLE001 - one bad chunk must not lose the rest
            logger.exception(
                "Extraction failed for chunk %s (%s); indexing it without entities",
                chunk.id,
                chunk.source,
            )
            return ExtractionResult()


    # -- retrieval + generation -----------------------------------------
    @property
    def retriever(self) -> HybridRetriever:
        return HybridRetriever(
            config=self.config,
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25=self.bm25,
            graph=self.graph,
            chunks=self.chunks,
            extractor=self.extractor,
        )

    def retrieve(
        self, question: str, top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """Run hybrid retrieval without generating an answer."""
        return self.retriever.retrieve(question, top_k=top_k)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        generate: bool = True,
    ) -> Dict[str, Any]:
        """Answer a question over the indexed corpus.

        Args:
            question: The user question.
            top_k: Number of chunks to retrieve (defaults to config).
            generate: When False, skip generation and return context only.

        Returns:
            Dict with ``answer``, ``sources``, ``entities``, ``context`` and
            ``retrieved`` (per-chunk scores and signal breakdown).
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("question must be a non-empty string")

        if not self.chunks:
            return {
                "answer": "No documents have been indexed yet. Add documents first.",
                "sources": [],
                "entities": [],
                "context": "",
                "retrieved": [],
            }

        retriever = self.retriever
        results = retriever.retrieve(question, top_k=top_k)
        context = retriever.build_context(results)
        seeds = retriever.query_entities(question)

        answer = (
            generate_answer(self.llm, question, context)
            if generate
            else "Generation skipped."
        )

        return {
            "answer": answer,
            "sources": [result.text for result in results],
            "entities": seeds,
            "context": context,
            "retrieved": [result.to_dict() for result in results],
        }

    def related_entities(self, entity: str, depth: int = 1) -> List[str]:
        """Entities within ``depth`` hops of ``entity`` in the graph."""
        return sorted(self.graph.neighbors(entity.lower(), depth=depth))

    def stats(self) -> Dict[str, Any]:
        """Corpus and graph statistics."""
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "vectors": len(self.vector_store),
            "llm": getattr(self.llm, "name", type(self.llm).__name__),
            "embedding_model": self.embedder.name,
            "graph": self.graph.stats(),
        }

    # -- persistence -----------------------------------------------------
    def save(self, directory: Optional[Union[str, Path]] = None) -> Path:
        """Persist the index to ``directory`` (defaults to config.storage_dir)."""
        target = Path(directory or self.config.storage_dir)
        target.mkdir(parents=True, exist_ok=True)

        self.graph.save(target / GRAPH_FILE)
        self.vector_store.save(target / VECTORS_FILE)
        self.bm25.save(target / BM25_FILE)
        atomic_write_text(
            target / CHUNKS_FILE,
            json.dumps(
                {
                    "documents": [doc.to_dict() for doc in self.documents.values()],
                    "chunks": [chunk.to_dict() for chunk in self.chunks.values()],
                },
                indent=2,
            ),
        )
        atomic_write_text(
            target / CONFIG_FILE, json.dumps(self.config.to_dict(), indent=2)
        )
        logger.info("Saved index to %s", target)
        return target

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        config: Optional[GraphRAGConfig] = None,
        llm: Any = None,
    ) -> "GraphRAG":
        """Load an index previously written by :meth:`save`."""
        source = Path(directory)
        if not source.exists():
            raise FileNotFoundError(f"No index directory at {source}")
        if not source.is_dir():
            raise NotADirectoryError(f"{source} is not an index directory")

        # Report every missing part at once: chasing one FileNotFoundError per
        # run to discover a half written index is nobody's idea of a good time.
        missing = [
            name
            for name in (GRAPH_FILE, VECTORS_FILE, BM25_FILE, CHUNKS_FILE)
            if not (source / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{source} is not a complete GraphRAG index; missing: "
                + ", ".join(sorted(missing))
            )

        if config is None:
            config_path = source / CONFIG_FILE
            try:
                stored = (
                    json.loads(config_path.read_text(encoding="utf-8"))
                    if config_path.exists()
                    else {}
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"{config_path} is not valid JSON: {exc}") from exc
            env_config = GraphRAGConfig.from_env()
            # A later save() with no argument must round-trip to where we
            # loaded from, not to whatever directory the index was built in.
            stored["storage_dir"] = str(source)
            config = GraphRAGConfig.from_dict(stored)
            config.api_key = env_config.api_key

        system = cls(config=config, llm=llm)
        system.graph = KnowledgeGraph.load(source / GRAPH_FILE)
        system.vector_store = VectorStore.load(source / VECTORS_FILE)
        system.bm25 = BM25Index.load(source / BM25_FILE)

        # An index built with a different embedding width only fails later, in
        # the middle of a query, as an opaque numpy shape error.
        embedder_dim = getattr(system.embedder, "dim", system.vector_store.dim)
        if len(system.vector_store) and system.vector_store.dim != embedder_dim:
            raise ValueError(
                f"index at {source} holds {system.vector_store.dim}-dimensional "
                f"vectors but the configured embedder produces "
                f"{embedder_dim}; set GRAPHRAG_EMBEDDING_DIM="
                f"{system.vector_store.dim} or re-index"
            )

        chunks_path = source / CHUNKS_FILE
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{chunks_path} is not valid JSON: {exc}") from exc
        try:
            system.documents = {
                data["id"]: Document.from_dict(data)
                for data in payload.get("documents", [])
            }
            system.chunks = {
                data["id"]: Chunk.from_dict(data) for data in payload.get("chunks", [])
            }
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{chunks_path} is missing required fields: {exc}") from exc
        logger.info(
            "Loaded index from %s (%d chunks, %d entities)",
            source,
            len(system.chunks),
            len(system.graph),
        )
        return system
