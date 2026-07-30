"""Hybrid retrieval: dense vectors + BM25 + graph traversal.

The three signals are combined with weighted Reciprocal Rank Fusion, which is
rank-based and therefore does not require the individual scorers to share a
scale (cosine similarity, BM25 scores and PageRank mass are not comparable
numerically, but their rankings are).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .documents import Chunk
from .embeddings import HashingEmbedder
from .extraction import extract_query_entities
from .graph import KnowledgeGraph
from .text import content_tokens
from .vectorstore import VectorStore


class BM25Index:
    """Okapi BM25 over pre-tokenised chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: List[str] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.avg_length: float = 0.0
        self.inverse_doc_freq: Dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def fit(self, ids: Sequence[str], documents: Sequence[Sequence[str]]) -> None:
        """(Re)build the index from token lists."""
        if len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")
        self.ids = list(ids)
        self.doc_freqs = [Counter(tokens) for tokens in documents]
        self.doc_lengths = [len(tokens) for tokens in documents]
        total = sum(self.doc_lengths)
        self.avg_length = total / len(self.doc_lengths) if self.doc_lengths else 0.0

        document_count = len(self.ids)
        containing: Counter = Counter()
        for freqs in self.doc_freqs:
            containing.update(freqs.keys())
        self.inverse_doc_freq = {
            term: math.log(1.0 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in containing.items()
        }

    def search(
        self, query_tokens: Sequence[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return the ``top_k`` ``(chunk_id, bm25_score)`` pairs."""
        if not self.ids or not query_tokens or top_k <= 0:
            return []
        scores: List[Tuple[str, float]] = []
        for index, freqs in enumerate(self.doc_freqs):
            length = self.doc_lengths[index] or 1
            score = 0.0
            for term in query_tokens:
                frequency = freqs.get(term)
                if not frequency:
                    continue
                idf = self.inverse_doc_freq.get(term, 0.0)
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / (self.avg_length or 1.0)
                )
                score += idf * (frequency * (self.k1 + 1.0)) / denominator
            if score > 0.0:
                scores.append((self.ids[index], score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "ids": self.ids,
            "doc_freqs": [dict(freqs) for freqs in self.doc_freqs],
            "doc_lengths": self.doc_lengths,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25Index":
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        documents = [
            [term for term, count in freqs.items() for _ in range(count)]
            for freqs in data.get("doc_freqs", [])
        ]
        index.fit(data.get("ids", []), documents)
        return index

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Index":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class RetrievedChunk:
    """A retrieval hit with its per-signal provenance."""

    chunk: Chunk
    score: float
    signals: Dict[str, float] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "score": round(self.score, 6),
            "signals": {k: round(v, 6) for k, v in self.signals.items()},
            "entities": self.entities,
        }


def reciprocal_rank_fusion(
    rankings: Dict[str, Sequence[str]],
    weights: Optional[Dict[str, float]] = None,
    k: int = 60,
) -> Dict[str, Dict[str, float]]:
    """Fuse named rankings into ``{chunk_id: {"total": s, <signal>: s}}``."""
    weights = weights or {}
    fused: Dict[str, Dict[str, float]] = {}
    for name, ordered_ids in rankings.items():
        weight = weights.get(name, 1.0)
        if weight <= 0.0:
            continue
        for rank, chunk_id in enumerate(ordered_ids):
            contribution = weight / (k + rank + 1)
            entry = fused.setdefault(chunk_id, {"total": 0.0})
            entry[name] = entry.get(name, 0.0) + contribution
            entry["total"] += contribution
    return fused


class HybridRetriever:
    """Fuses vector, keyword and graph retrieval over a shared chunk set."""

    def __init__(
        self,
        config,
        embedder: HashingEmbedder,
        vector_store: VectorStore,
        bm25: BM25Index,
        graph: KnowledgeGraph,
        chunks: Dict[str, Chunk],
        extractor: Any = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.graph = graph
        self.chunks = chunks
        self.extractor = extractor

    def retrieve(
        self, question: str, top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """Retrieve the chunks most relevant to ``question``."""
        top_k = top_k or self.config.retrieval_top_k
        if not question.strip() or not self.chunks or top_k <= 0:
            return []

        pool = max(top_k * 4, top_k + 5)
        rankings: Dict[str, List[str]] = {}

        query_vector = self.embedder.embed_one(question)
        rankings["vector"] = [
            chunk_id for chunk_id, _ in self.vector_store.search(query_vector, pool)
        ]

        rankings["keyword"] = [
            chunk_id
            for chunk_id, _ in self.bm25.search(content_tokens(question), pool)
        ]

        seeds = self.query_entities(question)
        graph_scores = self.graph.score_chunks(
            seeds,
            alpha=self.config.pagerank_alpha,
            depth=self.config.graph_expansion_depth,
        )
        rankings["graph"] = [
            chunk_id
            for chunk_id, _ in sorted(
                graph_scores.items(), key=lambda item: (-item[1], item[0])
            )[:pool]
        ]

        fused = reciprocal_rank_fusion(
            rankings,
            weights={
                "vector": self.config.vector_weight,
                "keyword": self.config.keyword_weight,
                "graph": self.config.graph_weight,
            },
            k=self.config.rrf_k,
        )

        ordered = sorted(
            fused.items(), key=lambda item: (-item[1]["total"], item[0])
        )[:top_k]

        results: List[RetrievedChunk] = []
        for chunk_id, signals in ordered:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            total = signals.pop("total")
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=total,
                    signals=signals,
                    entities=sorted(self.graph.chunk_entities.get(chunk_id, set())),
                )
            )
        return results

    def query_entities(self, question: str) -> List[str]:
        """Graph entities the question anchors to.

        Combines exact phrase matches from the extractor with token-coverage
        matches, so a question that phrases an entity differently from the
        indexed text still seeds the graph walk.
        """
        seeds: List[str] = []
        for key in extract_query_entities(question, self.extractor):
            if self.graph.has_entity(key) and key not in seeds:
                seeds.append(key)
        for key in self.graph.find_entities(content_tokens(question)):
            if key not in seeds:
                seeds.append(key)
        return seeds

    def build_context(
        self, results: Sequence[RetrievedChunk], max_chars: Optional[int] = None
    ) -> str:
        """Render retrieved chunks as a numbered, citation-friendly context."""
        max_chars = max_chars or self.config.max_context_chars
        parts: List[str] = []
        used = 0
        for index, result in enumerate(results, start=1):
            block = f"[{index}] (source: {result.chunk.source})\n{result.text}"
            if used + len(block) > max_chars and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)


def vectors_for_chunks(
    chunks: Sequence[Chunk], embedder: HashingEmbedder
) -> Tuple[List[str], np.ndarray]:
    """Embed chunks, returning aligned ids and a vector matrix."""
    ids = [chunk.id for chunk in chunks]
    matrix = embedder.embed([chunk.text for chunk in chunks])
    return ids, matrix
