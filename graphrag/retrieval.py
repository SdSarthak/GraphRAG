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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .documents import Chunk
from .embeddings import HashingEmbedder
from .extraction import extract_query_entities
from .graph import KnowledgeGraph
from .storage import atomic_write_text
from .text import content_tokens
from .vectorstore import VectorStore


class BM25Index:
    """Okapi BM25 over pre-tokenised chunks.

    Supports incremental :meth:`add` (indexing a corpus document by document
    is otherwise quadratic) and scores through an inverted index, so a query
    only touches the chunks that actually contain one of its terms.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: List[str] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.avg_length: float = 0.0
        self._positions: Dict[str, int] = {}
        self._postings: Dict[str, Set[int]] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._positions

    def fit(self, ids: Sequence[str], documents: Sequence[Sequence[str]]) -> None:
        """(Re)build the index from token lists, discarding anything held."""
        self.ids = []
        self.doc_freqs = []
        self.doc_lengths = []
        self._positions = {}
        self._postings = {}
        self.add(ids, documents)

    def add(self, ids: Sequence[str], documents: Sequence[Sequence[str]]) -> None:
        """Add (or replace) documents without re-reading the whole corpus."""
        if len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")
        for chunk_id, tokens in zip(ids, documents):
            freqs = Counter(tokens)
            length = sum(freqs.values())
            position = self._positions.get(chunk_id)
            if position is None:
                position = len(self.ids)
                self._positions[chunk_id] = position
                self.ids.append(chunk_id)
                self.doc_freqs.append(freqs)
                self.doc_lengths.append(length)
            else:
                for term in self.doc_freqs[position]:
                    postings = self._postings.get(term)
                    if postings is not None:
                        postings.discard(position)
                self.doc_freqs[position] = freqs
                self.doc_lengths[position] = length
            for term in freqs:
                self._postings.setdefault(term, set()).add(position)
        self.avg_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )

    def idf(self, term: str) -> float:
        """Inverse document frequency, computed on demand.

        Kept lazy so that adding a document costs the length of that document
        rather than the size of the whole vocabulary.
        """
        count = len(self._postings.get(term, ()))
        if not count:
            return 0.0
        return math.log(1.0 + (len(self.ids) - count + 0.5) / (count + 0.5))

    @property
    def inverse_doc_freq(self) -> Dict[str, float]:
        return {term: self.idf(term) for term in self._postings}

    def search(
        self, query_tokens: Sequence[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return the ``top_k`` ``(chunk_id, bm25_score)`` pairs."""
        if not self.ids or not query_tokens or top_k <= 0:
            return []

        scores: Dict[int, float] = {}
        average = self.avg_length or 1.0
        for term in set(query_tokens):
            candidates = self._postings.get(term)
            if not candidates:
                continue
            idf = self.idf(term)
            if idf <= 0.0:
                continue
            for index in candidates:
                frequency = self.doc_freqs[index].get(term, 0)
                if not frequency:
                    continue
                length = self.doc_lengths[index] or 1
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / average
                )
                scores[index] = scores.get(index, 0.0) + idf * (
                    frequency * (self.k1 + 1.0)
                ) / denominator

        ranked = sorted(
            ((self.ids[index], score) for index, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:top_k]

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
        if not isinstance(data, dict):
            raise ValueError("bm25 data must be a JSON object")
        index = cls(k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))
        ids = list(data.get("ids", []))
        doc_freqs = list(data.get("doc_freqs", []))
        doc_lengths = list(data.get("doc_lengths", []))
        if len(doc_freqs) != len(ids):
            raise ValueError(
                f"bm25 index is corrupt: {len(ids)} ids but "
                f"{len(doc_freqs)} frequency tables"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("bm25 index is corrupt: duplicate chunk ids")

        # Restoring used to rebuild a full token list per document just to
        # re-count it, which meant holding the entire corpus in memory again.
        for position, (chunk_id, freqs) in enumerate(zip(ids, doc_freqs)):
            counter = Counter({str(term): int(count) for term, count in freqs.items()})
            index._positions.setdefault(chunk_id, position)
            index.ids.append(chunk_id)
            index.doc_freqs.append(counter)
            index.doc_lengths.append(
                int(doc_lengths[position])
                if position < len(doc_lengths)
                else sum(counter.values())
            )
            for term in counter:
                index._postings.setdefault(term, set()).add(position)
        index.avg_length = (
            sum(index.doc_lengths) / len(index.doc_lengths) if index.doc_lengths else 0.0
        )
        return index

    def save(self, path: Union[str, Path]) -> None:
        atomic_write_text(path, json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BM25Index":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)


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
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        parts: List[str] = []
        used = 0
        for index, result in enumerate(results, start=1):
            block = f"[{index}] (source: {result.chunk.source})\n{result.text}"
            if used + len(block) > max_chars:
                if parts:
                    break
                # A single chunk larger than the whole budget used to be
                # emitted in full, so a big chunk_size silently blew past
                # max_context_chars and inflated every prompt.
                block = block[:max_chars]
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
