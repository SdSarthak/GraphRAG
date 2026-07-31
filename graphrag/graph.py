"""Knowledge graph built on networkx."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx

from .extraction import Entity, ExtractionResult, Relationship
from .storage import atomic_write_text

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """A directed, weighted entity graph with chunk provenance.

    Nodes are canonical entity keys (lowercased names). Every node keeps the
    set of chunk ids it was observed in, which is what turns graph structure
    into document retrieval.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.chunk_entities: Dict[str, Set[str]] = {}

    # -- construction ----------------------------------------------------
    def add_extraction(self, chunk_id: str, result: ExtractionResult) -> None:
        """Merge one chunk's extraction into the graph."""
        for entity in result.entities:
            self.add_entity(entity, chunk_id)
        for relationship in result.relationships:
            self.add_relationship(relationship)

    def add_entity(self, entity: Entity, chunk_id: Optional[str] = None) -> str:
        key = entity.key
        if not key:
            return key
        if self.graph.has_node(key):
            node = self.graph.nodes[key]
            node["mentions"] = node.get("mentions", 0) + entity.mentions
            if entity.type == "proper_noun":
                node["type"] = "proper_noun"
        else:
            self.graph.add_node(
                key,
                name=entity.name,
                type=entity.type,
                mentions=entity.mentions,
                chunks=set(),
            )
        if chunk_id:
            self.graph.nodes[key]["chunks"].add(chunk_id)
            self.chunk_entities.setdefault(chunk_id, set()).add(key)
        return key

    def add_relationship(self, relationship: Relationship) -> None:
        source, target = relationship.source, relationship.target
        if not source or not target or source == target:
            return
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return
        if self.graph.has_edge(source, target):
            edge = self.graph.edges[source, target]
            edge["weight"] = edge.get("weight", 0.0) + relationship.weight
            types = edge.setdefault("types", [])
            if relationship.type not in types:
                types.append(relationship.type)
        else:
            self.graph.add_edge(
                source,
                target,
                weight=relationship.weight,
                types=[relationship.type],
                context=relationship.context,
            )

    # -- inspection ------------------------------------------------------
    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def entities(self) -> List[str]:
        return list(self.graph.nodes)

    def has_entity(self, key: str) -> bool:
        return self.graph.has_node(key.lower())

    def entity_name(self, key: str) -> str:
        if self.graph.has_node(key):
            return self.graph.nodes[key].get("name", key)
        return key

    def chunks_for(self, key: str) -> Set[str]:
        if not self.graph.has_node(key):
            return set()
        return set(self.graph.nodes[key].get("chunks", set()))

    def find_entities(self, tokens: Iterable[str], limit: int = 12) -> List[str]:
        """Entities whose every token appears in ``tokens``.

        Lets a question anchor to "knowledge graphs" even when the question
        phrases it as "how do knowledge graphs help" — exact phrase equality is
        too brittle to be the only seeding path.
        """
        available = set(tokens)
        if not available:
            return []
        matches: List[Tuple[int, int, str]] = []
        for node, data in self.graph.nodes(data=True):
            parts = node.split()
            if parts and all(part in available for part in parts):
                matches.append((len(parts), int(data.get("mentions", 1)), node))
        # Prefer the most specific (longest) and most frequent matches.
        matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [node for _, _, node in matches[:limit]]

    def neighbors(self, key: str, depth: int = 1) -> Set[str]:
        """Entities reachable within ``depth`` hops, ignoring edge direction."""
        key = key.lower()
        if not self.graph.has_node(key) or depth < 1:
            return set()
        return self.reachable([key], depth) - {key}

    def reachable(self, keys: Iterable[str], depth: int = 1) -> Set[str]:
        """Everything within ``depth`` hops of any of ``keys``, seeds included.

        One breadth-first sweep from all seeds at once. Expanding each seed
        separately re-walked the same neighbourhoods on every query, which on
        a densely connected graph is the most expensive part of retrieval.
        """
        frontier = {
            key.lower() for key in keys if self.graph.has_node(key.lower())
        }
        if not frontier or depth < 0:
            return set()
        visited = set(frontier)
        for _ in range(depth):
            nxt: Set[str] = set()
            for node in frontier:
                nxt.update(self.graph.successors(node))
                nxt.update(self.graph.predecessors(node))
            frontier = nxt - visited
            if not frontier:
                break
            visited |= frontier
        return visited

    def relationships(self) -> List[Dict[str, Any]]:
        return [
            {
                "source": source,
                "target": target,
                "types": data.get("types", []),
                "weight": data.get("weight", 1.0),
                "context": data.get("context", ""),
            }
            for source, target, data in self.graph.edges(data=True)
        ]

    def stats(self) -> Dict[str, Any]:
        undirected = self.graph.to_undirected(as_view=True)
        components = (
            nx.number_connected_components(undirected) if len(self.graph) else 0
        )
        degrees = dict(self.graph.degree())
        top = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:5]
        return {
            "entities": self.graph.number_of_nodes(),
            "relationships": self.graph.number_of_edges(),
            "chunks": len(self.chunk_entities),
            "components": components,
            "top_entities": [{"entity": key, "degree": deg} for key, deg in top],
        }

    def communities(self, resolution: float = 1.0) -> List[List[str]]:
        """Greedy modularity communities over the undirected projection."""
        if not len(self.graph):
            return []
        undirected = nx.Graph()
        undirected.add_nodes_from(self.graph.nodes)
        for source, target, data in self.graph.edges(data=True):
            weight = data.get("weight", 1.0)
            if undirected.has_edge(source, target):
                undirected.edges[source, target]["weight"] += weight
            else:
                undirected.add_edge(source, target, weight=weight)
        groups = nx.community.greedy_modularity_communities(
            undirected, weight="weight", resolution=resolution
        )
        return [sorted(group) for group in groups]

    # -- ranking ---------------------------------------------------------
    def _walk_adjacency(self) -> Dict[str, Dict[str, float]]:
        """Symmetric, weight-normalised transition table for random walks.

        Relations are stored directed, but a reader following a relation
        backwards is just as relevant for retrieval, so the walk uses the
        undirected projection.
        """
        adjacency: Dict[str, Dict[str, float]] = {
            node: {} for node in self.graph.nodes
        }
        for source, target, data in self.graph.edges(data=True):
            weight = float(data.get("weight", 1.0))
            adjacency[source][target] = adjacency[source].get(target, 0.0) + weight
            adjacency[target][source] = adjacency[target].get(source, 0.0) + weight
        return adjacency

    def personalized_pagerank(
        self,
        seeds: Sequence[str],
        alpha: float = 0.85,
        top_k: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1.0e-8,
    ) -> Dict[str, float]:
        """Rank entities by their proximity to ``seeds``.

        Implemented as a power iteration over the undirected projection so the
        package depends only on networkx (networkx's own PageRank delegates to
        scipy, which is a heavier requirement than this problem warrants).

        Unknown seeds are ignored. With no usable seed the result is empty,
        which lets the caller fall back to text-only retrieval.
        """
        if not len(self.graph):
            return {}

        restart = {key.lower(): 1.0 for key in seeds if self.graph.has_node(key.lower())}
        if not restart:
            return {}
        total = sum(restart.values())
        restart = {node: weight / total for node, weight in restart.items()}

        adjacency = self._walk_adjacency()
        nodes = list(adjacency)
        scores = {node: restart.get(node, 0.0) for node in nodes}

        for _ in range(max_iter):
            updated = {node: 0.0 for node in nodes}
            dangling_mass = 0.0
            for node, score in scores.items():
                neighbours = adjacency[node]
                if not neighbours:
                    dangling_mass += score
                    continue
                out_weight = sum(neighbours.values())
                for neighbour, weight in neighbours.items():
                    updated[neighbour] += score * weight / out_weight

            delta = 0.0
            for node in nodes:
                value = alpha * (updated[node] + dangling_mass * restart.get(node, 0.0))
                value += (1.0 - alpha) * restart.get(node, 0.0)
                delta += abs(value - scores[node])
                updated[node] = value
            scores = updated
            if delta < tol:
                break
        else:  # pragma: no cover - only on pathological graphs
            logger.debug("PageRank hit the iteration cap before converging")

        ranked = sorted(
            ((node, score) for node, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return dict(ranked)

    def score_chunks(
        self, seeds: Sequence[str], alpha: float = 0.85, depth: int = 2
    ) -> Dict[str, float]:
        """Score chunks by the PageRank mass of the entities they contain."""
        entity_scores = self.personalized_pagerank(seeds, alpha=alpha)
        if not entity_scores:
            return {}

        reachable = self.reachable(seeds, depth=depth)
        chunk_scores: Dict[str, float] = {}
        for entity, score in entity_scores.items():
            if reachable and entity not in reachable:
                continue
            for chunk_id in self.graph.nodes[entity].get("chunks", set()):
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score
        return chunk_scores

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": key,
                    "name": data.get("name", key),
                    "type": data.get("type", "concept"),
                    "mentions": data.get("mentions", 1),
                    "chunks": sorted(data.get("chunks", set())),
                }
                for key, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": data.get("weight", 1.0),
                    "types": data.get("types", []),
                    "context": data.get("context", ""),
                }
                for source, target, data in self.graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        if not isinstance(data, dict):
            raise ValueError("graph data must be a JSON object")
        graph = cls()
        for node in data.get("nodes", []):
            key = node.get("id")
            if not key:
                raise ValueError("graph node is missing its 'id'")
            chunks = set(node.get("chunks", []))
            graph.graph.add_node(
                key,
                name=node.get("name", key),
                type=node.get("type", "concept"),
                mentions=node.get("mentions", 1),
                chunks=chunks,
            )
            for chunk_id in chunks:
                graph.chunk_entities.setdefault(chunk_id, set()).add(key)

        dangling = 0
        for edge in data.get("edges", []):
            source, target = edge.get("source"), edge.get("target")
            # networkx creates missing endpoints on demand, which would inject
            # attribute-less phantom entities that then get ranked and returned
            # as if they were real. Drop those edges instead.
            if not graph.graph.has_node(source) or not graph.graph.has_node(target):
                dangling += 1
                continue
            graph.graph.add_edge(
                source,
                target,
                weight=float(edge.get("weight", 1.0)),
                types=list(edge.get("types", [])),
                context=edge.get("context", ""),
            )
        if dangling:
            logger.warning(
                "Skipped %d relationship(s) pointing at unknown entities", dangling
            )
        return graph

    def save(self, path: Union[str, Path]) -> None:
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "KnowledgeGraph":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)

    def export_graphml(self, path: Union[str, Path]) -> None:
        """Write a GraphML file for Gephi / Cytoscape."""
        exportable = nx.DiGraph()
        for key, data in self.graph.nodes(data=True):
            exportable.add_node(
                key,
                name=data.get("name", key),
                type=data.get("type", "concept"),
                mentions=int(data.get("mentions", 1)),
                chunks=len(data.get("chunks", set())),
            )
        for source, target, data in self.graph.edges(data=True):
            exportable.add_edge(
                source,
                target,
                weight=float(data.get("weight", 1.0)),
                types=",".join(data.get("types", [])),
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(exportable, str(path))


def merge_extractions(
    items: Iterable[Tuple[str, ExtractionResult]]
) -> KnowledgeGraph:
    """Build a graph from ``(chunk_id, extraction)`` pairs."""
    graph = KnowledgeGraph()
    for chunk_id, result in items:
        graph.add_extraction(chunk_id, result)
    return graph
