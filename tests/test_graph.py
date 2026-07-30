import pytest

from graphrag.extraction import Entity, ExtractionResult, Relationship
from graphrag.graph import KnowledgeGraph


def build_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_extraction(
        "chunk-1",
        ExtractionResult(
            entities=[Entity(name="Deep Learning"), Entity(name="Neural Networks")],
            relationships=[
                Relationship(source="deep learning", target="neural networks", type="uses")
            ],
        ),
    )
    graph.add_extraction(
        "chunk-2",
        ExtractionResult(
            entities=[Entity(name="Neural Networks"), Entity(name="Computer Vision")],
            relationships=[
                Relationship(
                    source="neural networks", target="computer vision", type="powers"
                )
            ],
        ),
    )
    return graph


def test_graph_merges_entities_across_chunks():
    graph = build_graph()

    assert len(graph) == 3
    assert graph.chunks_for("neural networks") == {"chunk-1", "chunk-2"}
    assert graph.graph.nodes["neural networks"]["mentions"] == 2


def test_relationships_to_unknown_entities_are_ignored():
    graph = KnowledgeGraph()
    graph.add_entity(Entity(name="A"), "chunk-1")
    graph.add_relationship(Relationship(source="a", target="ghost"))

    assert graph.graph.number_of_edges() == 0


def test_self_loops_are_ignored():
    graph = KnowledgeGraph()
    graph.add_entity(Entity(name="A"), "chunk-1")
    graph.add_relationship(Relationship(source="a", target="a"))

    assert graph.graph.number_of_edges() == 0


def test_duplicate_relationship_accumulates_weight():
    graph = build_graph()
    graph.add_relationship(
        Relationship(source="deep learning", target="neural networks", type="requires")
    )
    edge = graph.graph.edges["deep learning", "neural networks"]

    assert edge["weight"] > 1.0
    assert set(edge["types"]) == {"uses", "requires"}


def test_neighbors_respects_depth_and_direction():
    graph = build_graph()

    assert graph.neighbors("deep learning", depth=1) == {"neural networks"}
    assert graph.neighbors("deep learning", depth=2) == {
        "neural networks",
        "computer vision",
    }
    # Traversal ignores edge direction.
    assert "neural networks" in graph.neighbors("computer vision", depth=1)


def test_personalized_pagerank_prefers_the_seed_neighbourhood():
    graph = build_graph()
    scores = graph.personalized_pagerank(["deep learning"])

    assert scores
    # Mass decays with distance from the seed. (The seed itself need not be the
    # maximum: on a path graph the walk always leaves a leaf seed, so the hub
    # it points at accumulates more mass — standard personalised PageRank.)
    assert scores["deep learning"] > scores["computer vision"]
    assert scores["neural networks"] > scores["computer vision"]
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)


def test_find_entities_matches_by_token_coverage():
    graph = build_graph()
    # The extra question words must not prevent the match.
    matches = graph.find_entities(
        ["how", "does", "deep", "learning", "use", "neural", "networks"]
    )

    assert "deep learning" in matches
    assert "neural networks" in matches
    assert "computer vision" not in matches
    # Most specific match first.
    assert len(matches[0].split()) >= len(matches[-1].split())


def test_find_entities_requires_every_token_to_be_present():
    graph = build_graph()
    assert graph.find_entities(["deep"]) == []
    assert graph.find_entities([]) == []


def test_personalized_pagerank_without_known_seeds_is_empty():
    assert build_graph().personalized_pagerank(["nothing here"]) == {}


def test_score_chunks_ranks_by_entity_mass():
    graph = build_graph()
    scores = graph.score_chunks(["deep learning"], depth=2)

    assert set(scores) == {"chunk-1", "chunk-2"}
    assert scores["chunk-1"] > scores["chunk-2"]


def test_stats_reports_structure():
    stats = build_graph().stats()

    assert stats["entities"] == 3
    assert stats["relationships"] == 2
    assert stats["chunks"] == 2
    assert stats["components"] == 1
    assert stats["top_entities"][0]["entity"] == "neural networks"


def test_communities_partition_all_nodes():
    graph = build_graph()
    groups = graph.communities()

    covered = {node for group in groups for node in group}
    assert covered == set(graph.entities)


def test_graph_round_trips_through_json(tmp_path):
    graph = build_graph()
    path = tmp_path / "graph.json"
    graph.save(path)
    restored = KnowledgeGraph.load(path)

    assert restored.entities == graph.entities
    assert restored.chunks_for("neural networks") == {"chunk-1", "chunk-2"}
    assert restored.chunk_entities == graph.chunk_entities
    assert restored.graph.number_of_edges() == graph.graph.number_of_edges()


def test_export_graphml(tmp_path):
    path = tmp_path / "graph.graphml"
    build_graph().export_graphml(path)

    assert path.exists()
    assert "graphml" in path.read_text(encoding="utf-8").lower()


def test_empty_graph_is_safe():
    graph = KnowledgeGraph()

    assert len(graph) == 0
    assert graph.personalized_pagerank(["anything"]) == {}
    assert graph.score_chunks(["anything"]) == {}
    assert graph.communities() == []
    assert graph.stats()["entities"] == 0
