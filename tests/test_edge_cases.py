# -*- coding: utf-8 -*-
"""Edge cases across corpus loading, the graph and context assembly."""

import os

import pytest

from graphrag import GraphRAG, GraphRAGConfig
from graphrag.documents import iter_text_files, load_documents
from graphrag.extraction import Entity, Relationship
from graphrag.graph import KnowledgeGraph
from graphrag.llm import ExtractiveLLM


def make_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for name in ("a", "b", "c", "d", "island"):
        graph.add_entity(Entity(name=name), chunk_id=f"chunk-{name}")
    for source, target in (("a", "b"), ("b", "c"), ("c", "d")):
        graph.add_relationship(Relationship(source=source, target=target))
    return graph


# -- corpus loading -----------------------------------------------------
def test_a_symlink_cycle_does_not_hang_the_walk(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "deep").mkdir(parents=True)
    (corpus / "deep" / "note.txt").write_text("neural networks", encoding="utf-8")
    try:
        os.symlink(corpus, corpus / "deep" / "loop", target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symlinks are not available to this user")

    assert [path.name for path in iter_text_files(corpus)] == ["note.txt"]


def test_noise_directories_are_skipped(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.txt").write_text("junk", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "COMMIT_EDITMSG.txt").write_text("wip", encoding="utf-8")
    (tmp_path / "real.txt").write_text("neural networks", encoding="utf-8")

    assert [path.name for path in iter_text_files(tmp_path)] == ["real.txt"]


def test_nested_directories_are_all_visited(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "one.txt").write_text("first", encoding="utf-8")
    (tmp_path / "a" / "b" / "two.md").write_text("second", encoding="utf-8")

    assert len(load_documents(tmp_path)) == 2


def test_loading_a_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_text_files(tmp_path / "nope"))


def test_max_files_is_honoured_and_validated(tmp_path):
    for index in range(4):
        (tmp_path / f"{index}.txt").write_text(f"document {index}", encoding="utf-8")

    assert len(load_documents(tmp_path, max_files=2)) == 2
    with pytest.raises(ValueError):
        load_documents(tmp_path, max_files=0)


def test_unreadable_files_are_skipped_with_a_warning(tmp_path, caplog, monkeypatch):
    (tmp_path / "a.txt").write_text("readable", encoding="utf-8")
    (tmp_path / "b.txt").write_text("also readable", encoding="utf-8")

    original = type(tmp_path).read_text

    def explode(self, *args, **kwargs):
        if self.name == "b.txt":
            raise PermissionError("denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", explode)

    documents = load_documents(tmp_path)
    assert len(documents) == 1
    assert "b.txt" in caplog.text


# -- graph traversal ----------------------------------------------------
def test_reachable_expands_every_seed_at_once():
    graph = make_graph()
    assert graph.reachable(["a"], depth=1) == {"a", "b"}
    assert graph.reachable(["a", "d"], depth=1) == {"a", "b", "c", "d"}
    assert graph.reachable(["a"], depth=3) == {"a", "b", "c", "d"}


def test_reachable_ignores_unknown_seeds_and_zero_depth():
    graph = make_graph()
    assert graph.reachable(["nowhere"], depth=2) == set()
    assert graph.reachable(["a"], depth=0) == {"a"}


def test_neighbours_follow_edges_in_both_directions():
    graph = make_graph()
    assert graph.neighbors("b", depth=1) == {"a", "c"}
    assert graph.neighbors("island", depth=5) == set()


def test_pagerank_without_a_usable_seed_is_empty():
    graph = make_graph()
    assert graph.personalized_pagerank(["nowhere"]) == {}
    assert graph.score_chunks(["nowhere"]) == {}


def test_pagerank_matches_the_reference_power_iteration():
    """Cross-check the hand-rolled walk against the textbook equation.

    ``pi = (1 - alpha) * restart + alpha * P.T @ pi`` over the undirected
    projection. The package implements this itself rather than calling
    ``networkx.pagerank``, which pulls in scipy, so it needs checking against
    something other than itself.
    """
    import numpy as np

    graph = make_graph()
    scores = graph.personalized_pagerank(["a"], alpha=0.85)

    nodes = ["a", "b", "c", "d"]
    position = {node: index for index, node in enumerate(nodes)}
    weights = np.zeros((4, 4))
    for source, target in (("a", "b"), ("b", "c"), ("c", "d")):
        weights[position[source], position[target]] = 1.0
        weights[position[target], position[source]] = 1.0
    transition = weights / weights.sum(axis=1, keepdims=True)

    restart = np.array([1.0, 0.0, 0.0, 0.0])
    expected = restart.copy()
    for _ in range(500):
        expected = 0.85 * (transition.T @ expected) + 0.15 * restart

    for node in nodes:
        assert scores[node] == pytest.approx(expected[position[node]], abs=1e-6)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)
    # Distance from the seed still orders the far end of the chain.
    assert scores["c"] > scores["d"]
    assert "island" not in scores  # unreachable entities carry no mass


def test_score_chunks_is_restricted_to_the_reachable_neighbourhood():
    graph = make_graph()
    scores = graph.score_chunks(["a"], depth=1)

    assert set(scores) <= {"chunk-a", "chunk-b"}
    assert scores


def test_empty_graph_operations_are_safe():
    graph = KnowledgeGraph()
    assert graph.personalized_pagerank(["a"]) == {}
    assert graph.communities() == []
    assert graph.stats()["entities"] == 0
    assert graph.neighbors("a") == set()


# -- context assembly ---------------------------------------------------
def test_a_single_oversized_passage_is_truncated_to_the_budget():
    rag = GraphRAG(
        config=GraphRAGConfig(embedding_dim=64, chunk_size=4000, chunk_overlap=10),
        llm=ExtractiveLLM(),
    )
    rag.add_documents([" ".join(["neural networks"] * 2000)], source="unit")
    results = rag.retrieve("neural networks", top_k=1)
    context = rag.retriever.build_context(results, max_chars=500)

    assert results
    assert len(context) <= 500


def test_build_context_rejects_a_non_positive_budget():
    rag = GraphRAG(config=GraphRAGConfig(embedding_dim=64), llm=ExtractiveLLM())
    with pytest.raises(ValueError):
        rag.retriever.build_context([], max_chars=-1)


def test_retrieval_survives_a_query_with_no_usable_tokens():
    rag = GraphRAG(config=GraphRAGConfig(embedding_dim=64), llm=ExtractiveLLM())
    rag.add_documents(["Deep learning uses neural networks."], source="unit")

    assert rag.retrieve("!!! ??? ...") == []
    assert rag.query("!!! ??? ...")["answer"]
