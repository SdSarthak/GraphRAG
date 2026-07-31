"""Failure modes around saving, loading and incremental indexing."""

import json

import numpy as np
import pytest

from graphrag import GraphRAG, GraphRAGConfig
from graphrag.graph import KnowledgeGraph
from graphrag.llm import ExtractiveLLM
from graphrag.pipeline import BM25_FILE, CHUNKS_FILE, GRAPH_FILE, VECTORS_FILE
from graphrag.retrieval import BM25Index
from graphrag.storage import atomic_write_text
from graphrag.text import content_tokens
from graphrag.vectorstore import VectorStore

DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
]


def build(tmp_path):
    rag = GraphRAG(config=GraphRAGConfig(embedding_dim=64), llm=ExtractiveLLM())
    rag.add_documents(DOCUMENTS, source="unit")
    return rag, rag.save(tmp_path / "index")


# -- vector store -------------------------------------------------------
def test_duplicate_ids_in_one_batch_do_not_duplicate_rows():
    store = VectorStore(dim=4)
    store.add(["a", "a", "b"], np.arange(12, dtype=np.float32).reshape(3, 4))

    assert store.ids == ["a", "b"]
    assert store.matrix.shape == (2, 4)
    # The last vector supplied for an id wins, as it does across batches.
    assert np.allclose(store.get("a"), [4, 5, 6, 7])
    assert len(store.search(np.array([4, 5, 6, 7], dtype=np.float32), top_k=5)) == 2


def test_add_accepts_a_single_flat_vector():
    store = VectorStore(dim=3)
    store.add(["a"], np.ones(3, dtype=np.float32))
    assert len(store) == 1


def test_add_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        VectorStore(dim=3).add(["a"], np.ones((1, 5), dtype=np.float32))


def test_loading_a_truncated_vector_store_is_reported(tmp_path):
    path = tmp_path / "vectors.npz"
    np.savez_compressed(
        path,
        ids=np.array(["a", "b", "c"], dtype=object),
        matrix=np.ones((2, 4), dtype=np.float32),
        dim=np.array([4]),
    )
    with pytest.raises(ValueError) as excinfo:
        VectorStore.load(path)
    assert "corrupt" in str(excinfo.value)


def test_loading_a_foreign_npz_is_reported(tmp_path):
    path = tmp_path / "other.npz"
    np.savez_compressed(path, something=np.ones(3))
    with pytest.raises(ValueError):
        VectorStore.load(path)


# -- graph --------------------------------------------------------------
def test_dangling_edges_do_not_create_phantom_entities():
    graph = KnowledgeGraph.from_dict(
        {
            "nodes": [{"id": "deep learning", "chunks": ["c1"]}],
            "edges": [{"source": "deep learning", "target": "ghost"}],
        }
    )
    assert graph.entities == ["deep learning"]
    assert graph.graph.number_of_edges() == 0


def test_graph_load_reports_invalid_json(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        KnowledgeGraph.load(path)


# -- bm25 ---------------------------------------------------------------
def test_bm25_add_matches_a_full_refit():
    incremental = BM25Index()
    for index, text in enumerate(DOCUMENTS):
        incremental.add([str(index)], [content_tokens(text)])

    batch = BM25Index()
    batch.fit(
        [str(index) for index in range(len(DOCUMENTS))],
        [content_tokens(text) for text in DOCUMENTS],
    )

    query = content_tokens("neural networks learning")
    assert incremental.search(query, top_k=5) == batch.search(query, top_k=5)
    assert incremental.avg_length == batch.avg_length


def test_bm25_add_replaces_an_existing_document():
    index = BM25Index()
    index.add(["a"], [content_tokens("photosynthesis in plants")])
    index.add(["a"], [content_tokens("neural networks and deep learning")])

    assert len(index) == 1
    assert index.search(content_tokens("photosynthesis"), top_k=3) == []
    assert index.search(content_tokens("neural"), top_k=3)[0][0] == "a"


def test_bm25_fit_discards_previous_documents():
    index = BM25Index()
    index.fit(["a"], [content_tokens("photosynthesis in plants")])
    index.fit(["b"], [content_tokens("neural networks")])

    assert index.ids == ["b"]
    assert index.search(content_tokens("photosynthesis"), top_k=3) == []


def test_bm25_from_dict_rejects_a_corrupt_payload():
    with pytest.raises(ValueError):
        BM25Index.from_dict({"ids": ["a", "b"], "doc_freqs": [{"x": 1}]})


# -- index round trip ---------------------------------------------------
def test_incremental_indexing_keeps_every_store_in_step(tmp_path):
    rag = GraphRAG(config=GraphRAGConfig(embedding_dim=64), llm=ExtractiveLLM())
    for document in DOCUMENTS:
        rag.add_documents([document], source="unit")
    rag.add_documents(["Reinforcement learning trains agents with rewards."])

    assert len(rag.bm25) == len(rag.chunks) == len(rag.vector_store)
    assert rag.retrieve("neural networks")


def test_duplicate_documents_in_one_call_are_indexed_once():
    rag = GraphRAG(config=GraphRAGConfig(embedding_dim=64), llm=ExtractiveLLM())
    added = rag.add_documents([DOCUMENTS[0], DOCUMENTS[0]], source="unit")

    assert added == len(rag.chunks)
    assert len(rag.vector_store) == len(rag.chunks)


def test_a_failing_extractor_does_not_lose_the_chunk(caplog):
    class _Broken:
        def extract(self, text):
            raise RuntimeError("extractor exploded")

    rag = GraphRAG(
        config=GraphRAGConfig(embedding_dim=64),
        llm=ExtractiveLLM(),
        extractor=_Broken(),
    )
    added = rag.add_documents(DOCUMENTS, source="unit")

    assert added == len(rag.chunks)
    assert len(rag.vector_store) == len(rag.chunks) == len(rag.bm25)
    # Retrieval still works through the vector and keyword signals.
    assert rag.retrieve("neural networks")


def test_a_failing_embedder_leaves_no_half_indexed_chunks():
    class _Broken:
        name = "broken"
        dim = 64

        def embed(self, texts):
            raise RuntimeError("embedder exploded")

        def embed_one(self, text):
            raise RuntimeError("embedder exploded")

    rag = GraphRAG(
        config=GraphRAGConfig(embedding_dim=64),
        llm=ExtractiveLLM(),
        embedder=_Broken(),
    )
    with pytest.raises(RuntimeError):
        rag.add_documents(DOCUMENTS, source="unit")

    # Nothing may be recorded as indexed, otherwise a retry silently skips it.
    assert rag.chunks == {}
    assert rag.documents == {}
    assert len(rag.bm25) == 0


def test_load_lists_every_missing_file(tmp_path):
    _, target = build(tmp_path)
    (target / GRAPH_FILE).unlink()
    (target / BM25_FILE).unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        GraphRAG.load(target)
    message = str(excinfo.value)
    assert GRAPH_FILE in message and BM25_FILE in message


def test_load_rejects_an_embedding_dimension_mismatch(tmp_path):
    _, target = build(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        GraphRAG.load(target, config=GraphRAGConfig(embedding_dim=128))
    assert "re-index" in str(excinfo.value)


def test_load_reports_corrupt_chunk_metadata(tmp_path):
    _, target = build(tmp_path)
    (target / CHUNKS_FILE).write_text(
        json.dumps({"documents": [], "chunks": [{"text": "no id here"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        GraphRAG.load(target, llm=ExtractiveLLM())


def test_load_reports_invalid_json(tmp_path):
    _, target = build(tmp_path)
    (target / CHUNKS_FILE).write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError):
        GraphRAG.load(target, llm=ExtractiveLLM())


def test_saving_over_an_index_twice_leaves_no_temporary_files(tmp_path):
    rag, target = build(tmp_path)
    rag.save(target)

    assert not [item.name for item in target.iterdir() if item.suffix == ".tmp"]
    assert len(GraphRAG.load(target, llm=ExtractiveLLM()).chunks) == len(rag.chunks)


def test_atomic_write_keeps_the_previous_file_when_the_writer_fails(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_text(path, "original")

    def explode(_stream):
        raise RuntimeError("disk full")

    from graphrag.storage import atomic_write_binary

    with pytest.raises(RuntimeError):
        atomic_write_binary(path, explode)

    assert path.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_vector_store_round_trip_survives_a_reload(tmp_path):
    rag, target = build(tmp_path)
    restored = GraphRAG.load(target, llm=ExtractiveLLM())

    assert restored.vector_store.ids == rag.vector_store.ids
    assert len(restored.bm25) == len(rag.bm25)
    original = rag.query("What is deep learning?")
    assert restored.query("What is deep learning?")["sources"] == original["sources"]


def test_reloaded_index_can_be_extended(tmp_path):
    _, target = build(tmp_path)
    restored = GraphRAG.load(target, llm=ExtractiveLLM())
    added = restored.add_documents(["Knowledge graphs link related entities."])

    assert added > 0
    assert len(restored.bm25) == len(restored.chunks) == len(restored.vector_store)
    assert restored.retrieve("knowledge graphs")


def test_vectors_file_is_written_where_it_is_expected(tmp_path):
    _, target = build(tmp_path)
    assert (target / VECTORS_FILE).is_file()
