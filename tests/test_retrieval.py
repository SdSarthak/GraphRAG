import numpy as np

from graphrag.config import GraphRAGConfig
from graphrag.documents import Chunk
from graphrag.embeddings import HashingEmbedder, cosine_similarity
from graphrag.extraction import RuleBasedExtractor
from graphrag.graph import KnowledgeGraph
from graphrag.retrieval import (
    BM25Index,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from graphrag.text import content_tokens
from graphrag.vectorstore import VectorStore

TEXTS = {
    "c1": "Machine learning lets computers learn patterns from data.",
    "c2": "Deep learning uses neural networks with many layers.",
    "c3": "Photosynthesis converts sunlight into chemical energy in plants.",
}


def build_retriever() -> HybridRetriever:
    config = GraphRAGConfig(embedding_dim=256, retrieval_top_k=2)
    embedder = HashingEmbedder(dim=config.embedding_dim)
    extractor = RuleBasedExtractor(max_entities=8)

    chunks = {
        key: Chunk(text=text, document_id="doc", source="unit", index=index, id=key)
        for index, (key, text) in enumerate(TEXTS.items())
    }
    store = VectorStore(dim=config.embedding_dim)
    store.add(list(chunks), embedder.embed([chunk.text for chunk in chunks.values()]))

    bm25 = BM25Index()
    bm25.fit(
        list(chunks), [content_tokens(chunk.text) for chunk in chunks.values()]
    )

    graph = KnowledgeGraph()
    for key, chunk in chunks.items():
        graph.add_extraction(key, extractor.extract(chunk.text))

    return HybridRetriever(
        config=config,
        embedder=embedder,
        vector_store=store,
        bm25=bm25,
        graph=graph,
        chunks=chunks,
        extractor=extractor,
    )


# -- embeddings ---------------------------------------------------------
def test_embeddings_are_normalised_and_deterministic():
    embedder = HashingEmbedder(dim=128)
    first = embedder.embed_one("machine learning")
    second = embedder.embed_one("machine learning")

    assert first.shape == (128,)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_embedding_of_empty_text_is_zero():
    vector = HashingEmbedder(dim=64).embed_one("")
    assert np.allclose(vector, 0.0)


def test_similar_texts_score_higher_than_unrelated_ones():
    embedder = HashingEmbedder(dim=512)
    query = embedder.embed_one("neural networks for deep learning")
    related = embedder.embed_one("deep learning uses neural networks")
    unrelated = embedder.embed_one("photosynthesis in plants")

    assert float(query @ related) > float(query @ unrelated)


def test_cosine_similarity_handles_empty_matrix():
    scores = cosine_similarity(np.zeros((0, 4), dtype=np.float32), np.ones(4))
    assert scores.shape == (0,)


# -- vector store -------------------------------------------------------
def test_vector_store_search_and_persistence(tmp_path):
    embedder = HashingEmbedder(dim=128)
    store = VectorStore(dim=128)
    store.add(["a", "b"], embedder.embed(["neural networks", "banana bread recipe"]))

    hits = store.search(embedder.embed_one("neural networks"), top_k=1)
    assert hits[0][0] == "a"

    path = tmp_path / "vectors.npz"
    store.save(path)
    restored = VectorStore.load(path)

    assert restored.ids == store.ids
    assert "a" in restored
    assert restored.search(embedder.embed_one("neural networks"), top_k=1)[0][0] == "a"


def test_vector_store_replaces_existing_ids():
    store = VectorStore(dim=4)
    store.add(["a"], np.ones((1, 4), dtype=np.float32))
    store.add(["a"], np.zeros((1, 4), dtype=np.float32))

    assert len(store) == 1
    assert np.allclose(store.get("a"), 0.0)


def test_empty_vector_store_search_returns_nothing():
    assert VectorStore(dim=8).search(np.ones(8), top_k=3) == []


# -- bm25 ---------------------------------------------------------------
def test_bm25_ranks_the_matching_document_first():
    index = BM25Index()
    index.fit(
        ["a", "b"],
        [content_tokens(TEXTS["c1"]), content_tokens(TEXTS["c3"])],
    )
    hits = index.search(content_tokens("machine learning patterns"), top_k=2)

    assert hits[0][0] == "a"


def test_bm25_ignores_unknown_terms():
    index = BM25Index()
    index.fit(["a"], [content_tokens(TEXTS["c1"])])
    assert index.search(content_tokens("quantum chromodynamics"), top_k=3) == []


def test_bm25_round_trip(tmp_path):
    index = BM25Index()
    index.fit(["a", "b"], [content_tokens(TEXTS["c1"]), content_tokens(TEXTS["c2"])])
    path = tmp_path / "bm25.json"
    index.save(path)
    restored = BM25Index.load(path)

    assert restored.ids == index.ids
    assert restored.search(content_tokens("neural networks"), top_k=1)[0][0] == "b"


# -- fusion -------------------------------------------------------------
def test_rrf_rewards_agreement_across_signals():
    fused = reciprocal_rank_fusion({"vector": ["a", "b"], "keyword": ["b", "a"]})
    assert set(fused) == {"a", "b"}

    fused_agree = reciprocal_rank_fusion({"vector": ["a", "b"], "keyword": ["a", "b"]})
    assert fused_agree["a"]["total"] > fused_agree["b"]["total"]


def test_rrf_skips_zero_weighted_signals():
    fused = reciprocal_rank_fusion(
        {"vector": ["a"], "graph": ["b"]}, weights={"graph": 0.0}
    )
    assert set(fused) == {"a"}


# -- hybrid retriever ---------------------------------------------------
def test_retriever_returns_relevant_chunks_with_signal_breakdown():
    retriever = build_retriever()
    results = retriever.retrieve("What are neural networks used for?")

    assert results
    assert results[0].chunk.id == "c2"
    assert results[0].signals
    assert all(result.score > 0 for result in results)


def test_retriever_respects_top_k():
    results = build_retriever().retrieve("learning", top_k=1)
    assert len(results) == 1


def test_retriever_ignores_unrelated_query_terms():
    results = build_retriever().retrieve("machine learning from data")
    assert "c3" not in [result.chunk.id for result in results]


def test_retriever_handles_blank_query():
    assert build_retriever().retrieve("   ") == []


def test_query_entities_are_grounded_in_the_graph():
    retriever = build_retriever()
    seeds = retriever.query_entities("deep learning and neural networks")

    assert seeds
    assert all(retriever.graph.has_entity(seed) for seed in seeds)


def test_query_entities_survive_rephrasing():
    """A question that wraps an entity in extra words must still seed the graph."""
    retriever = build_retriever()
    seeds = retriever.query_entities("How exactly do neural networks work here?")

    assert "neural networks" in seeds


def test_graph_signal_contributes_to_the_fused_ranking():
    retriever = build_retriever()
    results = retriever.retrieve("How do neural networks support deep learning?")

    assert any("graph" in result.signals for result in results)


def test_build_context_is_numbered_and_truncated():
    retriever = build_retriever()
    results = retriever.retrieve("neural networks", top_k=2)
    context = retriever.build_context(results)

    assert context.startswith("[1]")
    short = retriever.build_context(results, max_chars=10)
    assert short.count("[") >= 1
