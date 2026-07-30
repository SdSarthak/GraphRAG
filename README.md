# GraphRAG

Graph-based Retrieval Augmented Generation.

Plain RAG retrieves passages that look like the question. That breaks down when
the answer depends on how things *relate* — "how is deep learning connected to
computer vision?" needs the link between two passages, not the passages that
best match the wording.

GraphRAG builds a knowledge graph over the corpus during indexing, then fuses
three retrieval signals at query time:

| Signal | What it catches | How |
| --- | --- | --- |
| **Dense** | Paraphrases and semantic similarity | Cosine over hashed embeddings |
| **Lexical** | Exact terms, names, rare tokens | Okapi BM25 |
| **Graph** | Multi-hop and relational context | Personalised PageRank seeded with the question's entities |

The three rankings are merged with weighted Reciprocal Rank Fusion, and the
top passages are handed to Claude as numbered, citable context.

It runs **fully offline by default** — no API key required. Generation then
uses a built-in extractive answerer. Set `ANTHROPIC_API_KEY` and it switches to
Claude automatically.

---

## Install

```bash
git clone https://github.com/SdSarthak/GraphRAG.git
cd GraphRAG
python -m venv venv && venv\Scripts\activate      # Windows
# python3 -m venv venv && source venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

Optional — enable Claude-backed answers:

```bash
cp .env.example .env      # then edit .env and set ANTHROPIC_API_KEY
```

Every setting in `.env.example` is optional; the defaults are the ones in
`graphrag/config.py`.

---

## Quick start

```bash
python main.py                       # end-to-end demo on a built-in corpus
python -m graphrag demo               # same thing via the CLI
```

Index your own documents and ask questions:

```bash
python -m graphrag index ./docs                       # .txt .md .rst .log
python -m graphrag query "How does X relate to Y?"
python -m graphrag stats
python -m graphrag entities "machine learning" --depth 2
python -m graphrag export graph.graphml               # open in Gephi
```

Useful flags:

```bash
python -m graphrag index ./docs --append              # add to an existing index
python -m graphrag index ./docs --extractor llm       # entity extraction via Claude
python -m graphrag query "..." --top-k 8 --json       # raw JSON with score breakdown
python -m graphrag query "..." --no-generate          # retrieval only
python -m graphrag --storage ./my-index stats         # custom index location
```

---

## Python API

```python
from graphrag import GraphRAG

rag = GraphRAG()
rag.add_documents([
    "Deep learning is a branch of machine learning that uses neural networks.",
    "Convolutional neural networks power most computer vision systems.",
])

result = rag.query("How is deep learning related to computer vision?")
print(result["answer"])
for hit in result["retrieved"]:
    print(hit["score"], hit["signals"], hit["text"][:80])

rag.save("storage")
rag = GraphRAG.load("storage")
```

`query()` returns:

| Key | Description |
| --- | --- |
| `answer` | Generated answer, citing passages as `[1]`, `[2]`, … |
| `sources` | Retrieved passage texts, best first |
| `entities` | Graph entities the question was anchored to |
| `context` | The exact numbered context handed to the model |
| `retrieved` | Per-passage score plus the per-signal breakdown |

Other entry points:

```python
rag.add_path("./docs")                       # index a directory
rag.retrieve("question", top_k=8)            # retrieval without generation
rag.related_entities("neural networks", 2)   # graph neighbourhood
rag.stats()                                  # corpus + graph statistics
rag.graph.communities()                      # greedy-modularity clusters
rag.graph.export_graphml("graph.graphml")
```

---

## How it works

**Indexing**

1. Documents are normalised and split into overlapping word windows
   (`chunk_size` / `chunk_overlap`).
2. Each chunk is embedded (`HashingEmbedder`) and added to the vector store and
   the BM25 index.
3. Each chunk is passed to an extractor:
   - `RuleBasedExtractor` (default) — proper-noun detection plus
     frequency-ranked key phrases, with relation types read off a verb lexicon
     (`uses`, `is_a`, `part_of`, `enables`, …). Deterministic and offline.
   - `LLMExtractor` — Claude with a strict JSON schema, falling back to the
     rule-based extractor if the call fails or returns nothing.
4. Entities and relations are merged into a `networkx` graph. Every node keeps
   the set of chunks it appeared in — that provenance is what turns graph
   structure back into retrievable documents.

**Querying**

1. Entities are extracted from the question and matched against graph nodes.
2. Personalised PageRank spreads mass from those seeds; chunks inherit the
   PageRank mass of the entities they contain.
3. Dense, lexical and graph rankings are fused with weighted RRF (rank-based,
   so the three incomparable score scales never need calibrating).
4. The top chunks are rendered as numbered context and answered by the
   configured LLM backend, which is instructed to cite passages and to say when
   the context does not contain the answer.

**Design notes**

- PageRank is implemented directly (`graph.personalized_pagerank`) rather than
  via `networkx.pagerank`, which delegates to SciPy — an unnecessarily heavy
  dependency for one power iteration.
- The default embedder is a deterministic hashing vectoriser: no downloads, no
  warm-up, reproducible across machines. Swap in any object exposing
  `embed(texts)` / `embed_one(text)` via `GraphRAG(embedder=...)`.
- Any signal can be turned off by setting its weight to `0`
  (e.g. `GRAPHRAG_GRAPH_WEIGHT=0` for a plain hybrid baseline) — useful for
  measuring what the graph is actually contributing.

---

## Configuration

Set values in `.env`, as `GRAPHRAG_*` environment variables, or in code:

```python
from graphrag import GraphRAG, GraphRAGConfig

config = GraphRAGConfig(chunk_size=120, retrieval_top_k=8, graph_weight=2.0)
rag = GraphRAG(config=config)
```

See `.env.example` for every supported variable.

---

## Project layout

```
graphrag/
  config.py       GraphRAGConfig + environment loading
  documents.py    Document/Chunk models, chunking, corpus loading
  text.py         Tokenisation, stopwords, relation-verb lexicon
  extraction.py   Rule-based and LLM entity/relation extraction
  graph.py        KnowledgeGraph, personalised PageRank, communities, export
  embeddings.py   HashingEmbedder + cosine similarity
  vectorstore.py  In-memory vector index with numpy persistence
  retrieval.py    BM25, reciprocal rank fusion, HybridRetriever
  llm.py          AnthropicLLM (Claude) and ExtractiveLLM (offline)
  pipeline.py     GraphRAG: indexing, querying, save/load
  cli.py          Command line interface
tests/            pytest suite
main.py           Demo entrypoint
```

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite covers chunking, extraction, graph construction and PageRank,
BM25/vector/hybrid retrieval, rank fusion, persistence round-trips and the CLI.
It runs entirely offline — no API key and no network access needed.

---

## License

MIT
