from graphrag.documents import (
    Chunk,
    Document,
    chunk_document,
    chunk_text,
    coerce_documents,
    load_documents,
)


def test_chunk_text_short_input_is_single_chunk():
    chunks = chunk_text("one two three", chunk_size=10, chunk_overlap=2)
    assert chunks == ["one two three"]


def test_chunk_text_empty_input():
    assert chunk_text("   ", chunk_size=10, chunk_overlap=2) == []


def test_chunk_text_windows_overlap():
    words = " ".join(str(i) for i in range(25))
    chunks = chunk_text(words, chunk_size=10, chunk_overlap=4)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 10 for chunk in chunks)
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-4:] == second[:4]  # overlap is preserved
    assert chunks[-1].split()[-1] == "24"  # tail is not dropped


def test_chunk_text_rejects_bad_overlap():
    try:
        chunk_text("a b c", chunk_size=5, chunk_overlap=5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= chunk_size")


def test_document_ids_are_stable_and_content_addressed():
    a = Document(text="hello  world", source="s")
    b = Document(text="hello world", source="s")
    c = Document(text="different", source="s")

    assert a.id == b.id  # whitespace is normalised before hashing
    assert a.id != c.id


def test_chunk_document_assigns_indexes():
    document = Document(text=" ".join(str(i) for i in range(50)), source="unit")
    chunks = chunk_document(document, chunk_size=20, chunk_overlap=5)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_coerce_documents_accepts_mixed_input():
    docs = coerce_documents(
        ["plain string", Document(text="a document"), {"text": "from dict"}]
    )
    assert [doc.text for doc in docs] == ["plain string", "a document", "from dict"]


def test_round_trip_serialisation():
    chunk = Chunk(text="body", document_id="doc-1", source="file.md", index=3)
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored.id == chunk.id
    assert restored.index == 3


def test_load_documents_reads_a_directory(tmp_path):
    (tmp_path / "a.txt").write_text("first document", encoding="utf-8")
    (tmp_path / "b.md").write_text("second document", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert {doc.text for doc in documents} == {"first document", "second document"}
