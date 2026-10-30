import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.rag import chunk_text

client = TestClient(main.app)


def test_chunk_text_overlap_and_size():
    chunks = chunk_text("a" * 1000, chunk_size=400, overlap=100)
    assert all(len(c) <= 400 for c in chunks)
    assert len(chunks) == 3


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)


def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_success(monkeypatch):
    monkeypatch.setattr(main, "answer_question",
                        lambda q, k: {"answer": "42", "sources": []})
    r = client.post("/ask", json={"question": "What is a smart meter?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "42"


def test_ask_empty_question_rejected():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_index_missing_returns_503(monkeypatch):
    def boom(q, k):
        raise RuntimeError("Index not built")
    monkeypatch.setattr(main, "answer_question", boom)
    r = client.post("/ask", json={"question": "What is a smart meter?"})
    assert r.status_code == 503
