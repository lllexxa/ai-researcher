from pathlib import Path

import httpx
import numpy as np
import pytest

from app.deps import Settings
from app.embed import EmbeddingBackend
from app.retrieval import Paper
from app.service import ReviewBundle, build_review


@pytest.mark.asyncio
async def test_build_review_exports_files(monkeypatch, tmp_path):
    papers = [
        Paper(
            identifier="paper-1",
            title="Test Paper",
            abstract="This study explores testing strategies.",
            authors=["Ada Lovelace", "Grace Hopper"],
            year=2024,
            url="https://example.com/test",
            source="arxiv",
            doi="10.5555/test",
        ),
        Paper(
            identifier="paper-2",
            title="Second Paper",
            abstract="Another abstract for diversity.",
            authors=["Alan Turing"],
            year=2022,
            url="https://example.com/second",
            source="semantic_scholar",
        ),
    ]

    async def fake_retrieve(topic, description, settings, http_client):
        return papers

    monkeypatch.setattr("app.service.retrieve_papers", fake_retrieve)
    monkeypatch.setattr(
        "app.service.embed_corpus",
        lambda corpus, backend: np.array([[1.0], [0.5]], dtype=np.float32),
    )

    class DummyResult:
        def __init__(self, paper, score):
            self.paper = paper
            self.score = score

    monkeypatch.setattr(
        "app.service.query_similar",
        lambda query, corpus, matrix, backend, top_k: [DummyResult(papers[0], 0.9)],
    )

    class DummyGenerator:
        async def generate_review(self, topic, description, selected, *, style="apa"):
            return "# Introduction\nTesting pipeline [1]\n\n- Insight\n"

    settings = Settings(downloads_dir=str(tmp_path), top_k=1)

    async with httpx.AsyncClient() as client:
        bundle: ReviewBundle = await build_review(
            "Testing AI",
            "Focus on evaluation frameworks",
            "apa",
            settings=settings,
            http_client=client,
            embed_backend=EmbeddingBackend("all-MiniLM-L6-v2"),
            generator=DummyGenerator(),
        )

    assert "Testing pipeline" in bundle.markdown
    assert bundle.references
    assert all(ref.style == "apa" for ref in bundle.references)
    assert bundle.style == "apa"
    assert bundle.latency_ms >= 0
    assert "## References" in bundle.markdown
    assert "[https://example.com/test](https://example.com/test)" in bundle.markdown

    md_path = Path(bundle.markdown_path)
    docx_path = Path(bundle.docx_path)
    assert md_path.exists() and md_path.suffix == ".md"
    assert docx_path.exists() and docx_path.suffix == ".docx"
