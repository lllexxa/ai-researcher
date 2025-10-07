from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx

from .cite import Reference, map_references
from .deps import Settings
from .embed import EmbeddingBackend, embed_corpus, query_similar
from .export import ExportResult, export_review
from .gen import GeminiGenerator
from .retrieval import Paper, retrieve_papers


@dataclass
class ReviewBundle:
    markdown: str
    references: list[Reference]
    markdown_path: Path
    docx_path: Path
    latency_ms: int
    total_candidates: int
    source_counts: dict[str, int]
    selected_source_counts: dict[str, int]
    style: str


async def build_review(
    topic: str,
    description: str,
    style: str,
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
    embed_backend: EmbeddingBackend,
    generator: GeminiGenerator,
) -> ReviewBundle:
    start = time.perf_counter()
    candidates = await retrieve_papers(topic, description, settings, http_client)
    if not candidates:
        raise ValueError("No sources were retrieved for this query. Please refine your request.")

    matrix = embed_corpus(candidates, embed_backend)
    query = f"{topic}. {description}"
    ranked = query_similar(query, candidates, matrix, embed_backend, settings.top_k)
    selected = [result.paper for result in ranked] or candidates[: settings.top_k]

    total_candidates = len(candidates)
    source_counts = dict(Counter(paper.source for paper in candidates))
    selected_source_counts = dict(Counter(paper.source for paper in selected))

    markdown = await generator.generate_review(topic, description, selected, style=style)
    markdown, references = map_references(markdown, selected, style=style)
    exports: ExportResult = export_review(markdown, references, settings.downloads_path)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return ReviewBundle(
        markdown=markdown,
        references=references,
        markdown_path=exports.markdown_path,
        docx_path=exports.docx_path,
        latency_ms=elapsed_ms,
        total_candidates=total_candidates,
        source_counts=source_counts,
        selected_source_counts=selected_source_counts,
        style=style,
    )
