from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx
from rank_bm25 import BM25Okapi

from .deps import Settings


logger = logging.getLogger(__name__)

_CACHE: dict[Tuple[str, str, int, str], tuple[float, List["Paper"]]] = {}
_CACHE_LOCK = asyncio.Lock()


_STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "that",
    "with",
    "this",
    "have",
    "has",
    "are",
    "was",
    "were",
    "about",
    "using",
    "into",
    "onto",
    "between",
    "over",
    "under",
    "while",
    "where",
    "your",
    "their",
    "there",
    "which",
    "shall",
    "will",
    "can",
    "could",
    "would",
    "should",
    "these",
    "those",
    "such",
    "than",
    "also",
    "been",
    "within",
    "each",
    "other",
    "some",
    "more",
    "most",
    "many",
    "much",
}


def _build_cache_key(topic: str, description: str, settings: Settings) -> Tuple[str, str, int, str]:
    return (
        topic.strip().lower(),
        description.strip().lower(),
        settings.arxiv_max_results,
        (settings.semantic_scholar_api_key or "").strip(),
    )


async def _get_cached_papers(key: Tuple[str, str, int, str]) -> Optional[List["Paper"]]:
    async with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if not cached:
            return None
        expires_at, papers = cached
        if expires_at < time.monotonic():
            del _CACHE[key]
            return None
        return copy.deepcopy(papers)


async def _store_cached_papers(
    key: Tuple[str, str, int, str],
    papers: List["Paper"],
    settings: Settings,
) -> None:
    ttl = max(0, settings.retrieval_cache_ttl_seconds)
    if ttl == 0 or not papers:
        return
    max_entries = max(1, settings.retrieval_cache_max_entries)
    entry = (time.monotonic() + ttl, copy.deepcopy(papers))
    async with _CACHE_LOCK:
        _CACHE[key] = entry
        if len(_CACHE) > max_entries:
            oldest_key = min(_CACHE.items(), key=lambda item: item[1][0])[0]
            if oldest_key in _CACHE:
                del _CACHE[oldest_key]


@dataclass
class Paper:
    identifier: str
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    url: str
    source: str
    doi: Optional[str] = None
    score: float = 0.0

    def tokenized(self) -> List[str]:
        body = f"{self.title} {self.abstract}".lower()
        return [token for token in re.findall(r"[a-z0-9\-]+", body) if len(token) > 2]


def build_keywords(*chunks: str) -> List[str]:
    merged = " ".join(chunks).lower()
    tokens = re.findall(r"[a-z0-9\-]+", merged)
    seen = set()
    keywords: List[str] = []
    for token in tokens:
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords[:32]


def _keyword_score(text: str, keywords: Iterable[str]) -> float:
    text_tokens = re.findall(r"[a-z0-9\-]+", text.lower())
    text_set = set(text_tokens)
    return sum(1.0 for kw in keywords if kw in text_set)


async def fetch_arxiv(
    keywords: List[str],
    settings: Settings,
    client: httpx.AsyncClient,
) -> List[Paper]:
    if not keywords:
        return []
    query = "+OR+".join(quote_plus(kw) for kw in keywords[:6])
    url = (
        "https://export.arxiv.org/api/query?search_query="
        f"(ti:{query})+(abs:{query})&max_results={settings.arxiv_max_results}"
    )
    try:
        response = await client.get(url, headers={"User-Agent": "literature-review-bot"})
        response.raise_for_status()
        feed = ET.fromstring(response.text)
    except httpx.RequestError as exc:
        logger.warning("arXiv request error: %s. Returning empty set.", exc)
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("arXiv returned HTTP %s. Returning empty set.", exc.response.status_code)
        return []
    except ET.ParseError as exc:
        logger.warning("arXiv returned invalid XML: %s. Returning empty set.", exc)
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results: List[Paper] = []
    for entry in feed.findall("atom:entry", ns):
        identifier = entry.findtext("atom:id", default="", namespaces=ns)
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
        authors = [a.findtext("atom:name", default="", namespaces=ns) or "" for a in entry.findall("atom:author", ns)]
        year_text = entry.findtext("atom:published", default="", namespaces=ns)
        try:
            year = int(year_text[:4]) if year_text else None
        except ValueError:
            year = None
        doi = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "doi":
                doi = link.attrib.get("href")
        pdf_link = next(
            (
                link.attrib.get("href")
                for link in entry.findall("atom:link", ns)
                if link.attrib.get("type") == "application/pdf"
            ),
            identifier,
        )
        results.append(
            Paper(
                identifier=(identifier or pdf_link or title or "arxiv")[:200],
                title=title,
                abstract=summary,
                authors=[a for a in authors if a],
                year=year,
                url=pdf_link,
                source="arxiv",
                doi=doi,
            )
        )
    logger.info("Fetched %d arXiv papers", len(results))
    return results


async def fetch_semantic_scholar(
    keywords: List[str],
    settings: Settings,
    client: httpx.AsyncClient,
) -> List[Paper]:
    if not keywords:
        return []
    limit = min(settings.arxiv_max_results, 100)
    query = quote_plus(" ".join(keywords[:10]))
    fields = "title,authors,year,abstract,url,externalIds"
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={query}&limit={limit}&fields={fields}"
    )
    headers = {"User-Agent": "literature-review-bot", "Accept": "application/json"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    attempts = max(1, settings.semantic_retry_attempts)
    response: httpx.Response | None = None
    for attempt in range(attempts):
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 403:
                logger.warning("Semantic Scholar returned 403 (forbidden). Continuing without S2.")
                return []
            if status == 429 and attempt + 1 < attempts:
                retry_after = exc.response.headers.get("Retry-After")
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = settings.semantic_retry_delay_seconds
                delay = max(delay, settings.semantic_retry_delay_seconds)
                logger.info(
                    "Semantic Scholar rate limit hit. Retrying in %.2f seconds (attempt %s/%s).",
                    delay,
                    attempt + 1,
                    attempts,
                )
                await asyncio.sleep(delay)
                continue
            if status in (429, 503):
                logger.warning("Semantic Scholar returned %s. Falling back to arXiv-only.", status)
                return []
            raise
        except httpx.RequestError as exc:
            logger.warning("Semantic Scholar request error: %s. Continuing without S2.", exc)
            if attempt + 1 < attempts:
                delay = settings.semantic_retry_delay_seconds
                logger.debug("Retrying Semantic Scholar in %.2f seconds due to request error.", delay)
                await asyncio.sleep(delay)
                continue
            return []
    else:
        return []

    if response is None:
        return []

    try:
        payload = response.json()
    except ValueError:
        logger.warning("Semantic Scholar returned non-JSON payload. Continuing without S2.")
        return []
    results: List[Paper] = []
    for item in payload.get("data", []):
        external = item.get("externalIds", {}) or {}
        identifier = external.get("ArXiv") or external.get("DOI") or item.get("paperId") or item.get("title")
        url_value = item.get("url") or (
            f"https://www.semanticscholar.org/p/{item.get('paperId')}" if item.get("paperId") else ""
        )
        results.append(
            Paper(
                identifier=str(identifier or "semantic_scholar"),
                title=(item.get("title") or "").strip(),
                abstract=(item.get("abstract") or "").strip(),
                authors=[author.get("name", "") for author in item.get("authors", [])],
                year=item.get("year"),
                url=url_value or "https://www.semanticscholar.org/",
                source="semantic_scholar",
                doi=external.get("DOI"),
            )
        )
    logger.info("Fetched %d Semantic Scholar papers", len(results))
    return results


def _abstract_from_inverted(index: Optional[dict]) -> str:
    if not index:
        return ""
    positions: List[Tuple[int, str]] = []
    for word, indexes in index.items():
        for position in indexes:
            positions.append((position, word))
    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


async def fetch_openalex(
    keywords: List[str],
    settings: Settings,
    client: httpx.AsyncClient,
) -> List[Paper]:
    if not keywords:
        return []
    query = " ".join(keywords[:8])
    params = {
        "search": query,
        "per-page": min(settings.arxiv_max_results, 20),
        "filter": "type:journal-article",
        "mailto": settings.openalex_mailto,
    }
    url = "https://api.openalex.org/works"
    try:
        response = await client.get(url, params=params, headers={"User-Agent": "literature-review-bot"})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("OpenAlex fallback failed: %s", exc)
        return []

    payload = response.json()
    results: List[Paper] = []
    for item in payload.get("results", []):
        title = (item.get("display_name") or "").strip()
        if not title:
            continue
        abstract = _abstract_from_inverted(item.get("abstract_inverted_index"))
        authors = [
            (auth.get("author", {}) or {}).get("display_name", "")
            for auth in item.get("authorships", [])
        ]
        year = item.get("publication_year")
        doi = None
        ids = item.get("ids") or {}
        doi = ids.get("doi")
        landing = None
        primary_location = item.get("primary_location") or {}
        landing = (primary_location.get("landing_page_url") or "").strip()
        identifier = ids.get("openalex") or landing or title
        results.append(
            Paper(
                identifier=str(identifier),
                title=title,
                abstract=abstract,
                authors=[name for name in authors if name],
                year=year,
                url=landing or ids.get("openalex") or "",
                source="openalex",
                doi=doi,
            )
        )
    if results:
        logger.info("OpenAlex fallback returned %d entries for query '%s'", len(results), query)
    return results


def dedupe_papers(papers: Iterable[Paper]) -> List[Paper]:
    seen: set[str] = set()
    filtered: List[Paper] = []
    for paper in papers:
        doi_key = (paper.doi or "").lower().strip()
        identifier_key = (paper.identifier or "").lower().strip()
        key = doi_key or identifier_key or f"{paper.source}:{len(filtered)}"
        if key in seen:
            continue
        seen.add(key)
        filtered.append(paper)
    return filtered


def select_relevant(papers: List[Paper], keywords: List[str], limit: int = 50) -> List[Paper]:
    if not papers:
        return []
    tokenized_corpus = [paper.tokenized() for paper in papers]
    query_tokens = keywords or [kw for kw in {token for doc in tokenized_corpus for token in doc}][:8]
    if not query_tokens:
        return papers[:limit]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    for paper, score in zip(papers, scores):
        keyword_boost = _keyword_score(f"{paper.title} {paper.abstract}", keywords)
        paper.score = float(score + keyword_boost)
    papers.sort(key=lambda p: p.score, reverse=True)
    return papers[:limit]


async def retrieve_papers(
    topic: str,
    description: str,
    settings: Settings,
    client: httpx.AsyncClient,
) -> List[Paper]:
    cache_key = _build_cache_key(topic, description, settings)
    cached = await _get_cached_papers(cache_key)
    if cached:
        logger.info("Retrieval cache hit for topic '%s'", topic)
        return cached

    keywords = build_keywords(topic, description)
    arxiv_task = asyncio.create_task(fetch_arxiv(keywords, settings, client))
    semantic_task = asyncio.create_task(fetch_semantic_scholar(keywords, settings, client))
    openalex_task = (
        asyncio.create_task(fetch_openalex(keywords, settings, client))
        if settings.openalex_enabled
        else None
    )

    results = await asyncio.gather(
        arxiv_task,
        semantic_task,
        openalex_task if openalex_task else asyncio.sleep(0, result=[]),
    )
    arxiv_results, semantic_results, openalex_results = results
    combined = dedupe_papers([*arxiv_results, *semantic_results, *openalex_results])
    top_candidates = select_relevant(combined, keywords, limit=min(50, settings.arxiv_max_results))
    if top_candidates:
        await _store_cached_papers(cache_key, top_candidates, settings)
    else:
        logger.warning("No papers retrieved for topic '%s'", topic)
    return top_candidates
