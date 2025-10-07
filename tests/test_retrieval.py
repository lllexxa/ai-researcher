import asyncio
import re

import httpx
import pytest
import respx

from app.deps import Settings
from app.retrieval import retrieve_papers


@pytest.mark.asyncio
async def test_retrieve_papers_combines_and_deduplicates():
    settings = Settings(arxiv_max_results=5)
    arxiv_feed = """<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry>
        <id>http://arxiv.org/abs/1234.5678</id>
        <title>Sample Paper One</title>
        <summary>This abstract discusses transformers for AI.</summary>
        <author><name>Jane Doe</name></author>
        <published>2024-01-01T00:00:00Z</published>
        <link rel='alternate' type='text/html' href='http://arxiv.org/abs/1234.5678v1'/>
        <link title='pdf' rel='related' type='application/pdf' href='http://arxiv.org/pdf/1234.5678v1'/>
      </entry>
    </feed>
    """
    semantic_payload = {
        "data": [
            {
                "paperId": "abcdef",
                "title": "Sample Paper One",
                "abstract": "Different summary, same DOI should be deduped.",
                "year": 2024,
                "authors": [{"name": "Jane Doe"}],
                "url": "https://example.com/sample",
                "externalIds": {"DOI": "10.1000/sample"},
            },
            {
                "paperId": "ghijk",
                "title": "Another Paper",
                "abstract": "Second paper abstract.",
                "year": 2023,
                "authors": [{"name": "John Roe"}],
                "url": "https://example.com/another",
                "externalIds": {},
            },
        ]
    }

    openalex_payload = {
        "results": [
            {
                "display_name": "Journal Paper",
                "abstract_inverted_index": {"journal": [0], "study": [1]},
                "authorships": [{"author": {"display_name": "J. Author"}}],
                "publication_year": 2022,
                "ids": {"openalex": "https://openalex.org/WXYZ"},
                "primary_location": {"landing_page_url": "https://example.com/journal"},
            }
        ]
    }

    async with respx.MockRouter(assert_all_called=False) as router:
        router.get(re.compile(r"https://export\.arxiv\.org/api/query.*")).mock(
            return_value=httpx.Response(200, text=arxiv_feed)
        )
        router.get(re.compile(r"https://api\.semanticscholar\.org/graph/v1/paper/search.*")).mock(
            return_value=httpx.Response(200, json=semantic_payload)
        )
        router.get(re.compile(r"https://api\.openalex\.org/works.*")).mock(
            return_value=httpx.Response(200, json=openalex_payload)
        )
        async with httpx.AsyncClient() as client:
            papers = await retrieve_papers("Transformers", "Research on AI transformers", settings, client)

    assert len(papers) >= 2
    identifiers = {paper.identifier for paper in papers}
    assert len(identifiers) == len(papers)
    assert any(paper.source == "arxiv" for paper in papers)
    assert any(paper.source == "semantic_scholar" for paper in papers)


@pytest.mark.asyncio
async def test_retrieve_papers_cache_hit_skips_network(monkeypatch):
    settings = Settings(
        arxiv_max_results=5,
        retrieval_cache_ttl_seconds=3600,
        retrieval_cache_max_entries=8,
    )
    arxiv_feed = """<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry>
        <id>http://arxiv.org/abs/9999.8888</id>
        <title>Caching Paper</title>
        <summary>Cache behaviour description.</summary>
        <author><name>Alice Smith</name></author>
        <published>2023-05-01T00:00:00Z</published>
        <link rel='alternate' type='text/html' href='http://arxiv.org/abs/9999.8888v1'/>
        <link title='pdf' rel='related' type='application/pdf' href='http://arxiv.org/pdf/9999.8888v1'/>
      </entry>
    </feed>
    """
    semantic_payload = {
        "data": [
            {
                "paperId": "cache123",
                "title": "Caching Paper",
                "abstract": "Same DOI should dedupe.",
                "year": 2023,
                "authors": [{"name": "Alice Smith"}],
                "url": "https://example.com/cache",
                "externalIds": {"DOI": "10.4242/cache"},
            }
        ]
    }

    openalex_payload = {
        "results": [
            {
                "display_name": "Journal Cache Paper",
                "abstract_inverted_index": {"cache": [0], "entry": [1]},
                "authorships": [{"author": {"display_name": "Cache Author"}}],
                "publication_year": 2023,
                "ids": {"openalex": "https://openalex.org/WCACHE"},
                "primary_location": {"landing_page_url": "https://example.com/cache"},
            }
        ]
    }

    async with respx.MockRouter(assert_all_called=False) as router:
        router.get(re.compile(r"https://export\.arxiv\.org/api/query.*")).mock(
            return_value=httpx.Response(200, text=arxiv_feed)
        )
        router.get(re.compile(r"https://api\.semanticscholar\.org/graph/v1/paper/search.*")).mock(
            return_value=httpx.Response(200, json=semantic_payload)
        )
        router.get(re.compile(r"https://api\.openalex\.org/works.*")).mock(
            return_value=httpx.Response(200, json=openalex_payload)
        )
        async with httpx.AsyncClient() as client:
            first_pass = await retrieve_papers("Cache Test", "A test query", settings, client)
            assert first_pass

        # Replace routes with assertions so any further HTTP call would fail the test
        router.routes.clear()
        router.get(re.compile(r"https://export\.arxiv\.org/api/query.*")).mock(
            side_effect=AssertionError("arXiv endpoint should not be called on cache hit")
        )
        router.get(re.compile(r"https://api\.semanticscholar\.org/graph/v1/paper/search.*")).mock(
            side_effect=AssertionError("Semantic Scholar endpoint should not be called on cache hit")
        )
        router.get(re.compile(r"https://api\.openalex\.org/works.*")).mock(
            side_effect=AssertionError("OpenAlex endpoint should not be called on cache hit")
        )
        async with httpx.AsyncClient() as client:
            second_pass = await retrieve_papers("Cache Test", "A test query", settings, client)

    assert [paper.identifier for paper in second_pass] == [paper.identifier for paper in first_pass]


@pytest.mark.asyncio
async def test_semantic_scholar_falls_back_to_openalex():
    settings = Settings(arxiv_max_results=5, semantic_retry_attempts=1)
    arxiv_feed = """<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry>
        <id>http://arxiv.org/abs/1111.2222</id>
        <title>Arxiv Paper</title>
        <summary>Arxiv abstract.</summary>
        <author><name>John Smith</name></author>
        <published>2021-01-01T00:00:00Z</published>
        <link rel='alternate' type='text/html' href='http://arxiv.org/abs/1111.2222v1'/>
        <link title='pdf' rel='related' type='application/pdf' href='http://arxiv.org/pdf/1111.2222v1'/>
      </entry>
    </feed>
    """
    openalex_payload = {
        "results": [
            {
                "display_name": "OpenAlex Fallback Paper",
                "abstract_inverted_index": {"fallback": [0], "paper": [1]},
                "authorships": [{"author": {"display_name": "Fallback Author"}}],
                "publication_year": 2024,
                "ids": {"openalex": "https://openalex.org/W123"},
                "primary_location": {"landing_page_url": "https://example.com/fallback"},
            }
        ]
    }

    async with respx.MockRouter(assert_all_called=False) as router:
        router.get(re.compile(r"https://export\.arxiv\.org/api/query.*")).mock(
            return_value=httpx.Response(200, text=arxiv_feed)
        )
        router.get(re.compile(r"https://api\.semanticscholar\.org/graph/v1/paper/search.*")).mock(
            return_value=httpx.Response(429, json={"message": "rate limited"})
        )
        router.get(re.compile(r"https://api\.openalex\.org/works.*")).mock(
            return_value=httpx.Response(200, json=openalex_payload)
        )

        async with httpx.AsyncClient() as client:
            papers = await retrieve_papers(
                "Fallback Query",
                "Testing fallback for semantic scholar",
                settings,
                client,
            )

    assert any(paper.source == "openalex" for paper in papers)
