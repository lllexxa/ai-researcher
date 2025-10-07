# Literature Review Companion

![CI](https://img.shields.io/github/actions/workflow/status/your-org/ai-literature-review/ci.yml?branch=main)

Production-ready FastAPI service that assembles scholarly sources from arXiv and Semantic Scholar, synthesises them with the Gemini API, and serves a citation-aware literature review with Markdown and DOCX exports. The frontend is a single Jinja2 page enhanced with HTMX and Tailwind.

## Architecture at a Glance

```
Browser (HTMX form)
        |
        v
FastAPI app --> Retrieval (arXiv + Semantic Scholar via httpx)
        |             |
        |             --> Scoring (BM25 + FAISS embeddings)
        |
        --> Gemini generation (structured prompt with citations)
                     |
                     --> Citation mapper + exporters (Markdown / DOCX)
                                    |
                                    --> Downloads served via /downloads
```

Key components live under `app/`:
- `main.py` - FastAPI endpoints, HTMX-ready partials, rate limiting.
- `retrieval.py` - keyword expansion, arXiv & Semantic Scholar fetchers, dedupe + BM25 scoring.
- `embed.py` - SentenceTransformer/FAISS embedding pipeline with TF-IDF fallback.
- `gen.py` - Gemini prompt builder and async wrapper.
- `cite.py` - placeholder verification and reference formatting.
- `export.py` - Markdown + DOCX writers for downloadable artifacts.
- `service.py` - end-to-end orchestration for API and UI calls.

## Quick Start

### Prerequisites
- Python 3.11+
- [Google Gemini API key](https://ai.google.dev/)
- Optional: Semantic Scholar API key for higher rate limits

### 1. Configure environment

```bash
cp .env.example .env
# edit .env and add GEMINI_API_KEY=<your key>
```

### 2. Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit http://127.0.0.1:8000 and submit a topic + description.

### 3. Run with Docker

```bash
docker compose up --build
```

Download files appear under `app/static/downloads` and are exposed at `/downloads/<file>`.

## Environment Variables

| Name | Description | Default |
| ---- | ----------- | ------- |
| `GEMINI_API_KEY` | Required. Google Gemini API key. | - |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional. Boosts Semantic Scholar quotas. | empty |
| `ARXIV_MAX_RESULTS` | Max candidates fetched from each source. | `50` |
| `MODEL_EMBED` | SentenceTransformer CPU model. | `all-MiniLM-L6-v2` |
| `MODEL_GEMINI` | Gemini model name (fallbacks try related variants). | `gemini-1.5-flash-8b` |
| `TOP_K` | Papers sent to Gemini after vector search. | `20` |
| `RETRIEVAL_CACHE_TTL_SECONDS` | Cache lifetime for combined retrieval results. | `900` |
| `RETRIEVAL_CACHE_MAX_ENTRIES` | Maximum cached queries kept in memory. | `64` |
| `SEMANTIC_RETRY_ATTEMPTS` | How many times to retry Semantic Scholar on 429 errors. | `2` |
| `SEMANTIC_RETRY_DELAY_SECONDS` | Backoff (seconds) between S2 retries when rate-limited. | `1.5` |
| `OPENALEX_ENABLED` | Enable OpenAlex queries for broader journal coverage. | `true` |
| `OPENALEX_MAILTO` | Contact email sent to OpenAlex (recommended). | `literature-review@example.com` |

## API Reference

### `POST /api/review`
Request body:
```json
{
  "topic": "string (5-200 chars)",
  "description": "string (20-1500 chars)",
  "style": "apa | ieee | chicago"
}
```
Response body:
```json
{
  "review_md": "Literature review in Markdown",
  "review_md_url": "/downloads/review-<id>.md",
  "review_docx_url": "/downloads/review-<id>.docx",
  "references": [
    {"index": 1, "text": "Formatted citation", "url": "source link"}
  ],
  "latency_ms": 1234,
  "style": "apa",
  "sources": [
    {"source": "Arxiv", "retrieved": 12, "used": 8}
  ]
}
```
Errors are returned as standard FastAPI JSON (and rendered inline for HTMX requests). Rate limited per IP as configured by `SLOWAPI_RATE_LIMIT`.

> **Gemini access tips**
> If your key cannot access the default model, set `MODEL_GEMINI` to a model you can use (for example `gemini-1.5-flash`, `gemini-2.0-flash`, or `gemini-1.0-pro`). The service will automatically try close variants and report which IDs it attempted if generation fails.

> **Semantic Scholar fallback**
> Anonymous access to the Semantic Scholar Graph API is heavily rate limited. When a 429/403 is encountered, the service automatically retries with backoff and, if needed, falls back to the OpenAlex API to supplement non-arXiv sources. Enable `SEMANTIC_SCHOLAR_API_KEY` to unlock higher quotas and skip the fallback path.

> **Formatting styles**
> The UI and API let you select APA, IEEE, or Chicago-style delivery. The choice adjusts Gemini's tone and the exported Markdown/DOCX references so you can match the expectations of specific venues without manual reformatting.

### `GET /`
Renders the HTMX + Tailwind interface for human users.

### `GET /healthz`
Returns `{ "ok": true }` for uptime monitoring.

### `GET /downloads/<filename>`
Serves generated Markdown and DOCX artefacts.

## Continuous Integration & Delivery

- Lint + tests run via [GitHub Actions](https://github.com/your-org/ai-literature-review/actions) (`.github/workflows/ci.yml`).
- Docker image build is validated on every push. Publish to GHCR by adding a push step and secrets.
- Suggested image tag: `ghcr.io/your-org/ai-literature-review:latest` (update once published).

## Testing

```bash
pytest -q
```

Health checks, retrieval dedupe logic, and an end-to-end pipeline smoke test (with mocked LLM + retrieval) are covered.

## Challenges & Lessons

- **Grounded citations**: Forcing placeholders like `[1]` keeps Gemini honest, but we still validate and warn if any reference is missing post-generation.
- **Fallback embeddings**: SentenceTransformer downloads can fail offline; a TF-IDF fallback guarantees functional vector search during tests and CI.
- **HTMX ergonomics**: Server-rendered partials mean minimal browser JS while still offering loading states and inline error handling.

## License

Distributed under the [MIT License](LICENSE).

## Citation

```
@software{literature_review_companion,
  author = {Your Name},
  title = {Literature Review Companion},
  year = {2025},
  url = {https://github.com/your-org/ai-literature-review}
}
```
