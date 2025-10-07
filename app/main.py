from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .deps import Settings, get_settings
from .embed import EmbeddingBackend
from .gen import GeminiGenerator, GenerationError
from .service import ReviewBundle, build_review


logger = logging.getLogger(__name__)

settings = get_settings()
BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
markdown_renderer = MarkdownIt("commonmark", {"linkify": True, "typographer": True})  # rich markdown support
markdown_renderer.enable("table")
markdown_renderer.enable("strikethrough")


def _link_open_renderer(self, tokens, idx, options, env):
    token = tokens[idx]
    token.attrSet("target", "_blank")
    token.attrSet("rel", "noopener")
    token.attrSet("class", "text-emerald-300 underline decoration-emerald-500/30 hover:decoration-emerald-400")
    return self.renderToken(tokens, idx, options, env)


markdown_renderer.renderer.rules["link_open"] = _link_open_renderer.__get__(markdown_renderer.renderer, type(markdown_renderer.renderer))

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.slowapi_rate_limit])


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Too many requests. Please retry shortly."})


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = httpx.AsyncClient(timeout=30.0)
    embed_backend = EmbeddingBackend(settings.model_embed)
    app.state.http_client = http_client
    app.state.embed_backend = embed_backend
    app.state.generator: GeminiGenerator | None = None
    yield
    await http_client.aclose()


app = FastAPI(title="AI Literature Review", lifespan=lifespan)
app.state.settings = settings
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.mount(
    "/downloads",
    StaticFiles(directory=str(settings.downloads_path)),
    name="downloads",
)


class ReviewRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=20, max_length=1500)
    style: Literal["apa", "ieee", "chicago"] = Field("apa", description="Reference formatting style.")

    @field_validator("topic", "description")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value cannot be blank")
        return stripped

    @field_validator("style", mode="before")
    @classmethod
    def normalize_style(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class ReferenceSchema(BaseModel):
    index: int
    text: str
    url: str
    style: Literal["apa", "ieee", "chicago"]


class SourceBreakdown(BaseModel):
    source: str
    retrieved: int
    used: int


class ReviewResponse(BaseModel):
    review_md: str
    review_md_url: str
    review_docx_url: str
    references: list[ReferenceSchema]
    latency_ms: int
    total_candidates: int
    sources: list[SourceBreakdown]
    style: Literal["apa", "ieee", "chicago"]


async def get_generator(request: Request) -> GeminiGenerator:
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        try:
            generator = GeminiGenerator(
                api_key=settings.gemini_api_key,
                model_name=settings.model_gemini,
            )
        except ValueError as error:
            logger.error("Gemini configuration error: %s", error)
            raise HTTPException(status_code=500, detail=str(error))
        request.app.state.generator = generator
    return generator


async def run_pipeline(request: ReviewRequest, app: FastAPI, generator: GeminiGenerator) -> ReviewBundle:
    try:
        bundle = await build_review(
            request.topic,
            request.description,
            request.style,
            settings=settings,
            http_client=app.state.http_client,
            embed_backend=app.state.embed_backend,
            generator=generator,
        )
    except GenerationError as exc:
        logger.error(
            "Gemini generation failed (models tried: %s): %s",
            getattr(exc, "candidates_tried", []),
            exc.last_error or exc,
        )
        detail = (
            "Gemini API rejected the request. Verify GEMINI_API_KEY permissions or set MODEL_GEMINI "
            f"to a model you can access. Tried: {', '.join(exc.candidates_tried)}"
        )
        raise HTTPException(status_code=502, detail=detail)
    except httpx.HTTPError as exc:
        logger.exception("Upstream API failure: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to fetch sources. Please retry.")
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    return bundle


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Any:
    return templates.TemplateResponse("index.html", {"request": request, "settings": settings})


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}

@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.post("/api/review", response_model=ReviewResponse)
@limiter.limit(settings.slowapi_rate_limit)
async def api_review(payload: ReviewRequest, request: Request, generator: GeminiGenerator = Depends(get_generator)) -> ReviewResponse:
    bundle = await run_pipeline(payload, request.app, generator)
    md_url = f"/downloads/{bundle.markdown_path.name}"
    docx_url = f"/downloads/{bundle.docx_path.name}"
    references = [
        ReferenceSchema(index=ref.index, text=ref.text, url=ref.url, style=ref.style) for ref in bundle.references
    ]
    sources = [
        SourceBreakdown(
            source=source.title().replace("_", " "),
            retrieved=count,
            used=bundle.selected_source_counts.get(source, 0),
        )
        for source, count in sorted(bundle.source_counts.items(), key=lambda item: item[0])
    ]
    return ReviewResponse(
        review_md=bundle.markdown,
        review_md_url=md_url,
        review_docx_url=docx_url,
        references=references,
        latency_ms=bundle.latency_ms,
        total_candidates=bundle.total_candidates,
        sources=sources,
        style=bundle.style,
    )


@app.post("/partials/review", response_class=HTMLResponse)
@limiter.limit(settings.slowapi_rate_limit)
async def partial_review(
    request: Request,
    topic: str = Form(..., min_length=5, max_length=200),
    description: str = Form(..., min_length=20, max_length=1500),
    style: str = Form("apa"),
    generator: GeminiGenerator = Depends(get_generator),
) -> Any:
    review_request = ReviewRequest(topic=topic, description=description, style=style)
    bundle = await run_pipeline(review_request, request.app, generator)
    rendered = markdown_renderer.render(bundle.markdown)
    sources = [
        SourceBreakdown(
            source=source.title().replace("_", " "),
            retrieved=count,
            used=bundle.selected_source_counts.get(source, 0),
        )
        for source, count in sorted(bundle.source_counts.items(), key=lambda item: item[0])
    ]
    return templates.TemplateResponse(
        "partials/result.html",
        {
            "request": request,
            "bundle": bundle,
        "rendered": rendered,
        "download_md": f"/downloads/{bundle.markdown_path.name}",
        "download_docx": f"/downloads/{bundle.docx_path.name}",
        "sources": sources,
    },
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
    if request.headers.get("hx-request") == "true":
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": exc.detail},
            status_code=exc.status_code,
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
