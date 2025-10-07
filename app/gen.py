from __future__ import annotations

import asyncio
import logging
import textwrap
from dataclasses import dataclass
from typing import List, Sequence

import google.generativeai as genai
from google.api_core import exceptions as gexc

from .retrieval import Paper


PROMPT_TEMPLATE = """You are an expert research assistant. Using only the supplied sources, craft a polished literature review in Markdown with the following structure:

# Introduction
# Thematic synthesis
# Comparison of methods
# Research gaps
# Conclusion

Guidelines:
- Cite using bracketed numbers that map to the provided sources, e.g. [1], [2].
- Only cite sources that are explicitly listed. No hallucinated references.
- Paraphrase; do not copy abstracts verbatim.
- Highlight consensus and disagreements across the literature.
- Use signposting language between sections to ensure a cohesive narrative.
- Emphasise limitations, datasets, and evaluation metrics when relevant.
- Finish with a bullet list of key references with the same numbering.
- Style / tone requirements: {style_instruction}

Topic: {topic}
Research focus: {description}

Sources:
{sources}
"""


logger = logging.getLogger(__name__)

STYLE_INSTRUCTIONS = {
    "apa": (
        "Write in an academic tone with clear topic sentences. Use past tense when summarising prior work and "
        "present tense for interpretations. Ensure each reference includes author surnames and publication year in "
        "the prose. Maintain objective, formal language suitable for APA-style manuscripts."
    ),
    "ieee": (
        "Adopt a concise, technical tone typical of IEEE survey articles. Prefer active voice and short sentences. "
        "Highlight quantitative results, comparative metrics, and implementation details. Finish each paragraph by "
        "clarifying the practical implication for engineers or system designers."
    ),
    "chicago": (
        "Write in a narrative, analytical style suitable for Chicago humanities publications. Integrate historical "
        "context, contrasting viewpoints, and methodological critiques. Use descriptive transitions and emphasize "
        "broader societal implications where relevant."
    ),
}


def _format_source(paper: Paper, idx: int) -> str:
    authors = ", ".join(paper.authors) if paper.authors else "Unknown"
    year = paper.year or "n.d."
    abstract = paper.abstract.replace("\n", " ").strip()
    abstract = abstract[:900] + "..." if len(abstract) > 900 else abstract
    return textwrap.dedent(
        f"""[{idx}] Title: {paper.title}
Authors: {authors}
Year: {year}
DOI/ID: {paper.doi or paper.identifier}
URL: {paper.url}
Abstract: {abstract}
"""
    )


def _normalize_model_name(name: str) -> str:
    # Accept both 'models/<id>' and '<id>' formats
    return name.split("/", 1)[-1] if name.startswith("models/") else name


def _dedupe_preserve_order(candidates: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in candidates:
        if not item:
            continue
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _expand_candidates(primary: str) -> List[str]:
    base = _normalize_model_name(primary)

    baseline = [primary, base, f"models/{base}"]

    if base.endswith("-8b"):
        baseline.append(base.replace("-8b", ""))
    else:
        baseline.append(f"{base}-8b")

    if not base.endswith("-latest"):
        baseline.append(f"{base}-latest")

    if not base.endswith("-001"):
        baseline.append(f"{base}-001")

    extras = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro",
        "gemini-pro",
        "gemini-1.0-pro",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-8b",
        "models/gemini-1.5-pro",
        "models/gemini-1.0-pro",
    ]

    return _dedupe_preserve_order([*baseline, *extras])


@dataclass
class GenerationError(RuntimeError):
    message: str
    candidates_tried: List[str]
    last_error: Exception | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)


class GeminiGenerator:
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for generation")
        genai.configure(api_key=api_key)
        primary = _normalize_model_name(model_name)
        self._candidates = _expand_candidates(primary)
        self._model = None

    def _ensure_model(self, name: str):
        self._model = genai.GenerativeModel(name)
        return self._model

    async def _try_generate(self, model_name: str, prompt: str):
        model = self._ensure_model(model_name)
        logger.info("Attempting Gemini generation with model '%s'", model_name)
        return await asyncio.to_thread(
            model.generate_content,
            prompt,
            request_options={"timeout": 120},
        )

    async def generate_review(
        self,
        topic: str,
        description: str,
        papers: Sequence[Paper],
        *,
        style: str = "apa",
    ) -> str:
        if not papers:
            raise ValueError("At least one paper is required to generate a review")
        sources_block = "\n".join(_format_source(paper, idx + 1) for idx, paper in enumerate(papers))
        style_instruction = STYLE_INSTRUCTIONS.get(style.lower(), STYLE_INSTRUCTIONS["apa"])
        prompt = PROMPT_TEMPLATE.format(
            topic=topic,
            description=description,
            sources=sources_block,
            style_instruction=style_instruction,
        )

        last_exc: Exception | None = None
        for candidate in self._candidates:
            try:
                response = await self._try_generate(candidate, prompt)
                if response and getattr(response, "text", None):
                    logger.info("Gemini generation succeeded with model '%s'", candidate)
                    return response.text.strip()
            except (gexc.NotFound, gexc.PermissionDenied, gexc.FailedPrecondition) as exc:
                logger.warning("Gemini model '%s' rejected request: %s", candidate, exc)
                last_exc = exc
                continue
            except Exception as exc:  # network, transient, etc.
                logger.warning("Gemini request failed for model '%s': %s", candidate, exc)
                last_exc = exc
                continue

        raise GenerationError(
            message=(
                "Gemini generation failed after trying multiple models. "
                "Check GEMINI_API_KEY permissions and update MODEL_GEMINI if needed."
            ),
            candidates_tried=self._candidates,
            last_error=last_exc,
        )
