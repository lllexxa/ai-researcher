from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .retrieval import Paper


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _short_author(author: str) -> str:
    parts = author.strip().split()
    if not parts:
        return "Unknown"
    family = parts[-1]
    given = parts[:-1]
    if not given:
        return family
    initials = " ".join(f"{name[0].upper()}." for name in given if name)
    return f"{family}, {initials}" if initials else family


@dataclass
class Reference:
    index: int
    text: str
    url: str
    style: str


def extract_indices(text: str) -> List[int]:
    return sorted({int(match.group(1)) for match in _CITATION_PATTERN.finditer(text)})


def _format_authors(paper: Paper, style: str) -> str:
    authors = paper.authors or []
    if not authors:
        return "Unknown"
    style = style.lower()

    def split_name(name: str) -> tuple[str, List[str]]:
        parts = name.strip().split()
        if not parts:
            return "Unknown", []
        return parts[-1], parts[:-1]

    formatted: List[str] = []
    for author in authors[:6]:
        last, given_parts = split_name(author)
        initials = [p[0].upper() + "." for p in given_parts if p]
        if style == "ieee":
            formatted.append(f"{' '.join(initials)} {last}".strip())
        elif style == "chicago":
            formatted.append(f"{last}, {' '.join(given_parts)}".strip())
        else:  # APA
            formatted.append(f"{last}, {' '.join(initials)}".strip(", "))

    if len(authors) > len(formatted):
        formatted.append("et al.")

    if style == "ieee":
        return ", ".join(formatted)
    if style == "chicago":
        if len(formatted) == 1:
            return formatted[0]
        if len(formatted) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"
    # APA format
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def format_reference(paper: Paper, style: str) -> str:
    style = style.lower()
    authors = _format_authors(paper, style)
    year = paper.year or "n.d."
    title = paper.title.rstrip(".")
    source = paper.source.replace("_", " ").title()
    doi = paper.doi
    url = paper.url
    locator = f"https://doi.org/{doi}" if doi and not doi.startswith("http") else (doi or url or "")

    if style == "ieee":
        locator_part = f", {locator}" if locator else ""
        return f"{authors}, \"{title},\" {source}, {year}{locator_part}"
    if style == "chicago":
        locator_part = f". {locator}" if locator else ""
        return f"{authors}. {year}. \"{title}.\" {source}{locator_part}."
    # APA default
    locator_part = f" {locator}" if locator else ""
    return f"{authors} ({year}). {title}. {source}.{locator_part}"


def map_references(text: str, papers: Sequence[Paper], style: str = "apa") -> tuple[str, List[Reference]]:
    indices = extract_indices(text)
    references: List[Reference] = []
    missing: List[str] = []
    for idx in indices:
        try:
            paper = papers[idx - 1]
        except IndexError:
            missing.append(f"[{idx}]")
            continue
        references.append(
            Reference(
                index=idx,
                text=format_reference(paper, style),
                url=paper.url or (f"https://doi.org/{paper.doi}" if paper.doi else ""),
                style=style,
            )
        )
    if missing:
        text += "\n\n" + "\n".join(f"Missing reference: {placeholder}" for placeholder in missing)

    if references:
        text = text.rstrip() + "\n\n## References\n"
        for ref in references:
            link = ref.url
            link_part = f" [{link}]({link})" if link else ""
            text += f"{ref.index}. {ref.text}"
            if link_part:
                text += link_part
            text += "\n"
    return text, references


def ensure_citations(text: str, papers: Iterable[Paper]) -> bool:
    indices = extract_indices(text)
    return bool(indices) and max(indices, default=0) <= len(list(papers))
