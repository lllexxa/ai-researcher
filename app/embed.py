from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import faiss
import numpy as np

if TYPE_CHECKING:
    from .retrieval import Paper


logger = logging.getLogger(__name__)


class EmbeddingBackend:
    def __init__(self, model_name: str):
        normalized = self._normalize_name(model_name)
        self.use_transformer = normalized not in {"", "tfidf", "disabled", "none"}
        self.model_name = normalized if self.use_transformer else ""
        self._model = None
        self._vectorizer = None

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not name:
            return ""
        # Map common aliases to canonical HF model ids
        aliases = {
            "e5-small": "intfloat/e5-small-v2",
            "e5-small-v2": "intfloat/e5-small-v2",
            "sentence-transformers/e5-small-v2": "intfloat/e5-small-v2",
        }
        return aliases.get(name, name)

    def _load_model(self):
        if not self.use_transformer:
            raise RuntimeError("Transformer backend disabled")
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "sentence-transformers is not installed. Install it or switch MODEL_EMBED to 'tfidf'."
                ) from exc

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        try:
            if self.use_transformer:
                model = self._load_model()
                embeddings = model.encode(
                    list(texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                return embeddings.astype(np.float32)
            raise RuntimeError("Transformer backend disabled")
        except Exception as exc:  # pragma: no cover - safety fallback
            logger.warning("Falling back to TF-IDF embeddings: %s", exc)
            from sklearn.feature_extraction.text import TfidfVectorizer

            if self._vectorizer is None:
                self._vectorizer = TfidfVectorizer()
                matrix = self._vectorizer.fit_transform(texts)
            else:
                matrix = self._vectorizer.transform(texts)
            return matrix.toarray().astype(np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    faiss.normalize_L2(matrix)
    return matrix


def embed_corpus(papers: Sequence["Paper"], backend: EmbeddingBackend) -> np.ndarray:
    texts = [f"{paper.title}. {paper.abstract}" for paper in papers]
    embeddings = backend.encode(texts)
    return _normalize(embeddings)


@dataclass
class RetrievalResult:
    paper: "Paper"
    score: float


def query_similar(
    query: str,
    papers: Sequence["Paper"],
    corpus_embeddings: np.ndarray,
    backend: EmbeddingBackend,
    top_k: int,
) -> list[RetrievalResult]:
    if not papers:
        return []
    query_vec = backend.encode([query])
    query_vec = _normalize(query_vec)
    if corpus_embeddings.size == 0 or query_vec.size == 0:
        return [RetrievalResult(paper=papers[0], score=0.0)]
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    scores, indices = index.search(query_vec, min(top_k, len(papers)))
    results: list[RetrievalResult] = []
    for idx, score in zip(indices[0], scores[0]):
        paper = papers[int(idx)]
        results.append(RetrievalResult(paper=paper, score=float(score)))
    return results
