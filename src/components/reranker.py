from __future__ import annotations

import threading
from typing import Any, Callable

from src.core.config import Settings, get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """CPU-friendly app-level cross-encoder reranker."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_factory: Callable[[str, str], Any] | None = None,
    ):
        self.settings = settings or get_settings()
        self._model_factory = model_factory or self._default_model_factory
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @staticmethod
    def _default_model_factory(model_name: str, device: str) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name, device=device)

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info(
                        "Loading app reranker model %s on %s",
                        self.settings.app_reranker_model,
                        self.settings.app_reranker_device,
                    )
                    self._model = self._model_factory(
                        self.settings.app_reranker_model,
                        self.settings.app_reranker_device,
                    )
        return self._model

    def warmup(self) -> None:
        """Load the reranker model before the first query."""
        self._get_model()

    @staticmethod
    def _score_to_float(score: Any) -> float:
        if hasattr(score, "tolist"):
            score = score.tolist()
        if isinstance(score, list):
            if not score:
                return 0.0
            return float(score[-1])
        return float(score)

    @staticmethod
    def _doc_text(doc: Any) -> str:
        props = getattr(doc, "properties", {}) or {}
        return props.get("text", "") or ""

    @staticmethod
    def _attach_score(doc: Any, score: float) -> None:
        try:
            metadata = getattr(doc, "metadata", None)
            if metadata is not None:
                setattr(metadata, "rerank_score", score)
        except Exception:
            # Weaviate return objects may not expose mutable metadata.
            return

    def rerank(self, query: str, docs: list[Any], top_k: int) -> list[Any]:
        if not docs or top_k <= 0:
            return docs[:top_k] if top_k > 0 else docs

        pairs = [(query, self._doc_text(doc)) for doc in docs]
        model = self._get_model()

        with self._predict_lock:
            raw_scores = model.predict(
                pairs,
                batch_size=self.settings.app_reranker_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        scores = [self._score_to_float(score) for score in raw_scores]
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda item: item[1], reverse=True)

        for doc, score in scored_docs:
            self._attach_score(doc, score)

        return [doc for doc, _score in scored_docs[:top_k]]


_RERANKER: Reranker | None = None
_RERANKER_LOCK = threading.Lock()


def get_reranker(settings: Settings | None = None) -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:
                _RERANKER = Reranker(settings=settings)
    return _RERANKER
