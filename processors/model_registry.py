"""
Generic loader/cache for text-classification models pulled from the
Hugging Face Hub at request time.

This lets a caller pick *any* HF model id for sentiment analysis instead of
being limited to the single fine-tuned model. Pipelines are expensive to
instantiate (they download weights on first use), so loaded pipelines are
kept in a small in-memory cache.

Security note: ``trust_remote_code`` is never enabled. Enabling it would let
an arbitrary Hub repository execute Python code inside this process, which
is not acceptable for a model id supplied by an API caller.
"""

from __future__ import annotations

import threading
from typing import Any

# Maximum number of distinct models kept warm in memory at once. Oldest
# (first loaded) is evicted when the cache is full - simple FIFO, adequate
# for a demo/thesis system that isn't serving many concurrent model ids.
_MAX_CACHED_MODELS = 3

_cache: dict[str, Any] = {}
_cache_order: list[str] = []
_load_errors: dict[str, str] = {}
_lock = threading.Lock()


def get_pipeline(model_id: str, task: str = "sentiment-analysis"):
    """
    Return a cached (or newly loaded) Hugging Face pipeline for ``model_id``.

    Raises ``RuntimeError`` if the model cannot be loaded (unknown repo,
    not a classification model, network error, ...) so callers can turn it
    into a clean HTTP error instead of crashing.
    """
    with _lock:
        if model_id in _cache:
            return _cache[model_id]
        if model_id in _load_errors:
            raise RuntimeError(_load_errors[model_id])

    try:
        from transformers import pipeline

        clf = pipeline(
            task,
            model=model_id,
            tokenizer=model_id,
            trust_remote_code=False,
        )
    except Exception as e:  # pragma: no cover - depends on network/model
        with _lock:
            _load_errors[model_id] = str(e)
        raise RuntimeError(f"Could not load model '{model_id}': {e}") from e

    with _lock:
        _cache[model_id] = clf
        _cache_order.append(model_id)
        while len(_cache_order) > _MAX_CACHED_MODELS:
            oldest = _cache_order.pop(0)
            _cache.pop(oldest, None)

    return clf


def cached_models() -> list[str]:
    """Model ids currently kept warm in memory."""
    with _lock:
        return list(_cache_order)
