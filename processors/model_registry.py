"""
Generic loader/cache for Hugging Face pipelines pulled from the Hub at
request time.

This lets a caller pick *any* HF model id - for sentiment analysis, named
entity recognition, or zero-shot topic classification - instead of being
limited to the built-in default models. Pipelines are expensive to
instantiate (they download weights on first use), so loaded pipelines are
kept in a small in-memory cache, namespaced by task so the same model id
used for two different tasks never collides.

Security note: ``trust_remote_code`` is never enabled. Enabling it would let
an arbitrary Hub repository execute Python code inside this process, which
is not acceptable for a model id supplied by an API caller.
"""

from __future__ import annotations

import threading
from typing import Any

# Maximum number of distinct (task, model) pipelines kept warm in memory at
# once. Oldest (first loaded) is evicted when the cache is full - simple
# FIFO, adequate for a demo/thesis system that isn't serving many concurrent
# model ids.
_MAX_CACHED_PIPELINES = 6

_cache: dict[str, Any] = {}
_cache_order: list[str] = []
_load_errors: dict[str, str] = {}
_lock = threading.Lock()


def _cache_key(model_id: str, task: str) -> str:
    return f"{task}::{model_id}"


def get_pipeline(model_id: str, task: str = "sentiment-analysis", **pipeline_kwargs: Any):
    """
    Return a cached (or newly loaded) Hugging Face pipeline for
    ``(task, model_id)``. Extra ``pipeline_kwargs`` (e.g.
    ``aggregation_strategy="simple"`` for NER) are forwarded to
    ``transformers.pipeline`` on first load only.

    Raises ``RuntimeError`` if the model cannot be loaded (unknown repo,
    incompatible with the task, network error, ...) so callers can turn it
    into a clean HTTP error instead of crashing.
    """
    key = _cache_key(model_id, task)

    with _lock:
        if key in _cache:
            return _cache[key]
        if key in _load_errors:
            raise RuntimeError(_load_errors[key])

    try:
        from transformers import pipeline

        clf = pipeline(
            task,
            model=model_id,
            tokenizer=model_id,
            trust_remote_code=False,
            **pipeline_kwargs,
        )
    except Exception as e:  # pragma: no cover - depends on network/model
        with _lock:
            _load_errors[key] = str(e)
        raise RuntimeError(f"Could not load model '{model_id}' for task '{task}': {e}") from e

    with _lock:
        _cache[key] = clf
        _cache_order.append(key)
        while len(_cache_order) > _MAX_CACHED_PIPELINES:
            oldest = _cache_order.pop(0)
            _cache.pop(oldest, None)

    return clf


def cached_models(task: str | None = None) -> dict[str, list[str]]:
    """
    Model ids currently kept warm in memory, grouped by task.

    If ``task`` is given, returns just that task's list under the same key
    (still as a dict, for a consistent return type).
    """
    with _lock:
        keys = list(_cache_order)

    grouped: dict[str, list[str]] = {}
    for key in keys:
        key_task, _, model_id = key.partition("::")
        if task is not None and key_task != task:
            continue
        grouped.setdefault(key_task, []).append(model_id)
    return grouped
