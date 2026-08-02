"""
Sentiment analysis.

Uses the fine-tuned RoBERTa model published on the Hugging Face Hub
(``vojmahdal/roberta-sentiment-3labels``) by default - this is the only model
in the system that was trained (fine-tuned) by the author, as described in
the methodology. Since V3, callers may instead pick any other text
classification model from the Hugging Face Hub at request time; that model is
loaded and cached via ``processors.model_registry``. Since V6,
``analyze_sentiment_batch`` scores many texts in one pipeline call instead of
one call per text, which is faster for large ``/ingest`` batches.
"""

from __future__ import annotations

from typing import Any

from processors import model_registry

DEFAULT_MODEL_NAME = "vojmahdal/roberta-sentiment-3labels"
_BATCH_SIZE = 16
_pipeline = None
_load_error: str | None = None


def _get_default_pipeline():
    """Lazy-load the default sentiment pipeline on first call."""
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return _pipeline

    try:
        from transformers import pipeline

        _pipeline = pipeline(
            "sentiment-analysis",
            model=DEFAULT_MODEL_NAME,
            tokenizer=DEFAULT_MODEL_NAME,
        )
        print(f"[sentiment] Loaded model {DEFAULT_MODEL_NAME}.")
    except Exception as e:  # pragma: no cover
        _load_error = str(e)
        print(f"[sentiment] Failed to load model: {e}")
    return _pipeline


def analyze_sentiment(text: str, model_id: str | None = None) -> dict[str, Any]:
    """
    Return {"label": <positive|neutral|negative>, "score": <confidence>}.

    ``model_id`` optionally selects a different Hugging Face Hub model
    (loaded/cached on demand via ``model_registry``) instead of the default
    fine-tuned model. Raises ``RuntimeError`` if that model cannot be loaded,
    so the API layer can turn it into a clean 400 response.
    """
    if not isinstance(text, str) or not text.strip():
        return {"label": None, "score": 0.0}

    if model_id and model_id != DEFAULT_MODEL_NAME:
        clf = model_registry.get_pipeline(model_id, task="sentiment-analysis")
    else:
        clf = _get_default_pipeline()

    if clf is None:
        return {"label": None, "score": 0.0}

    try:
        result = clf(text)[0]
        return {
            "label": result["label"],
            "score": round(float(result["score"]), 4),
        }
    except Exception as e:  # pragma: no cover
        print(f"[sentiment] Inference failed: {e}")
        return {"label": None, "score": 0.0}


def analyze_sentiment_batch(
    texts: list[str], model_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Batched version of ``analyze_sentiment``: runs sentiment analysis once
    over the whole list of texts instead of once per text. Empty/blank texts
    are skipped and get the empty-result shape back, at their original
    position.
    """
    empty = {"label": None, "score": 0.0}
    results: list[dict[str, Any]] = [dict(empty) for _ in texts]
    valid = [(i, t) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if not valid:
        return results

    if model_id and model_id != DEFAULT_MODEL_NAME:
        clf = model_registry.get_pipeline(model_id, task="sentiment-analysis")
    else:
        clf = _get_default_pipeline()
    if clf is None:
        return results

    indices, valid_texts = zip(*valid)
    try:
        raw_batch = clf(list(valid_texts), batch_size=_BATCH_SIZE)
    except Exception as e:  # pragma: no cover
        print(f"[sentiment] Batch inference failed: {e}")
        return results

    # A single-item input list should still come back as a list-of-one, but
    # be defensive in case a given pipeline/version collapses it to a dict.
    if isinstance(raw_batch, dict):
        raw_batch = [raw_batch]

    for idx, raw in zip(indices, raw_batch):
        results[idx] = {"label": raw["label"], "score": round(float(raw["score"]), 4)}
    return results


def is_ready() -> bool:
    return _load_error is None


def model_name() -> str:
    return DEFAULT_MODEL_NAME
