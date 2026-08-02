"""
Topic classification (zero-shot).

Uses ``facebook/bart-large-mnli`` via the Hugging Face
``zero-shot-classification`` pipeline by default. The task is framed as
Natural Language Inference: each candidate label becomes a hypothesis
("This text is about X.") and the model scores entailment.

No training data is required and the candidate labels can be changed at
runtime, which suits customer-support conversations where annotated topic
datasets are usually not available. Since V3, callers may instead pick any
other zero-shot-classification model from the Hugging Face Hub at request
time; that model is loaded and cached via ``processors.model_registry``.

Zero-shot classification is the single most expensive step in the pipeline:
each text is scored against every candidate label as a separate NLI pass
through a large model. Since V6, ``classify_topic_batch`` scores many texts
in one pipeline call instead of one call per text, which matters a lot for
large ``/ingest`` batches, especially on CPU-only hosting.
"""

from __future__ import annotations

from typing import Any

from processors import model_registry

DEFAULT_MODEL_NAME = "facebook/bart-large-mnli"
_BATCH_SIZE = 16

# Default candidate topics for a customer-support domain.
DEFAULT_LABELS = [
    "billing and payments",
    "technical issue",
    "complaint",
    "product question",
    "account management",
    "cancellation",
    "delivery and shipping",
    "praise and positive feedback",
]

_pipeline = None
_load_error: str | None = None


def _get_default_pipeline():
    """Lazy-load the default zero-shot pipeline on first call."""
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return _pipeline

    try:
        from transformers import pipeline

        _pipeline = pipeline(
            "zero-shot-classification",
            model=DEFAULT_MODEL_NAME,
        )
        print(f"[topics] Loaded model {DEFAULT_MODEL_NAME}.")
    except Exception as e:  # pragma: no cover
        _load_error = str(e)
        print(f"[topics] Failed to load model: {e}")
    return _pipeline


def classify_topic(
    text: str,
    labels: list[str] | None = None,
    top_k: int = 3,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Classify ``text`` into one of ``labels``.

    ``model_id`` optionally selects a different Hugging Face Hub
    zero-shot-classification model (loaded/cached on demand via
    ``model_registry``) instead of the default. Raises ``RuntimeError`` if
    that model cannot be loaded, so the API layer can turn it into a clean
    400 response.

    Returns:
        {
          "topic": <best label or None>,
          "score": <confidence of best label>,
          "all": [{"label", "score"}, ...]  # top_k candidates
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {"topic": None, "score": 0.0, "all": []}

    candidate_labels = labels or DEFAULT_LABELS

    if model_id and model_id != DEFAULT_MODEL_NAME:
        clf = model_registry.get_pipeline(model_id, task="zero-shot-classification")
    else:
        clf = _get_default_pipeline()

    if clf is None:
        return {"topic": None, "score": 0.0, "all": []}

    try:
        result = clf(text, candidate_labels, multi_label=False)
    except Exception as e:  # pragma: no cover
        print(f"[topics] Inference failed: {e}")
        return {"topic": None, "score": 0.0, "all": []}

    return _format_result(result, top_k)


def classify_topic_batch(
    texts: list[str],
    labels: list[str] | None = None,
    top_k: int = 3,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Batched version of ``classify_topic``: scores the whole list of texts
    against the candidate labels in one pipeline call instead of one call
    per text. Empty/blank texts are skipped and get the empty-result shape
    back, at their original position.
    """
    empty = {"topic": None, "score": 0.0, "all": []}
    results: list[dict[str, Any]] = [dict(empty) for _ in texts]
    valid = [(i, t) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if not valid:
        return results

    candidate_labels = labels or DEFAULT_LABELS

    if model_id and model_id != DEFAULT_MODEL_NAME:
        clf = model_registry.get_pipeline(model_id, task="zero-shot-classification")
    else:
        clf = _get_default_pipeline()
    if clf is None:
        return results

    indices, valid_texts = zip(*valid)
    try:
        raw_batch = clf(list(valid_texts), candidate_labels, multi_label=False, batch_size=_BATCH_SIZE)
    except Exception as e:  # pragma: no cover
        print(f"[topics] Batch inference failed: {e}")
        return results

    # A single-item input list should still come back as a list-of-one, but
    # be defensive in case a given pipeline/version collapses it to a dict.
    if isinstance(raw_batch, dict):
        raw_batch = [raw_batch]

    for idx, raw in zip(indices, raw_batch):
        results[idx] = _format_result(raw, top_k)
    return results


def _format_result(result: dict[str, Any], top_k: int) -> dict[str, Any]:
    pairs = list(zip(result["labels"], result["scores"]))
    top = pairs[:top_k]
    return {
        "topic": pairs[0][0] if pairs else None,
        "score": round(float(pairs[0][1]), 4) if pairs else 0.0,
        "all": [{"label": l, "score": round(float(s), 4)} for l, s in top],
    }


def is_ready() -> bool:
    return _load_error is None


def model_name() -> str:
    return DEFAULT_MODEL_NAME
