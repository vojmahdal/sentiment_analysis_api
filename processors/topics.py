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
"""

from __future__ import annotations

from typing import Any

from processors import model_registry

DEFAULT_MODEL_NAME = "facebook/bart-large-mnli"

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
