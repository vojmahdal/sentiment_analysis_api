"""
Sentiment analysis.

Wraps the fine-tuned RoBERTa model published on the Hugging Face Hub
(``vojmahdal/roberta-sentiment-3labels``). This is the only model in the system
that was trained (fine-tuned) by the author - the others use direct inference of
pre-trained models. The training serves as a demonstration of the fine-tuning
process, as described in the methodology.
"""

from __future__ import annotations

from typing import Any

_MODEL_NAME = "vojmahdal/roberta-sentiment-3labels"
_pipeline = None
_load_error: str | None = None


def _get_pipeline():
    """Lazy-load the sentiment pipeline on first call."""
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return _pipeline

    try:
        from transformers import pipeline

        _pipeline = pipeline(
            "sentiment-analysis",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
        )
        print(f"[sentiment] Loaded model {_MODEL_NAME}.")
    except Exception as e:  # pragma: no cover
        _load_error = str(e)
        print(f"[sentiment] Failed to load model: {e}")
    return _pipeline


def analyze_sentiment(text: str) -> dict[str, Any]:
    """
    Return {"label": <positive|neutral|negative>, "score": <confidence>}.
    """
    if not isinstance(text, str) or not text.strip():
        return {"label": None, "score": 0.0}

    clf = _get_pipeline()
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


def is_ready() -> bool:
    return _load_error is None


def model_name() -> str:
    return _MODEL_NAME
