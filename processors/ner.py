"""
Named Entity Recognition (NER).

Uses the pre-trained model ``dslim/bert-base-NER`` (fine-tuned on CoNLL-2003,
~92.6% F1) via the Hugging Face ``token-classification`` pipeline.

No training is required here - this is direct inference on a pre-trained model,
in line with the methodology. The model is loaded lazily on first use so the
application starts quickly and only pays the memory cost when NER is actually
needed.
"""

from __future__ import annotations

from typing import Any

_MODEL_NAME = "dslim/bert-base-NER"
_pipeline = None
_load_error: str | None = None


def _get_pipeline():
    """Lazy-load the NER pipeline on first call."""
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return _pipeline

    try:
        from transformers import pipeline

        _pipeline = pipeline(
            "token-classification",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            aggregation_strategy="simple",  # merge sub-word tokens into whole entities
        )
        print(f"[ner] Loaded model {_MODEL_NAME}.")
    except Exception as e:  # pragma: no cover
        _load_error = str(e)
        print(f"[ner] Failed to load model: {e}")
    return _pipeline


def extract_entities(text: str) -> list[dict[str, Any]]:
    """
    Extract named entities from ``text``.

    Returns a list of dicts: {"text", "type", "start", "end", "score"}.
    Entity types follow CoNLL-2003: PER (person), LOC (location),
    ORG (organization), MISC (miscellaneous).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    nlp = _get_pipeline()
    if nlp is None:
        return []

    try:
        raw = nlp(text)
    except Exception as e:  # pragma: no cover
        print(f"[ner] Inference failed: {e}")
        return []

    entities: list[dict[str, Any]] = []
    for ent in raw:
        entities.append(
            {
                "text": ent.get("word", ""),
                "type": ent.get("entity_group", ent.get("entity", "")),
                "start": int(ent.get("start", 0)),
                "end": int(ent.get("end", 0)),
                "score": round(float(ent.get("score", 0.0)), 4),
            }
        )
    return entities


def is_ready() -> bool:
    """True if the model is loaded or can be loaded (no fatal error)."""
    return _load_error is None


def model_name() -> str:
    return _MODEL_NAME
