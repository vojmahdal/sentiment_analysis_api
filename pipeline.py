"""
NLP pipeline orchestration.

Applies the full processing chain to a single message:

    raw text
       -> named entity recognition (NER)
       -> topic classification
       -> sentiment analysis
       -> anonymization (pseudonymization) of the text
       -> structured result

This is the core component referenced in the methodology. Each step is a
separate processor module so the components can be developed, replaced or
scaled independently.
"""

from __future__ import annotations

from typing import Any

from processors import ner, topics, sentiment, anonymizer


def process_message(
    text: str,
    topic_labels: list[str] | None = None,
    sentiment_model: str | None = None,
    ner_model: str | None = None,
    topic_model: str | None = None,
) -> dict[str, Any]:
    """
    Run the full pipeline on a single message and return a structured result.

    The returned ``anonymized_text`` is safe to store / display; the original
    ``text`` is returned only for the immediate response and is never persisted
    in readable form (see db.py). ``sentiment_model``, ``ner_model`` and
    ``topic_model`` each optionally select a Hugging Face Hub model id to use
    instead of that step's default model.
    """
    text = (text or "").strip()
    if not text:
        return {
            "text": "",
            "anonymized_text": "",
            "entities": [],
            "topic": None,
            "topic_score": 0.0,
            "sentiment": None,
            "sentiment_score": 0.0,
        }

    entities = ner.extract_entities(text, model_id=ner_model)
    topic_result = topics.classify_topic(text, labels=topic_labels, model_id=topic_model)
    sentiment_result = sentiment.analyze_sentiment(text, model_id=sentiment_model)
    anonymized = anonymizer.anonymize_text(text)

    return {
        "text": text,
        "anonymized_text": anonymized,
        "entities": entities,
        "topic": topic_result["topic"],
        "topic_score": topic_result["score"],
        "topic_candidates": topic_result["all"],
        "sentiment": sentiment_result["label"],
        "sentiment_score": sentiment_result["score"],
    }


def process_batch(
    messages: list[dict[str, Any]],
    topic_labels: list[str] | None = None,
    sentiment_model: str | None = None,
    ner_model: str | None = None,
    topic_model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Process a list of normalized messages (from ingest).

    Each input item is expected to contain at least a ``text`` field, plus
    optional metadata (conversation_id, speaker, timestamp) which is passed
    through to the result.
    """
    results: list[dict[str, Any]] = []
    for msg in messages:
        processed = process_message(
            msg.get("text", ""),
            topic_labels=topic_labels,
            sentiment_model=sentiment_model,
            ner_model=ner_model,
            topic_model=topic_model,
        )
        # carry over ingest metadata
        for key in ("conversation_id", "speaker", "timestamp"):
            if key in msg:
                processed[key] = msg[key]
        results.append(processed)
    return results
