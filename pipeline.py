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

``process_batch`` (used by ``/ingest``) does NOT loop over ``process_message``
per item. Since V6 it calls each processor's batched function once per chunk
of messages (NER, topic classification and sentiment analysis all support
batched inference), then reassembles the per-message results. This matters a
lot for large batches: zero-shot topic classification in particular scores
every text against every candidate label, so batching the underlying model
calls instead of looping message-by-message meaningfully reduces total
``/ingest`` time, especially on CPU-only hosting (e.g. the free tier of
Hugging Face Spaces).

Since V7, ``process_batch`` processes messages in fixed-size chunks (rather
than one call covering the whole batch) and reports cumulative progress via
an optional ``on_progress`` callback after each chunk. This is what lets
``/ingest`` run as a background job with a progress bar instead of one long
blocking request - see ``jobs.py`` and the README section on background
ingest jobs. The chunk size is a middle ground: large enough to keep most of
the V6 batching speedup, small enough to give a few progress updates instead
of only 0% and 100%.
"""

from __future__ import annotations

from typing import Any, Callable

from processors import ner, topics, sentiment, anonymizer

# Messages per chunk when processing a batch. Each chunk still runs one
# batched pipeline call per step (see processors/*.py), so this trades off
# progress-reporting granularity against the throughput benefit of larger
# batches - 40 keeps most of that benefit while giving ~5 progress updates
# for a full 200-message /ingest batch.
_CHUNK_SIZE = 40


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


def _process_chunk(
    messages: list[dict[str, Any]],
    topic_labels: list[str] | None,
    sentiment_model: str | None,
    ner_model: str | None,
    topic_model: str | None,
) -> list[dict[str, Any]]:
    """Batched processing of a single chunk of messages (see _CHUNK_SIZE)."""
    texts = [(msg.get("text") or "").strip() for msg in messages]

    entities_batch = ner.extract_entities_batch(texts, model_id=ner_model)
    topic_batch = topics.classify_topic_batch(texts, labels=topic_labels, model_id=topic_model)
    sentiment_batch = sentiment.analyze_sentiment_batch(texts, model_id=sentiment_model)

    results: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        text = texts[i]
        if not text:
            processed = {
                "text": "",
                "anonymized_text": "",
                "entities": [],
                "topic": None,
                "topic_score": 0.0,
                "sentiment": None,
                "sentiment_score": 0.0,
            }
        else:
            topic_result = topic_batch[i]
            sentiment_result = sentiment_batch[i]
            processed = {
                "text": text,
                "anonymized_text": anonymizer.anonymize_text(text),
                "entities": entities_batch[i],
                "topic": topic_result["topic"],
                "topic_score": topic_result["score"],
                "topic_candidates": topic_result["all"],
                "sentiment": sentiment_result["label"],
                "sentiment_score": sentiment_result["score"],
            }

        # carry over ingest metadata
        for key in ("conversation_id", "speaker", "timestamp"):
            if key in msg:
                processed[key] = msg[key]
        results.append(processed)
    return results


def process_batch(
    messages: list[dict[str, Any]],
    topic_labels: list[str] | None = None,
    sentiment_model: str | None = None,
    ner_model: str | None = None,
    topic_model: str | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Process a list of normalized messages (from ingest).

    Each input item is expected to contain at least a ``text`` field, plus
    optional metadata (conversation_id, speaker, timestamp) which is passed
    through to the result. Unlike ``process_message``, this runs NER, topic
    classification and sentiment analysis as batched pipeline calls (one per
    step, per chunk of ``_CHUNK_SIZE`` messages) instead of looping the full
    per-message pipeline ``len(messages)`` times.

    ``on_progress``, if given, is called with the cumulative number of
    messages processed so far after each chunk completes - used by
    ``/ingest`` to report progress on a background job (see ``jobs.py``).
    """
    results: list[dict[str, Any]] = []
    for start in range(0, len(messages), _CHUNK_SIZE):
        chunk = messages[start : start + _CHUNK_SIZE]
        results.extend(
            _process_chunk(chunk, topic_labels, sentiment_model, ner_model, topic_model)
        )
        if on_progress is not None:
            on_progress(len(results))
    return results
