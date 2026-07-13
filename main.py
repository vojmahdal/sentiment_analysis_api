"""
Conversation Data Extraction System - REST API.

Endpoints:
    GET  /                    -> web dashboard
    GET  /health              -> service + model status
    POST /analyze             -> run full pipeline on a single message
    POST /predict             -> sentiment only (backward compatible)
    POST /ingest               -> batch ingest of a CSV/JSON file
    GET  /records             -> recent stored (anonymized) records
    GET  /records/export.xml  -> stored records exported as XML
    GET  /stats               -> aggregate statistics
    GET  /models              -> default + suggested + currently loaded HF sentiment models

The full pipeline extracts named entities, classifies the topic, evaluates
sentiment, pseudonymizes the text and stores the structured result. Since V3,
the sentiment step can use any Hugging Face Hub text-classification model
selected by the caller, instead of only the fine-tuned default model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import ingest
from pipeline import process_message, process_batch
from processors import ner, topics, sentiment, anonymizer, model_registry

app = FastAPI(
    title="Conversation Data Extraction System",
    description=(
        "Extracts named entities, topics and sentiment from chat conversations, "
        "pseudonymizes personal data (GDPR) and stores structured results."
    ),
    version="3.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Curated list of well-known Hugging Face sentiment/text-classification
# models offered as suggestions in the UI. Any other model id can still be
# supplied manually - this list is not a whitelist.
SUGGESTED_SENTIMENT_MODELS = [
    sentiment.DEFAULT_MODEL_NAME,
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "nlptown/bert-base-multilingual-uncased-sentiment",
]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str
    topic_labels: list[str] | None = None
    sentiment_model: str | None = None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Conversation Data Extraction System is running. See /docs."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "ner": {"name": ner.model_name(), "ready": ner.is_ready()},
            "topics": {"name": topics.model_name(), "ready": topics.is_ready()},
            "sentiment": {"name": sentiment.model_name(), "ready": sentiment.is_ready()},
            "anonymizer": {"backend": anonymizer.backend_name()},
        },
    }


@app.get("/models")
def list_models():
    """Default, suggested and currently warm-cached sentiment models."""
    return {
        "default": sentiment.DEFAULT_MODEL_NAME,
        "suggested": SUGGESTED_SENTIMENT_MODELS,
        "cached": model_registry.cached_models(),
    }


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
@app.post("/analyze")
def analyze(payload: TextRequest):
    """Run the full pipeline on a single message and store the result."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = process_message(
            payload.text,
            topic_labels=payload.topic_labels,
            sentiment_model=payload.sentiment_model,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        db.save_record(result, source="single")
    except Exception as e:
        print(f"[main] Failed to store record: {e}")

    # do not return the raw text in a way that encourages storing it client-side;
    # we return both for the immediate UI, but only anonymized is persisted.
    return result


@app.post("/predict")
def predict(payload: TextRequest):
    """
    Backward-compatible sentiment-only endpoint.
    Kept so existing clients of the original API keep working.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        s = sentiment.analyze_sentiment(payload.text, model_id=payload.sentiment_model)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"text": payload.text, "label": s["label"], "score": s["score"]}


# ---------------------------------------------------------------------------
# Ingest (batch)
# ---------------------------------------------------------------------------
@app.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    sentiment_model: str | None = Form(None),
):
    """
    Ingest a CSV or JSON file of conversations, run the full pipeline on each
    message, store the results and return a summary.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        messages = ingest.parse_upload(file.filename or "", raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if not messages:
        raise HTTPException(
            status_code=400,
            detail="No messages found. Expected a 'text'/'message' column or field.",
        )

    # Cap batch size to keep the demo responsive on limited hardware.
    MAX_BATCH = 200
    truncated = len(messages) > MAX_BATCH
    messages = messages[:MAX_BATCH]

    try:
        results = process_batch(messages, sentiment_model=sentiment_model)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    stored = 0
    for r in results:
        try:
            db.save_record(r, source="ingest")
            stored += 1
        except Exception as e:
            print(f"[main] Failed to store ingest record: {e}")

    return {
        "received": len(messages),
        "processed": len(results),
        "stored": stored,
        "truncated": truncated,
        "items": results,
    }


# ---------------------------------------------------------------------------
# Stored data
# ---------------------------------------------------------------------------
@app.get("/records")
def records(limit: int = 100):
    """Return recent stored records (anonymized only)."""
    return db.get_records(limit=limit)


@app.get("/records/export.xml")
def export_records_xml(limit: int | None = None):
    """Export stored (anonymized) records as XML."""
    xml_bytes = db.export_xml(limit=limit)
    return Response(content=xml_bytes, media_type="application/xml")


@app.get("/stats")
def get_stats():
    """Aggregate statistics for the dashboard."""
    return db.stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
