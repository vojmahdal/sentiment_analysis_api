"""
Conversation Data Extraction System - REST API.

Endpoints:
    GET  /                    -> web dashboard
    GET  /health              -> service + model status
    POST /analyze             -> run full pipeline on a single message
    POST /predict             -> sentiment only (backward compatible)
    POST /ingest               -> start a batch ingest job, returns a job id
    GET  /ingest/status/{id}  -> progress / result of a background ingest job
    GET  /records             -> recent stored (anonymized) records, optional ?engine=local|llm
    GET  /records/export      -> stored records exported as XML, JSON or CSV, optional ?engine=
    GET  /stats               -> aggregate statistics
    GET  /models              -> default + suggested + currently loaded HF models per task
    GET  /engines             -> local pipeline + configured LLM providers (V8)

The full pipeline extracts named entities, classifies the topic, evaluates
sentiment, pseudonymizes the text and stores the structured result. Since V3,
each of the three ML steps (NER, topic classification, sentiment) can use any
compatible Hugging Face Hub model selected by the caller, instead of only the
built-in default model for that step. Since V4, stored records can be
exported in a choice of formats (XML, JSON, CSV) via `/records/export`. Since
V7, `/ingest` runs as a background job (see `jobs.py`) instead of one long
blocking request, so the dashboard can show a progress bar and elapsed timer.

Since V8, `/analyze` and `/ingest` accept an `engine` field: `"local"` (the
default, unchanged BERT pipeline above) or the name of a configured LLM
provider (`"anthropic"`, `"google"`, `"openai"`) - see `llm_client/`. In LLM
mode, the same local anonymization runs first (pre_pseudonymized mode), and
the pseudonymized text is what actually leaves the system; a single LLM call
then determines sentiment, entities and topic together. LLM results are
mapped onto the same shape as the local pipeline's and stored in the same
`records` table (see `db.save_record`), so they show up in `/records` and
the dashboard next to local-model results, tagged with `engine`/`provider`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Must run before importing llm_client: ProviderConfig/PROVIDERS read the
# LLM API key env vars at import time, so .env has to be loaded first.
load_dotenv()

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import ingest
import jobs
from pipeline import process_message, process_batch
from processors import ner, topics, sentiment, anonymizer, model_registry

from llm_client.client import LLMError
from llm_client.config import PROVIDERS as LLM_PROVIDERS
from llm_client.pipeline import LLMPipeline
from llm_client.schemas import AnalysisResult

app = FastAPI(
    title="Conversation Data Extraction System",
    description=(
        "Extracts named entities, topics and sentiment from chat conversations, "
        "pseudonymizes personal data (GDPR) and stores structured results."
    ),
    version="8.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Curated lists of well-known Hugging Face models offered as suggestions in
# the UI, one per pipeline step. Any other compatible model id can still be
# supplied manually - these lists are not a whitelist.
SUGGESTED_MODELS = {
    "sentiment": [
        sentiment.DEFAULT_MODEL_NAME,
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "nlptown/bert-base-multilingual-uncased-sentiment",
    ],
    "ner": [
        ner.DEFAULT_MODEL_NAME,
        "dslim/bert-large-NER",
        "Jean-Baptiste/roberta-large-ner-english",
        "dbmdz/bert-large-cased-finetuned-conll03-english",
    ],
    "topics": [
        topics.DEFAULT_MODEL_NAME,
        "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        "valhalla/distilbart-mnli-12-3",
    ],
}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str
    topic_labels: list[str] | None = None
    sentiment_model: str | None = None
    ner_model: str | None = None
    topic_model: str | None = None
    # V8: "local" (default) or a configured LLM provider name (see /engines).
    engine: str = "local"
    llm_mode: str | None = None


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
    """Default, suggested and currently warm-cached models, per pipeline step."""
    cached = model_registry.cached_models()
    return {
        "sentiment": {
            "default": sentiment.DEFAULT_MODEL_NAME,
            "suggested": SUGGESTED_MODELS["sentiment"],
            "cached": cached.get("sentiment-analysis", []),
        },
        "ner": {
            "default": ner.DEFAULT_MODEL_NAME,
            "suggested": SUGGESTED_MODELS["ner"],
            "cached": cached.get("token-classification", []),
        },
        "topics": {
            "default": topics.DEFAULT_MODEL_NAME,
            "suggested": SUGGESTED_MODELS["topics"],
            "cached": cached.get("zero-shot-classification", []),
        },
    }


@app.get("/engines")
def list_engines():
    """
    Available analysis engines: the local BERT pipeline (always available)
    plus any LLM provider that has an API key configured (see
    `llm_client/config.py` / `.env.example`). The dashboard uses this to
    build the Engine selector - providers without a key are reported but
    marked `available: false` so the UI can grey them out.
    """
    engines = [{"id": "local", "label": "Local models (BERT)", "available": True}]
    for name, cfg in LLM_PROVIDERS.items():
        engines.append(
            {
                "id": name,
                "label": f"LLM: {name}",
                "model": cfg.model,
                "available": cfg.available,
            }
        )
    return {"engines": engines}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def _llm_result_to_response(original_text: str, result: AnalysisResult) -> dict[str, Any]:
    """
    Map an `llm_client` `AnalysisResult` onto the same response shape
    `/analyze` already returns for the local engine, so it can be stored via
    the same `db.save_record` and listed/exported alongside local-model
    records. Adds engine/provider/model/latency/cost on top, which the
    local-engine response doesn't have (those columns stay `NULL` for it).
    """
    top = max(result.topics, key=lambda t: t.score, default=None)
    entities = [
        {
            "text": e.text,
            "type": e.label,
            "start": e.start,
            "end": e.end,
            "score": e.score,
        }
        for e in result.entities
    ]
    return {
        "text": original_text,
        "anonymized_text": result.pseudonymized_text,
        "entities": entities,
        "topic": top.label if top else None,
        "topic_score": top.score if top else 0.0,
        "topic_candidates": [t.model_dump() for t in result.topics],
        "sentiment": result.sentiment.label,
        "sentiment_score": result.sentiment.score,
        "engine": result.engine,
        "provider": result.provider,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
    }


def _get_llm_pipeline(engine: str, llm_mode: str | None) -> LLMPipeline:
    try:
        return LLMPipeline(provider=engine, mode=llm_mode or "pre_pseudonymized")
    except (LLMError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze")
def analyze(payload: TextRequest):
    """Run the full pipeline on a single message and store the result."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if payload.engine == "local":
        try:
            result = process_message(
                payload.text,
                topic_labels=payload.topic_labels,
                sentiment_model=payload.sentiment_model,
                ner_model=payload.ner_model,
                topic_model=payload.topic_model,
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

    # LLM engine: local anonymization happens inside LLMPipeline.analyze()
    # (same processors.anonymizer as the local engine) before the
    # pseudonymized text is sent to the chosen provider in a single call.
    llm_pipeline = _get_llm_pipeline(payload.engine, payload.llm_mode)
    result = llm_pipeline.analyze(payload.text)
    if result.error:
        raise HTTPException(status_code=502, detail=result.error)

    response = _llm_result_to_response(payload.text, result)

    try:
        db.save_record(response, source="single")
    except Exception as e:
        print(f"[main] Failed to store record: {e}")

    return response


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
@app.post("/ingest", status_code=202)
async def ingest_file(
    file: UploadFile = File(...),
    sentiment_model: str | None = Form(None),
    ner_model: str | None = Form(None),
    topic_model: str | None = Form(None),
    engine: str = Form("local"),
    llm_mode: str | None = Form(None),
):
    """
    Start a background job that ingests a CSV or JSON file of conversations
    and runs the full pipeline on each message. Returns immediately with a
    job id; poll ``GET /ingest/status/{job_id}`` for progress and, once
    finished, the same summary this endpoint used to return directly
    (``received``, ``processed``, ``stored``, ``truncated``, ``items``).

    Parsing/validating the file happens synchronously here (fast, no model
    calls); only the actual NLP processing runs in the background.

    Since V8, ``engine`` selects "local" (default, unchanged) or a
    configured LLM provider name. LLM batches are processed with bounded
    concurrency via `LLMPipeline.analyze_batch` (local anonymization first,
    as always) and stored in the same `records` table as local-model
    results; progress is still reported through the same job mechanism.
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

    job_id = jobs.create_job(total=len(messages))

    def run_job() -> None:
        if engine != "local":
            _run_llm_ingest_job(job_id, messages, engine, llm_mode, truncated)
            return

        try:
            results = process_batch(
                messages,
                sentiment_model=sentiment_model,
                ner_model=ner_model,
                topic_model=topic_model,
                on_progress=jobs.progress_callback(job_id),
            )
        except RuntimeError as e:
            jobs.fail_job(job_id, str(e))
            return
        except Exception as e:  # pragma: no cover - safety net so a job never hangs
            jobs.fail_job(job_id, f"Unexpected error: {e}")
            return

        stored = 0
        for r in results:
            try:
                db.save_record(r, source="ingest")
                stored += 1
            except Exception as e:
                print(f"[main] Failed to store ingest record: {e}")

        jobs.finish_job(job_id, {
            "engine": "local",
            "received": len(messages),
            "processed": len(results),
            "stored": stored,
            "truncated": truncated,
            "items": results,
        })

    # Runs the (blocking, CPU-bound or network-bound) work in a plain thread
    # rather than an asyncio task, so it doesn't block the event loop and
    # status polls keep being served while it runs.
    threading.Thread(target=run_job, daemon=True).start()

    return {"job_id": job_id, "total": len(messages)}


def _run_llm_ingest_job(
    job_id: str,
    messages: list[dict[str, Any]],
    engine: str,
    llm_mode: str | None,
    truncated: bool,
) -> None:
    """
    LLM-engine counterpart of the local `/ingest` job body. Runs
    `LLMPipeline.analyze_batch` (bounded concurrency, local anonymization
    first as always) and stores each result via the same `db.save_record`
    used by the local engine, so LLM-analyzed messages show up in `/records`
    next to local-model ones instead of in a separate database.
    """
    try:
        llm_pipeline = LLMPipeline(provider=engine, mode=llm_mode or "pre_pseudonymized")
    except (LLMError, ValueError) as e:
        jobs.fail_job(job_id, str(e))
        return

    texts = [(m.get("text") or "").strip() for m in messages if (m.get("text") or "").strip()]
    update_progress = jobs.progress_callback(job_id)

    try:
        results = llm_pipeline.analyze_batch(
            texts, on_progress=lambda done, _total: update_progress(done)
        )
    except Exception as e:  # pragma: no cover - safety net so a job never hangs
        jobs.fail_job(job_id, f"Unexpected error: {e}")
        return

    responses = [_llm_result_to_response(text, r) for text, r in zip(texts, results)]
    failed = sum(1 for r in results if r.error)

    stored = 0
    for response in responses:
        try:
            db.save_record(response, source="ingest")
            stored += 1
        except Exception as e:
            print(f"[main] Failed to store LLM ingest record: {e}")

    jobs.finish_job(job_id, {
        "engine": engine,
        "received": len(messages),
        "processed": len(responses) - failed,
        "failed": failed,
        "stored": stored,
        "truncated": truncated,
        "items": responses,
    })


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    """
    Progress of a background ingest job started via ``POST /ingest``.

    ``status`` is one of ``running``, ``done`` or ``error``. Once ``done``,
    ``result`` holds the same summary the old synchronous ``/ingest``
    returned directly.
    """
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown job id (it may have finished long ago, or the server restarted).",
        )

    finished_at = job["finished_at"] or time.time()
    response: dict[str, Any] = {
        "status": job["status"],
        "processed": job["processed"],
        "total": job["total"],
        "elapsed_seconds": round(finished_at - job["started_at"], 1),
    }
    if job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response


# ---------------------------------------------------------------------------
# Stored data
# ---------------------------------------------------------------------------
def _validate_engine_filter(engine: str | None) -> str | None:
    """
    Shared validation for the ``engine`` filter on ``/records`` and
    ``/records/export``: ``"local"`` (BERT pipeline), ``"llm"`` (any
    provider), or omitted/``None`` for both.
    """
    if engine is not None and engine not in ("local", "llm"):
        raise HTTPException(
            status_code=400,
            detail="Invalid 'engine' filter. Use 'local', 'llm', or omit it for both.",
        )
    return engine


@app.get("/records")
def records(
    limit: int = 100,
    engine: str | None = None,
    topic: str | None = None,
    sentiment: str | None = None,
):
    """
    Return recent stored records (anonymized only).

    ``engine`` optionally filters to ``"local"`` (BERT pipeline) or
    ``"llm"`` (any provider). ``topic``/``sentiment`` optionally filter to
    an exact label (see ``GET /stats`` for the labels actually in use - a
    custom sentiment/topic model can in principle produce others, so these
    aren't restricted to a fixed set). Any combination may be used together;
    omitted filters simply don't narrow the result.
    """
    engine = _validate_engine_filter(engine)
    return db.get_records(limit=limit, engine=engine, topic=topic, sentiment=sentiment)


@app.get("/records/export")
def export_records(
    format: str = "xml",
    limit: int | None = None,
    engine: str | None = None,
    topic: str | None = None,
    sentiment: str | None = None,
):
    """
    Export stored (anonymized) records. ``format`` is one of the keys in
    ``db.EXPORT_FORMATS`` (currently ``xml``, ``json``, ``csv``).
    ``engine``/``topic``/``sentiment`` optionally filter, same as
    ``/records``.
    """
    engine = _validate_engine_filter(engine)
    try:
        content, media_type = db.export_records(
            format, limit=limit, engine=engine, topic=topic, sentiment=sentiment
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = f"records.{format.lower()}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/stats")
def get_stats():
    """Aggregate statistics for the dashboard, plus the supported export formats."""
    return {**db.stats(), "export_formats": sorted(db.EXPORT_FORMATS)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
