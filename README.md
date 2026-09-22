---
title: Conversation Data Extraction System
sdk: docker
emoji: 🗂️
colorFrom: indigo
colorTo: purple
app_port: 8000
---

# Conversation Data Extraction System

A system that extracts structured information from chat conversations:
**named entities (NER), topics and sentiment**, then **pseudonymizes** personal
data (GDPR) and stores the structured result.

## Project versions

This repository is developed incrementally as part of a diploma thesis. Each
version is tagged so the progression is visible in the git history.

| Version | Tag | Description |
|---------|-----|--------------|
| V1 | `v1.0` | Single fine-tuned RoBERTa model, `/predict` endpoint, simple SQLite logging with regex anonymization. |
| V2 | `v2.0` | Full extraction pipeline: batch `/ingest`, NER, zero-shot topic classification, Presidio-based anonymization, dashboard with stored records and statistics. |
| V3 | `v3.0` | Dynamic model selection from the Hugging Face Hub (sentiment, NER and topic classification) at request time, plus XML export of stored records. |
| V4 | `v4.0` | Export of stored records in a choice of formats (XML, JSON, CSV) via a single endpoint, picked from a dropdown button on the dashboard. |
| V5 | `v5.0` | Per-step model choice (sentiment, NER, topics) switched from radio buttons to a `<select>` dropdown; a free-text field for a custom model appears only when "Custom model" is selected. |
| V6 | `v6.0` | Batched inference for `/ingest`: NER, topic classification and sentiment analysis each run once over the whole batch instead of once per message, significantly reducing total processing time for large batches (especially on CPU-only hosting). |
| V7 | `v7.0` | `/ingest` runs as a background job instead of one long blocking request; the dashboard polls job status and shows a progress bar and elapsed-time timer. |
| V8 | `v8.0` | Optional LLM engine (Claude / Gemini / OpenAI) alongside the local BERT pipeline, selectable per request for both `/analyze` and `/ingest`; local pseudonymization always runs before any text reaches an LLM provider; results from both engines are stored and shown together in the same records list. |

See [`docs/class_diagram.md`](docs/class_diagram.md) for the current architecture.

## Pipeline

```
ingest → NER → topic classification → sentiment → anonymization → storage
```

| Step | Method / model |
|------|----------------|
| NER | `dslim/bert-base-NER` (inference) |
| Topics | `facebook/bart-large-mnli` (zero-shot) |
| Sentiment | fine-tuned RoBERTa (`vojmahdal/roberta-sentiment-3labels`) |
| Anonymization | Microsoft Presidio (spaCy NER + regex), regex fallback |
| Storage | SQLite (hash of original + anonymized text) |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web dashboard |
| GET | `/health` | Service + model status |
| GET | `/models` | Default, suggested and currently cached models, per pipeline step |
| GET | `/engines` | Local pipeline + configured LLM providers, with availability (V8) |
| POST | `/analyze` | Full pipeline on a single message (`engine`: `local` or a provider id, V8) |
| POST | `/predict` | Sentiment only (backward compatible, local models only) |
| POST | `/ingest` | Start a background batch-ingest job for a CSV/JSON file, returns a `job_id` (`engine` field, V8) |
| GET | `/ingest/status/{job_id}` | Progress (and, once done, the result) of a background ingest job |
| GET | `/records` | Recent stored (anonymized) records, local and LLM engine alike; optional `?engine=local\|llm`, `?topic=`, `?sentiment=`, `?provider=` filters (V8) |
| GET | `/records/export` | Stored records exported as XML, JSON or CSV (`?format=`), same optional filters (V8) |
| GET | `/stats` | Aggregate statistics (includes the list of supported export formats) |

### Example

```bash
curl -X POST https://<space-url>/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi, John Smith here, my order never arrived. Email john@example.com"}'
```

```bash
curl -X POST https://<space-url>/ingest -F "file=@sample_chats.csv"
# -> {"job_id": "...", "total": 8}
curl https://<space-url>/ingest/status/<job_id>
# -> {"status": "running", "processed": 4, "total": 8, "elapsed_seconds": 3.2}
# poll again once "status" is "done" (or "error") for the final result
```

```bash
curl "https://<space-url>/records/export?format=xml" -o records.xml
curl "https://<space-url>/records/export?format=json" -o records.json
curl "https://<space-url>/records/export?format=csv" -o records.csv
```

## Exporting stored records

Since V4, `GET /records/export` accepts a `format` query parameter (`xml`,
`json` or `csv`, default `xml`) and an optional `limit`. All three formats
share the same underlying data (`db._fetch_export_rows`); adding a new
format only requires one small function in `db.py` plus an entry in
`db.EXPORT_FORMATS` - `main.py` and the dashboard pick it up automatically.
CSV flattens the nested entity list into a single `"TYPE:text; ..."` cell
per record and is written with a UTF-8 BOM so it opens correctly in Excel.

On the dashboard (`/static/records.html`), an **Export ▾** dropdown button
lists the formats returned by `GET /stats` (`export_formats`); picking one
downloads the file via `Content-Disposition: attachment`.

## Choosing models from the Hugging Face Hub

Since V3, each of the three ML steps can use a different Hugging Face Hub
model, selected per request:

| Field | Task | Default |
|-------|------|---------|
| `sentiment_model` | `sentiment-analysis` | `vojmahdal/roberta-sentiment-3labels` |
| `ner_model` | `token-classification` | `dslim/bert-base-NER` |
| `topic_model` | `zero-shot-classification` | `facebook/bart-large-mnli` |

All three are accepted by `/analyze` (JSON body) and `/ingest` (form
fields); `/predict` only accepts `sentiment_model` since it is
sentiment-only. If a field is omitted, that step's default model is used.
Requested models are downloaded and cached in memory on first use
(`processors/model_registry.py`), namespaced by task with a small FIFO
cache (6 pipelines) to bound memory usage. `GET /models` lists, per step,
the default model, a few suggested models, and which ones are currently
cached.

The web dashboard exposes this as a **dropdown (`<select>`)** for each step,
listing the suggested models plus a "Custom model" option. The free-text
field for typing any other Hugging Face repo id is hidden by default and
only appears once "Custom model" is selected in the dropdown.

**Security note:** loaded pipelines never use `trust_remote_code=True`, so an
arbitrary/untrusted model id supplied by a caller cannot execute custom
Python code inside the server process - it is limited to standard
`transformers` inference for the given task. An invalid or incompatible
model id results in a clean HTTP 400 response instead of crashing the
server.

## Batched inference for large `/ingest` batches

Since V6, `pipeline.process_batch` (used only by `/ingest`) no longer loops
over `process_message` once per message. Instead it calls each processor's
batched function - `ner.extract_entities_batch`, `topics.classify_topic_batch`,
`sentiment.analyze_sentiment_batch` - once for the whole list of texts, and
reassembles the per-message results afterwards. Each batched function passes
`batch_size=16` to the underlying `transformers` pipeline so the model
itself processes several texts per forward pass instead of one at a time.

This matters most for topic classification: zero-shot classification scores
every text against every candidate label as a separate NLI pass through
`facebook/bart-large-mnli` (~400M parameters), so for the default 8 labels
that's 8 passes per message. Batching those calls together is the difference
between a 200-message `/ingest` request taking minutes rather than tens of
minutes on CPU-only hosting (e.g. the free tier of Hugging Face Spaces).
`process_message` (used by `/analyze` and `/predict`, always a single
message) is unchanged.

## Background ingest jobs (progress bar + timer)

Since V7, `POST /ingest` no longer blocks until the whole batch is
processed. It parses/validates the uploaded file synchronously (fast, no
model calls), registers a job via `jobs.py` and starts the actual NLP
processing in a background thread, returning `{"job_id", "total"}`
immediately (HTTP 202).

`pipeline.process_batch` processes messages in fixed-size chunks (40 by
default, see `pipeline._CHUNK_SIZE`) instead of one call covering the whole
batch, and reports cumulative progress via an `on_progress` callback after
each chunk - this is what gives `GET /ingest/status/{job_id}` something to
report before the whole job is done. The chunk size is a deliberate
trade-off: large enough to keep most of V6's batching speedup, small enough
to give a handful of progress updates instead of only 0% and 100%.

Job state (`status`, `processed`, `total`, timestamps, and the final result
or error) lives in an in-memory dict in `jobs.py` - intentionally simple,
consistent with the rest of this prototype. A server restart loses in-flight
job status; the dashboard treats an unknown `job_id` (HTTP 404) as an error
rather than hanging forever.

The dashboard polls `/ingest/status/{job_id}` once per second and updates a
`<progress>` bar (`processed` / `total`) plus a locally-ticking elapsed-time
timer (updated every 100ms from the browser's own clock, so it stays smooth
between polls). When the job reaches `done` or `error`, polling stops and
the final summary/error is shown, same as the old synchronous response.

## Optional LLM engine (Claude / Gemini / OpenAI)

Since V8, each of the three tasks - sentiment, NER, topic classification -
can also be resolved by a single call to a large language model instead of
the local BERT pipeline. This is a second, independent "engine", not a
replacement: both are always available side by side, so results can be
compared.

| Engine | How | Package |
|--------|-----|---------|
| `local` (default `engine`) | The existing `pipeline.process_message`/`process_batch` (three separate models) | `processors/` |
| `google` (default LLM provider) | One Gemini call, structured output via `response_schema` | `llm_client/` |
| `anthropic` | One Claude call, structured output via tool use - optional, if the user has their own key | `llm_client/` |
| `openai` | One GPT call, structured output via JSON schema - optional, if the user has their own key | `llm_client/` |

`local` is the request-level default when `engine` is omitted (unchanged
behavior); among the LLM providers, `google` (Gemini) is the
`LLM_PROVIDER` default (`llm_client/config.py`) since it has a usable free
tier - Claude and OpenAI are additional choices for a user who has their
own API key for those.

`GET /engines` reports which providers are actually usable (an API key is
configured); the dashboard's **Engine** dropdown disables the rest instead
of hiding them, so it's clear what a full deployment would offer.

**Anonymization always happens locally, first.** Regardless of which engine
is chosen, the same `processors.anonymizer` pseudonymizes the message before
anything leaves the server; for an LLM engine, only the pseudonymized text
is sent to the provider's API (`llm_client.pipeline.LLMPipeline`, mode
`pre_pseudonymized`). The provider never sees the raw message. A raw-text
mode exists in `llm_client` for controlled experiments but requires an
explicit `LLM_ALLOW_RAW=1` opt-in and is not exposed through the API.

`POST /analyze` and `POST /ingest` both accept an `engine` field
(`"local"` by default) and, for LLM engines, an optional `llm_mode`. The
local-engine response shape is unchanged; the LLM-engine response adds
`engine`, `provider`, `model`, `latency_ms` and `cost_usd` on top of the
same `sentiment`/`topic`/`entities`/`anonymized_text` fields. `/ingest`
with an LLM engine uses the same background-job/progress mechanism from V7
(`GET /ingest/status/{job_id}`); the batch is processed with bounded
concurrency (`LLMPipeline.analyze_batch`, `LLM_CONCURRENCY` workers).

**Both engines store into the same place.** LLM results are mapped onto the
same shape the local pipeline already returns and saved via the same
`db.save_record` into the same `records` table (`db.py`), tagged with
`engine`/`provider`/`model`/`latency_ms`/`cost_usd` (`NULL` for local-engine
records). They therefore show up together in `GET /records`, the
`/records.html` dashboard (with an added **Engine** column and filter
dropdowns for **Engine**, **Topic**, **Sentiment** and **Provider**) and
every export format - `/records` and `/records/export` both accept
optional `?engine=local|llm`, `?topic=`, `?sentiment=` and `?provider=`
query params, usable together, to narrow the list down (`?provider=`
distinguishes individual LLM providers - `anthropic`/`openai`/`google` -
which `?engine=llm` otherwise lumps together, useful when comparing them
side by side) - instead of living in a separate
database only reachable through a dedicated comparison view. A systematic
BERT-vs-LLM comparison
(agreement, entity-level P/R/F1, PII leak rate, latency/cost) was
originally planned as an automated dashboard here, but that comparison is
instead run manually and written up directly in the accompanying thesis.

## Data protection

The original message text is **never stored in readable form**. Only a SHA-256
hash (for deduplication) and the anonymized text are persisted. Because the
transformation is reversible in principle and re-identification could occur with
additional information, the approach is **pseudonymization** under the GDPR
(Art. 4(5)); stored data therefore remains personal data and is handled with
data minimization in mind.

## Local run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
uvicorn main:app --reload --port 8000
```

`requirements-dev.txt` holds extra dependencies (`pandas`, `seqeval`) needed
only by the offline helper scripts `prepare_dataset.py` and `evaluate.py` -
not required to run the API itself.

The LLM engines are entirely optional: without any provider API key set,
the app runs exactly as before (`local` engine only). To enable one or more
providers, copy `.env.example` to `.env` and set `GEMINI_API_KEY` (the
default provider - has a usable free tier) and/or, if you have your own key
for them, `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`; see
[`APPLY_V8.md`](APPLY_V8.md) for how to apply this version on top of an
existing V7 checkout.

## Future work

SQLite (used here for both engines' results) is a placeholder for local
development and Hugging Face Spaces hosting. A move to a self-hosted
deployment with MongoDB is planned but out of scope for this version.
