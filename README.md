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
| POST | `/analyze` | Full pipeline on a single message |
| POST | `/predict` | Sentiment only (backward compatible) |
| POST | `/ingest` | Batch ingest of a CSV/JSON file |
| GET | `/records` | Recent stored (anonymized) records |
| GET | `/records/export.xml` | Stored records exported as XML |
| GET | `/stats` | Aggregate statistics |

### Example

```bash
curl -X POST https://<space-url>/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hi, John Smith here, my order never arrived. Email john@example.com"}'
```

```bash
curl -X POST https://<space-url>/ingest -F "file=@sample_chats.csv"
```

```bash
curl https://<space-url>/records/export.xml -o records.xml
```

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

The web dashboard exposes this as **radio buttons** for each step - one per
suggested model, plus a "Custom model" option that reveals a text field
where any other Hugging Face repo id can be typed in.

**Security note:** loaded pipelines never use `trust_remote_code=True`, so an
arbitrary/untrusted model id supplied by a caller cannot execute custom
Python code inside the server process - it is limited to standard
`transformers` inference for the given task. An invalid or incompatible
model id results in a clean HTTP 400 response instead of crashing the
server.

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
