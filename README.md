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
| V3 | `v3.0` | Dynamic sentiment model selection from the Hugging Face Hub at request time, plus XML export of stored records. |

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
| POST | `/analyze` | Full pipeline on a single message |
| POST | `/predict` | Sentiment only (backward compatible) |
| POST | `/ingest` | Batch ingest of a CSV/JSON file |
| GET | `/records` | Recent stored (anonymized) records |
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
