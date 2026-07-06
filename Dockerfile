FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    CONV_DB_PATH=/tmp/conversation_logs.db

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# spaCy model required by Presidio for name/location detection
RUN python -m spacy download en_core_web_lg

# Pre-download the Hugging Face models at build time so the first request is fast.
# (Comment out to download lazily on first use instead.)
RUN python - <<'PY'
from transformers import pipeline
pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Models cached.")
PY

# Application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
