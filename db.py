"""
Database layer (SQLite).

Stores the structured output of the pipeline in a pseudonymized form. The
original text is never stored in readable form: only a SHA-256 hash (for
deduplication) and the anonymized text are persisted, together with the
extracted entities, topic and sentiment.

SQLite was chosen for the prototype (serverless, zero-config, portable). The
layer is intentionally small so it can be swapped for PostgreSQL in a
production deployment without changing the rest of the application.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from processors import anonymizer

# ---------------------------------------------------------------------------
# Database location: project folder on Windows, /tmp on Linux (HF Spaces).
# ---------------------------------------------------------------------------
if os.name == "nt":
    DB_PATH = os.getenv(
        "CONV_DB_PATH",
        str((Path(__file__).resolve().parent / "conversation_logs.db")),
    )
else:
    DB_PATH = os.getenv("CONV_DB_PATH", "/tmp/conversation_logs.db")

_db_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            original_hash TEXT NOT NULL,
            anonymized_text TEXT NOT NULL,
            entities TEXT,            -- JSON array
            topic TEXT,
            topic_score REAL,
            sentiment TEXT,
            sentiment_score REAL,
            conversation_id TEXT,
            source TEXT               -- 'single' or 'ingest'
        )
        """
    )
    conn.commit()
    return conn


_db_conn = _get_connection()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_record(result: dict[str, Any], source: str = "single") -> None:
    """
    Persist one pipeline result. The original text is hashed (not stored);
    the anonymized text is stored. If the result does not already contain an
    anonymized text, it is anonymized here as a safeguard.
    """
    original_text = result.get("text", "") or ""
    anonymized = result.get("anonymized_text") or anonymizer.anonymize_text(original_text)
    original_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()

    entities_json = json.dumps(result.get("entities", []), ensure_ascii=False)

    with _db_lock:
        _db_conn.execute(
            """
            INSERT INTO records (
                created_at, original_hash, anonymized_text, entities,
                topic, topic_score, sentiment, sentiment_score,
                conversation_id, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utcnow(),
                original_hash,
                anonymized,
                entities_json,
                result.get("topic"),
                result.get("topic_score"),
                result.get("sentiment"),
                result.get("sentiment_score"),
                result.get("conversation_id"),
                source,
            ),
        )
        _db_conn.commit()


def get_records(limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent records (anonymized only)."""
    with _db_lock:
        cur = _db_conn.execute(
            """
            SELECT id, created_at, anonymized_text, entities,
                   topic, topic_score, sentiment, sentiment_score,
                   conversation_id, source
            FROM records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()

    records = []
    for r in rows:
        try:
            entities = json.loads(r[3]) if r[3] else []
        except Exception:
            entities = []
        records.append(
            {
                "id": r[0],
                "created_at": r[1],
                "anonymized_text": r[2],
                "entities": entities,
                "topic": r[4],
                "topic_score": r[5],
                "sentiment": r[6],
                "sentiment_score": r[7],
                "conversation_id": r[8],
                "source": r[9],
            }
        )
    return records


def stats() -> dict[str, Any]:
    """Aggregate statistics for the dashboard (counts by sentiment / topic)."""
    with _db_lock:
        total = _db_conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        by_sentiment = _db_conn.execute(
            "SELECT sentiment, COUNT(*) FROM records GROUP BY sentiment"
        ).fetchall()
        by_topic = _db_conn.execute(
            "SELECT topic, COUNT(*) FROM records GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()

    return {
        "total": total,
        "by_sentiment": {(s or "unknown"): c for s, c in by_sentiment},
        "by_topic": {(t or "unknown"): c for t, c in by_topic},
    }
