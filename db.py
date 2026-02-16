import os
import sqlite3
import threading
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# SQLite "database" for logging / anonymized storage
# ---------------------------------------------------------------------------

if os.name == "nt":
    # Windows (locally) – save to project folder
    DB_PATH = os.getenv(
        "SENTIMENT_DB_PATH",
        str((Path(__file__).resolve().parent / "sentiment_logs.db")),
    )
else:
    # Linux (e.g. Hugging Face Spaces) – /tmp
    DB_PATH = os.getenv("SENTIMENT_DB_PATH", "/tmp/sentiment_logs.db")

_db_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    """
    Get global SQLite connection; creates table on first use.
    Uses /tmp by default so it is compatible with Hugging Face Spaces (data may be wiped on restart, which is fine for demo / thesis).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            original_hash TEXT NOT NULL,
            anonymized_text TEXT NOT NULL,
            label TEXT NOT NULL,
            score REAL NOT NULL,
            client_ip TEXT
        )
        """
    )
    return conn


_db_conn = _get_connection()


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
_DIGITS_RE = re.compile(r"\d{3,}")

# very simple detection of names: sequence of 2–3 words starting with a capital letter
_NAME_SEQ_RE = re.compile(
    r"\b([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]+(?:\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]+){1,2})\b"
)


def _anonymize_names_outside_sentence_start(text: str) -> str:
  
    parts = re.split(r"(\n+|[.!?])", text)
    if len(parts) == 1:
        return _anonymize_names_skip_first_word(text)

    processed: list[str] = []

    for i in range(0, len(parts), 2):
        sentence = parts[i]
        terminator = parts[i + 1] if i + 1 < len(parts) else ""
        processed_sentence = _anonymize_names_skip_first_word(sentence)
        processed.append(processed_sentence + terminator)

    return "".join(processed)


def _anonymize_names_skip_first_word(sentence: str) -> str:

    if not sentence:
        return sentence

    m = re.match(r"^(\s*\S+)(.*)$", sentence, flags=re.DOTALL)
    if not m:
        return sentence

    first_word = m.group(1)
    rest = m.group(2)


    rest = _NAME_SEQ_RE.sub("[NAME]", rest)
    return first_word + rest


def anonymize_text(text: str) -> str:
    """
    Very simple anonymization:
    - e-mails -> [EMAIL]
    - phone numbers -> [PHONE]
    - longer sequences of digits -> [NUMBER]
    - name-like sequences like 'Jan Novák' -> [NAME], but not at the beginning of a sentence
    """
    t = _EMAIL_RE.sub("[EMAIL]", text)
    t = _PHONE_RE.sub("[PHONE]", t)
    t = _DIGITS_RE.sub("[NUMBER]", t)
    t = _anonymize_names_outside_sentence_start(t)
    return t


def log_to_db(original_text: str, label: str, score: float, client_ip: str | None) -> None:
    """
    Save record to SQLite:
    - hash original text (SHA256) due to impossibility to reconstruct
    - anonymized text according to anonymize_text
    """
    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    original_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
    anonymized = anonymize_text(original_text)

    with _db_lock:
        _db_conn.execute(
            """
            INSERT INTO logs (created_at, original_hash, anonymized_text, label, score, client_ip)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (created_at, original_hash, anonymized, label, float(score), client_ip),
        )
        _db_conn.commit()


def get_logs_from_db(limit: int = 100) -> list[dict[str, Any]]:

    with _db_lock:
        cur = _db_conn.execute(
            """
            SELECT id, created_at, anonymized_text, label, score, client_ip
            FROM logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "created_at": r[1],
            "anonymized_text": r[2],
            "label": r[3],
            "score": r[4],
            "client_ip": r[5],
        }
        for r in rows
    ]

