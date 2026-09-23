"""
Database layer (SQLite by default, PostgreSQL optional).

Stores the structured output of the pipeline in a pseudonymized form. The
original text is never stored in readable form: only a SHA-256 hash (for
deduplication) and the anonymized text are persisted, together with the
extracted entities, topic and sentiment.

SQLite was chosen for the prototype (serverless, zero-config, portable) and
remains the default. Setting the ``DATABASE_URL`` environment variable to a
``postgres://`` / ``postgresql://`` connection string switches this module to
PostgreSQL instead - the rest of the application (main.py, pipeline.py) is
completely unaware of which backend is active, since every query below still
goes through the same functions. This mirrors how the LLM engine is selected
(env var, both branches supported side by side) rather than a one-way
migration.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from processors import anonymizer

DATABASE_URL = os.getenv("DATABASE_URL")
IS_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")))

# Export formats supported by export_records(), each mapped to its HTTP
# media type. Adding a new format only requires a new "_export_<fmt>"
# function plus an entry here.
EXPORT_FORMATS = {
    "xml": "application/xml",
    "json": "application/json",
    "csv": "text/csv",
}

# ---------------------------------------------------------------------------
# Database location: project folder on Windows, /tmp on Linux (HF Spaces).
# Only relevant for the SQLite backend - ignored when DATABASE_URL selects
# PostgreSQL instead.
# ---------------------------------------------------------------------------
if os.name == "nt":
    DB_PATH = os.getenv(
        "CONV_DB_PATH",
        str((Path(__file__).resolve().parent / "conversation_logs.db")),
    )
else:
    DB_PATH = os.getenv("CONV_DB_PATH", "/tmp/conversation_logs.db")

_db_lock = threading.Lock()

# SQLite columns are added one at a time via ALTER TABLE (no "ADD COLUMN IF
# NOT EXISTS" support), so a duplicate-column error on an already-migrated
# database is expected and ignored. PostgreSQL supports "IF NOT EXISTS"
# directly, so its migration loop (in _get_postgres_connection) needs no
# such try/except.
_NEW_COLUMNS = (
    ("engine", "TEXT DEFAULT 'local'"),
    ("provider", "TEXT"),
    ("model", "TEXT"),
    ("latency_ms", "INTEGER"),
    ("cost_usd", "REAL"),
)


def _get_sqlite_connection() -> sqlite3.Connection:
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
    # Since V8: which engine produced the record ('local' or 'llm') and,
    # for the LLM engine, which provider/model answered plus its latency and
    # cost. Added via ALTER TABLE so existing conversation_logs.db files from
    # earlier versions keep working.
    for name, coltype in _NEW_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE records ADD COLUMN {name} {coltype}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


def _get_postgres_connection():
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "DATABASE_URL points to PostgreSQL but the 'psycopg2-binary' "
            "package is not installed (pip install psycopg2-binary)."
        ) from exc

    conn = psycopg2.connect(DATABASE_URL)
    # Bez autocommitu otevírá psycopg2 implicitní transakci i pro čisté
    # SELECT dotazy (get_records/stats nikdy nevolají commit ani rollback);
    # pokud kterýkoli dotaz na sdíleném spojení někdy selže, zůstane
    # transakce ve stavu "aborted" a UPLNE VŠECHNY další dotazy na stejném
    # spojení pak hned selžou, dokud se transakce nezruší. SQLite se takto
    # nechová (proto to na SQLite fungovalo bez problémů) - autocommit tomu
    # u PostgreSQL předchází a zároveň sjednocuje chování obou backendů.
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id SERIAL PRIMARY KEY,
            created_at TEXT NOT NULL,
            original_hash TEXT NOT NULL,
            anonymized_text TEXT NOT NULL,
            entities TEXT,
            topic TEXT,
            topic_score REAL,
            sentiment TEXT,
            sentiment_score REAL,
            conversation_id TEXT,
            source TEXT
        )
        """
    )
    for name, coltype in _NEW_COLUMNS:
        cur.execute(f"ALTER TABLE records ADD COLUMN IF NOT EXISTS {name} {coltype}")
    conn.commit()
    cur.close()
    return conn


def _get_connection():
    return _get_postgres_connection() if IS_POSTGRES else _get_sqlite_connection()


def _execute(query: str, params: tuple = ()):
    """
    Run a query on whichever backend is active. Every query in this module
    is written with SQLite's ``?`` placeholder; PostgreSQL (via psycopg2)
    expects ``%s`` instead, so it is translated here - the one place that
    needs to know the difference. Both backends return a cursor-like object
    supporting ``fetchone()``/``fetchall()``.

    PostgreSQL specifically: a managed/serverless provider (e.g. Neon) can
    close an idle connection server-side (scale-to-zero, idle timeout) at
    any time. Since this module keeps ONE long-lived connection for the
    whole process lifetime, that leaves it permanently broken
    (``psycopg2.InterfaceError: connection already closed``) for every
    request from then on until the process restarts - reconnect once and
    retry instead of letting that happen.
    """
    global _db_conn

    if not IS_POSTGRES:
        return _db_conn.execute(query, params)

    import psycopg2

    translated = query.replace("?", "%s")
    try:
        cur = _db_conn.cursor()
        cur.execute(translated, params)
        return cur
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as exc:
        print(f"[db] PostgreSQL connection lost ({exc}); reconnecting...")
        _db_conn = _get_postgres_connection()
        cur = _db_conn.cursor()
        cur.execute(translated, params)
        return cur


_db_conn = _get_connection()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_record(result: dict[str, Any], source: str = "single") -> None:
    """
    Persist one pipeline result. The original text is hashed (not stored);
    the anonymized text is stored. If the result does not already contain an
    anonymized text, it is anonymized here as a safeguard.

    Since V8, ``result`` may come from either engine: the local BERT
    pipeline (no ``engine``/``provider`` keys - stored as ``engine="local"``,
    ``provider``/``model``/``latency_ms``/``cost_usd`` left ``NULL``) or the
    LLM pipeline (``engine="llm"``, plus ``provider``, ``model``,
    ``latency_ms``, ``cost_usd``). Both land in the same ``records`` table
    and are indistinguishable in the schema except for these columns, so the
    dashboard can list and export them together.
    """
    original_text = result.get("text", "") or ""
    anonymized = result.get("anonymized_text") or anonymizer.anonymize_text(original_text)
    original_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()

    entities_json = json.dumps(result.get("entities", []), ensure_ascii=False)

    with _db_lock:
        _execute(
            """
            INSERT INTO records (
                created_at, original_hash, anonymized_text, entities,
                topic, topic_score, sentiment, sentiment_score,
                conversation_id, source,
                engine, provider, model, latency_ms, cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                result.get("engine") or "local",
                result.get("provider"),
                result.get("model"),
                result.get("latency_ms"),
                result.get("cost_usd"),
            ),
        )
        _db_conn.commit()


# Sentiment is free text in the schema (custom models can return anything),
# but only these three values are treated as genuine sentiment labels for
# grouping/filtering purposes - everything else (e.g. nlptown's "5 stars",
# "1 star") is bucketed as "other" rather than shown as its own noisy group.
_KNOWN_SENTIMENTS = ("positive", "negative", "neutral")

# SQL CASE expression normalizing a raw sentiment value to one of
# positive/negative/neutral/other (NULL stays NULL) - standard SQL, works
# unchanged on both SQLite and PostgreSQL.
_SENTIMENT_BUCKET_SQL = (
    "CASE "
    "WHEN sentiment IS NULL THEN NULL "
    "WHEN LOWER(sentiment) IN ('positive', 'negative', 'neutral') THEN LOWER(sentiment) "
    "ELSE 'other' "
    "END"
)


def _build_filter(
    engine: str | None = None,
    topic: str | None = None,
    sentiment: str | None = None,
    provider: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """
    Shared ``WHERE`` clause for ``get_records``/``_fetch_export_rows``:
    optionally narrow to one engine (``"local"`` | ``"llm"``), one topic, one
    sentiment label, or one LLM provider (``"anthropic"`` | ``"google"`` |
    ``"openai"`` - ``NULL`` for local-engine records, so this filter is only
    meaningful together with ``engine="llm"``). Any combination may be used
    together; omitted filters are simply not applied.
    """
    conditions = []
    params: tuple[Any, ...] = ()
    if engine:
        conditions.append("engine = ?")
        params += (engine,)
    if topic:
        conditions.append("topic = ?")
        params += (topic,)
    if sentiment:
        # case-insensitive; "other" means "anything outside the three known
        # labels" (e.g. nlptown's "5 stars") rather than a literal match.
        if sentiment.lower() == "other":
            placeholders = ", ".join(["?"] * len(_KNOWN_SENTIMENTS))
            conditions.append(f"LOWER(sentiment) NOT IN ({placeholders})")
            params += _KNOWN_SENTIMENTS
        else:
            conditions.append("LOWER(sentiment) = ?")
            params += (sentiment.lower(),)
    if provider:
        conditions.append("provider = ?")
        params += (provider,)
    clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return clause, params


def get_records(
    limit: int = 100,
    engine: str | None = None,
    topic: str | None = None,
    sentiment: str | None = None,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return the most recent records (anonymized only).

    ``engine`` optionally filters to ``"local"`` (BERT pipeline) or ``"llm"``
    (any provider); ``topic``/``sentiment`` filter to an exact label;
    ``provider`` filters to one LLM provider (``"anthropic"``/``"google"``/
    ``"openai"``) - use it together with ``engine="llm"`` to separate results
    from different providers tested on the same batch. Any combination may
    be used together; omitted/``None`` means "don't filter on this".
    """
    where_clause, where_params = _build_filter(engine, topic, sentiment, provider)
    query = f"""
        SELECT id, created_at, anonymized_text, entities,
               topic, topic_score, sentiment, sentiment_score,
               conversation_id, source,
               engine, provider, model, latency_ms, cost_usd
        FROM records{where_clause}
        ORDER BY id DESC
        LIMIT ?
    """
    params = where_params + (limit,)

    with _db_lock:
        cur = _execute(query, params)
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
                "engine": r[10],
                "provider": r[11],
                "model": r[12],
                "latency_ms": r[13],
                "cost_usd": r[14],
            }
        )
    return records


def stats() -> dict[str, Any]:
    """Aggregate statistics for the dashboard (counts by sentiment / topic / provider)."""
    with _db_lock:
        total = _execute("SELECT COUNT(*) FROM records").fetchone()[0]
        # normalized to positive/negative/neutral (case-insensitive); any
        # other raw value (e.g. nlptown's "5 stars") is bucketed as "other"
        # instead of showing up as its own noisy group.
        by_sentiment = _execute(
            f"SELECT {_SENTIMENT_BUCKET_SQL}, COUNT(*) FROM records GROUP BY 1"
        ).fetchall()
        by_topic = _execute(
            "SELECT topic, COUNT(*) FROM records GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()
        by_provider = _execute(
            "SELECT provider, COUNT(*) FROM records WHERE provider IS NOT NULL GROUP BY provider"
        ).fetchall()

    return {
        "total": total,
        "by_sentiment": {(s or "unknown"): c for s, c in by_sentiment},
        "by_topic": {(t or "unknown"): c for t, c in by_topic},
        "by_provider": dict(by_provider),
    }


def _fetch_export_rows(
    limit: int | None,
    engine: str | None = None,
    topic: str | None = None,
    sentiment: str | None = None,
    provider: str | None = None,
) -> list[tuple[Any, ...]]:
    """Raw rows (most recent first) shared by every export format."""
    where_clause, params = _build_filter(engine, topic, sentiment, provider)
    query = f"""
        SELECT id, created_at, anonymized_text, entities,
               topic, topic_score, sentiment, sentiment_score,
               conversation_id, source,
               engine, provider, model, latency_ms, cost_usd
        FROM records{where_clause}
        ORDER BY id DESC
    """
    if limit is not None:
        query += " LIMIT ?"
        params += (limit,)

    with _db_lock:
        return _execute(query, params).fetchall()


def _row_entities(row: tuple[Any, ...]) -> list[dict[str, Any]]:
    try:
        return json.loads(row[3]) if row[3] else []
    except Exception:
        return []


def _export_xml(rows: list[tuple[Any, ...]]) -> bytes:
    root = ET.Element("records")
    for r in rows:
        record_el = ET.SubElement(root, "record", id=str(r[0]))
        ET.SubElement(record_el, "created_at").text = r[1]
        ET.SubElement(record_el, "anonymized_text").text = r[2]
        ET.SubElement(record_el, "topic").text = r[4]
        ET.SubElement(record_el, "topic_score").text = (
            str(r[5]) if r[5] is not None else None
        )
        ET.SubElement(record_el, "sentiment").text = r[6]
        ET.SubElement(record_el, "sentiment_score").text = (
            str(r[7]) if r[7] is not None else None
        )
        ET.SubElement(record_el, "conversation_id").text = r[8]
        ET.SubElement(record_el, "source").text = r[9]
        ET.SubElement(record_el, "engine").text = r[10]
        ET.SubElement(record_el, "provider").text = r[11]
        ET.SubElement(record_el, "model").text = r[12]
        ET.SubElement(record_el, "latency_ms").text = (
            str(r[13]) if r[13] is not None else None
        )
        ET.SubElement(record_el, "cost_usd").text = (
            str(r[14]) if r[14] is not None else None
        )

        entities_el = ET.SubElement(record_el, "entities")
        for ent in _row_entities(r):
            ET.SubElement(
                entities_el,
                "entity",
                type=str(ent.get("type", "")),
                score=str(ent.get("score", "")),
            ).text = ent.get("text", "")

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _export_json(rows: list[tuple[Any, ...]]) -> bytes:
    records = [
        {
            "id": r[0],
            "created_at": r[1],
            "anonymized_text": r[2],
            "entities": _row_entities(r),
            "topic": r[4],
            "topic_score": r[5],
            "sentiment": r[6],
            "sentiment_score": r[7],
            "conversation_id": r[8],
            "source": r[9],
            "engine": r[10],
            "provider": r[11],
            "model": r[12],
            "latency_ms": r[13],
            "cost_usd": r[14],
        }
        for r in rows
    ]
    return json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")


def _export_csv(rows: list[tuple[Any, ...]]) -> bytes:
    # CSV is flat, so entities (a nested list) are serialized into a single
    # "TYPE:text" cell per entity, semicolon-separated.
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "created_at",
            "anonymized_text",
            "topic",
            "topic_score",
            "sentiment",
            "sentiment_score",
            "conversation_id",
            "source",
            "engine",
            "provider",
            "model",
            "latency_ms",
            "cost_usd",
            "entities",
        ]
    )
    for r in rows:
        entities_cell = "; ".join(
            f"{ent.get('type', '')}:{ent.get('text', '')}" for ent in _row_entities(r)
        )
        writer.writerow(
            [r[0], r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], entities_cell]
        )

    # utf-8-sig (BOM) so Excel opens the file with correct encoding.
    return buffer.getvalue().encode("utf-8-sig")


def export_records(
    fmt: str,
    limit: int | None = None,
    engine: str | None = None,
    topic: str | None = None,
    sentiment: str | None = None,
    provider: str | None = None,
) -> tuple[bytes, str]:
    """
    Export stored (anonymized) records in the given format.

    Returns ``(content_bytes, media_type)``. Raises ``ValueError`` for an
    unsupported ``fmt`` so the API layer can turn it into a clean 400
    response. ``limit`` caps the number of most recent records exported
    (``None`` exports everything); ``engine``/``topic``/``sentiment``/
    ``provider`` optionally filter, same as ``get_records``.
    """
    fmt = (fmt or "xml").strip().lower()
    if fmt not in EXPORT_FORMATS:
        supported = ", ".join(sorted(EXPORT_FORMATS))
        raise ValueError(f"Unsupported export format '{fmt}'. Supported: {supported}.")

    rows = _fetch_export_rows(limit, engine, topic, sentiment, provider)
    exporter = {"xml": _export_xml, "json": _export_json, "csv": _export_csv}[fmt]
    return exporter(rows), EXPORT_FORMATS[fmt]
