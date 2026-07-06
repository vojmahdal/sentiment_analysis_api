"""
Ingest component.

Accepts conversation data as a batch (CSV or JSON), normalizes it into a unified
structure, and hands it over to the NLP pipeline. This is the "ingest" module
required by the assignment.

Unified message schema:
    {
        "conversation_id": <str | None>,
        "speaker":         <str | None>,
        "timestamp":       <str | None>,
        "text":            <str>          # required
    }
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

# Column / key names we accept for the message text, in priority order.
_TEXT_KEYS = ["text", "message", "content", "body", "utterance"]
_ID_KEYS = ["conversation_id", "conv_id", "id", "dialog_id"]
_SPEAKER_KEYS = ["speaker", "author", "role", "user"]
_TIME_KEYS = ["timestamp", "time", "created_at", "date"]


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
        # case-insensitive match
        for rk in row:
            if rk.lower() == k and row[rk] not in (None, ""):
                return row[rk]
    return None


def _normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    text = _first_present(row, _TEXT_KEYS)
    if text is None or str(text).strip() == "":
        return None
    return {
        "conversation_id": _first_present(row, _ID_KEYS),
        "speaker": _first_present(row, _SPEAKER_KEYS),
        "timestamp": _first_present(row, _TIME_KEYS),
        "text": str(text).strip(),
    }


def parse_csv(raw: bytes | str) -> list[dict[str, Any]]:
    """Parse CSV content into a list of normalized messages."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    messages = []
    for row in reader:
        norm = _normalize_row(row)
        if norm:
            messages.append(norm)
    return messages


def parse_json(raw: bytes | str) -> list[dict[str, Any]]:
    """
    Parse JSON content into a list of normalized messages.

    Accepts either a list of objects, or an object with a "messages"/"data" list,
    or a single object.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    data = json.loads(raw)

    if isinstance(data, dict):
        for key in ("messages", "data", "conversations", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    messages = []
    for row in data:
        if isinstance(row, dict):
            norm = _normalize_row(row)
            if norm:
                messages.append(norm)
        elif isinstance(row, str) and row.strip():
            messages.append(
                {"conversation_id": None, "speaker": None, "timestamp": None, "text": row.strip()}
            )
    return messages


def parse_upload(filename: str, raw: bytes) -> list[dict[str, Any]]:
    """Dispatch to the right parser based on the file extension."""
    name = (filename or "").lower()
    if name.endswith(".json"):
        return parse_json(raw)
    if name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt"):
        return parse_csv(raw)
    # try JSON first, then CSV, as a last resort
    try:
        return parse_json(raw)
    except Exception:
        return parse_csv(raw)
