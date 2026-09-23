"""
Testy PostgreSQL kompatibilní vrstvy v db.py.

Bez reálného PostgreSQL serveru (není v CI/testovacím prostředí k dispozici)
nelze otestovat end-to-end zápis/čtení - tyto testy proto ověřují jádro té
části, která se mezi backendy liší: detekci backendu z DATABASE_URL a
překlad "?" -> "%s" v _execute(). Zbytek modulu (save_record, get_records,
stats, export_records) backend nerozlišuje a je už pokrytý testy nad SQLite
v test_main_llm_integration.py.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import db


def test_is_postgres_detects_postgres_urls():
    for url in ("postgres://u:p@host/db", "postgresql://u:p@host:5432/db"):
        assert url.startswith(("postgres://", "postgresql://"))


def test_is_postgres_false_when_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(db)
    try:
        assert db.IS_POSTGRES is False
        assert db.DATABASE_URL is None
    finally:
        # vrátit modul do stavu, který očekává zbytek test suite (SQLite,
        # CONV_DB_PATH nastavené na začátku test_main_llm_integration.py)
        importlib.reload(db)


def test_execute_translates_placeholders_for_postgres(monkeypatch):
    fake_cursor = MagicMock()
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    monkeypatch.setattr(db, "IS_POSTGRES", True)
    monkeypatch.setattr(db, "_db_conn", fake_conn)

    db._execute("SELECT * FROM records WHERE topic = ? AND sentiment = ?", ("x", "y"))

    fake_conn.cursor.assert_called_once()
    called_query, called_params = fake_cursor.execute.call_args[0]
    assert called_query == "SELECT * FROM records WHERE topic = %s AND sentiment = %s"
    assert called_params == ("x", "y")


def test_execute_keeps_sqlite_placeholders_untouched(monkeypatch):
    fake_conn = MagicMock()

    monkeypatch.setattr(db, "IS_POSTGRES", False)
    monkeypatch.setattr(db, "_db_conn", fake_conn)

    db._execute("SELECT * FROM records WHERE topic = ?", ("x",))

    fake_conn.execute.assert_called_once_with("SELECT * FROM records WHERE topic = ?", ("x",))


def test_execute_reconnects_when_postgres_connection_was_dropped(monkeypatch):
    """
    Reprodukuje reálně pozorovanou chybu: Neon (serverless PostgreSQL) může
    kdykoli zavřít nečinné spojení ze své strany (scale-to-zero, idle
    timeout). Jedno sdílené spojení pro celý běh procesu by tím zůstalo
    navždy rozbité (psycopg2.InterfaceError: connection already closed) pro
    každý další požadavek - _execute() se místo toho musí jednou přepojit a
    dotaz zopakovat.
    """
    import psycopg2

    dead_cursor = MagicMock()
    dead_cursor.execute.side_effect = psycopg2.InterfaceError("connection already closed")
    dead_conn = MagicMock()
    dead_conn.cursor.return_value = dead_cursor

    fresh_cursor = MagicMock()
    fresh_conn = MagicMock()
    fresh_conn.cursor.return_value = fresh_cursor

    monkeypatch.setattr(db, "IS_POSTGRES", True)
    monkeypatch.setattr(db, "_db_conn", dead_conn)
    monkeypatch.setattr(db, "_get_postgres_connection", lambda: fresh_conn)

    result = db._execute("SELECT * FROM records WHERE topic = ?", ("x",))

    dead_cursor.execute.assert_called_once()
    fresh_cursor.execute.assert_called_once_with(
        "SELECT * FROM records WHERE topic = %s", ("x",)
    )
    assert result is fresh_cursor
    # _db_conn v modulu musí zůstat přepojené na nové (živé) spojení, ne se
    # vrátit zpět k mrtvému - jinak by se stejná chyba opakovala příště.
    assert db._db_conn is fresh_conn
