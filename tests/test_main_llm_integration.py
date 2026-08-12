"""
Testy integrace LLM enginu do hlavní aplikace (main.py).

Srovnávací vrstva (samostatná databáze `llm_client`u, /api/llm/dashboard)
byla z rozsahu odstraněna - srovnání BERT vs. LLM proběhne ručně a jeho
výsledky budou popsány přímo v diplomové práci. Místo toho se výsledky LLM
enginu ukládají do STEJNÉ tabulky `records` jako výsledky lokální BERT
pipeline (přes `db.save_record`), aby šly zobrazit a exportovat společně.
"""

from __future__ import annotations

import os
import tempfile

# Must happen before `db` (imported transitively via `main`) opens its
# connection, so these tests never touch the real project's
# conversation_logs.db.
_TEST_DB_FD, _TEST_DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_TEST_DB_FD)
os.environ["CONV_DB_PATH"] = _TEST_DB_PATH

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import db
import main
from llm_client.client import CallMeta, LLMClient
from llm_client.pipeline import LLMPipeline
from llm_client.schemas import AnalysisResult, Entity, Sentiment, TopicScore

client = TestClient(main.app)


class FakeLLMClient(LLMClient):
    def __init__(self, payload=None):
        self.payload = payload or {
            "sentiment": {"label": "positive", "score": 0.7},
            "entities": [{"text": "Praha", "label": "LOC", "score": 0.8}],
            "topics": [{"label": "praise and positive feedback", "score": 0.9}],
            "pii": [],
            "pseudonymized_text": "Diky moc!",
        }

    def analyze(self, text):
        meta = CallMeta(
            latency_ms=250,
            tokens_in=20,
            tokens_out=8,
            cost_usd=0.0021,
            model="fake-claude",
            provider="anthropic",
        )
        return self.payload, meta


def _fake_llm_pipeline(*_args, **_kwargs):
    return LLMPipeline(provider="anthropic", client=FakeLLMClient())


def test_llm_result_to_response_maps_analysis_result():
    result = AnalysisResult(
        engine="llm",
        provider="anthropic",
        model="fake-claude",
        sentiment=Sentiment(label="negative", score=0.8),
        entities=[Entity(text="Jan", label="PER", start=0, end=3, score=0.9)],
        topics=[TopicScore(label="complaint", score=0.6), TopicScore(label="other", score=0.1)],
        pseudonymized_text="<PERSON_1> je nespokojeny.",
        latency_ms=180,
        cost_usd=0.003,
    )
    response = main._llm_result_to_response("Jan je nespokojeny.", result)

    assert response["engine"] == "llm"
    assert response["provider"] == "anthropic"
    assert response["model"] == "fake-claude"
    assert response["topic"] == "complaint"
    assert response["sentiment"] == "negative"
    assert response["latency_ms"] == 180
    assert response["cost_usd"] == 0.003
    assert response["entities"][0] == {
        "text": "Jan",
        "type": "PER",
        "start": 0,
        "end": 3,
        "score": 0.9,
    }


def test_analyze_llm_engine_is_stored_in_same_records_table_as_local():
    with patch.object(main, "LLMPipeline", side_effect=_fake_llm_pipeline):
        resp = client.post(
            "/analyze", json={"text": "Diky moc za pomoc!", "engine": "anthropic"}
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["engine"] == "llm"
    assert body["provider"] == "anthropic"

    records = db.get_records(limit=5)
    assert records[0]["engine"] == "llm"
    assert records[0]["provider"] == "anthropic"
    assert records[0]["model"] == "fake-claude"
    assert records[0]["latency_ms"] == 250
    assert records[0]["cost_usd"] == pytest.approx(0.0021)


def test_local_engine_records_keep_null_provider_and_cost():
    with patch("main.process_message") as mock_process:
        mock_process.return_value = {
            "text": "ahoj",
            "anonymized_text": "ahoj",
            "entities": [],
            "topic": "other",
            "topic_score": 0.1,
            "sentiment": "neutral",
            "sentiment_score": 0.5,
        }
        resp = client.post("/analyze", json={"text": "ahoj", "engine": "local"})
    assert resp.status_code == 200, resp.text

    records = db.get_records(limit=5)
    assert records[0]["engine"] == "local"
    assert records[0]["provider"] is None
    assert records[0]["cost_usd"] is None


def test_ingest_llm_engine_stores_each_message_in_records():
    csv_bytes = b'text\n"Skvela podpora!"\n"Objednavka nedorazila."\n'
    with patch.object(main, "LLMPipeline", side_effect=_fake_llm_pipeline):
        resp = client.post(
            "/ingest",
            files={"file": ("msgs.csv", csv_bytes, "text/csv")},
            data={"engine": "anthropic"},
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        status = None
        for _ in range(100):
            status = client.get(f"/ingest/status/{job_id}").json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.05)

    assert status["status"] == "done", status
    assert status["result"]["engine"] == "anthropic"
    assert status["result"]["processed"] == 2
    assert status["result"]["failed"] == 0

    records = db.get_records(limit=10)
    llm_records = [r for r in records if r["provider"] == "anthropic"]
    assert len(llm_records) >= 2


def test_analyze_llm_engine_missing_key_returns_clean_400():
    resp = client.post("/analyze", json={"text": "hi", "engine": "google"})
    assert resp.status_code == 400
    assert "GEMINI_API_KEY" in resp.json()["detail"]


def test_engines_endpoint_lists_local_and_llm_providers():
    resp = client.get("/engines")
    assert resp.status_code == 200
    ids = {e["id"] for e in resp.json()["engines"]}
    assert ids == {"local", "anthropic", "openai", "google"}


def test_records_can_be_filtered_by_engine():
    with patch("main.process_message") as mock_process:
        mock_process.return_value = {
            "text": "filter test local",
            "anonymized_text": "filter test local",
            "entities": [],
            "topic": "other",
            "topic_score": 0.1,
            "sentiment": "neutral",
            "sentiment_score": 0.5,
        }
        client.post("/analyze", json={"text": "filter test local", "engine": "local"})

    with patch.object(main, "LLMPipeline", side_effect=_fake_llm_pipeline):
        client.post(
            "/analyze", json={"text": "filter test llm", "engine": "anthropic"}
        )

    local_records = client.get("/records?engine=local").json()
    assert all(r["engine"] == "local" for r in local_records)
    assert any(r["anonymized_text"] == "filter test local" for r in local_records)

    llm_records = client.get("/records?engine=llm").json()
    assert all(r["engine"] == "llm" for r in llm_records)
    # FakeLLMClient always returns the same fixed payload regardless of the
    # input text, so the stored anonymized_text is its fixed
    # pseudonymized_text ("Diky moc!"), not the original input.
    assert any(r["anonymized_text"] == "Diky moc!" for r in llm_records)

    all_records = client.get("/records").json()
    assert len(all_records) >= len(local_records) + len(llm_records)


def test_records_invalid_engine_filter_returns_clean_400():
    resp = client.get("/records?engine=bogus")
    assert resp.status_code == 400
    assert "engine" in resp.json()["detail"].lower()


def test_records_export_respects_engine_filter():
    resp = client.get("/records/export?format=json&engine=llm")
    assert resp.status_code == 200, resp.text
    import json as _json

    rows = _json.loads(resp.content)
    assert all(r["engine"] == "llm" for r in rows)


def test_records_can_be_filtered_by_topic_and_sentiment():
    with patch("main.process_message") as mock_process:
        mock_process.return_value = {
            "text": "billing complaint",
            "anonymized_text": "billing complaint",
            "entities": [],
            "topic": "billing and payments",
            "topic_score": 0.7,
            "sentiment": "negative",
            "sentiment_score": 0.8,
        }
        client.post("/analyze", json={"text": "billing complaint", "engine": "local"})

        mock_process.return_value = {
            "text": "great service",
            "anonymized_text": "great service",
            "entities": [],
            "topic": "praise and positive feedback",
            "topic_score": 0.9,
            "sentiment": "positive",
            "sentiment_score": 0.95,
        }
        client.post("/analyze", json={"text": "great service", "engine": "local"})

    billing_records = client.get("/records?topic=billing and payments").json()
    assert all(r["topic"] == "billing and payments" for r in billing_records)
    assert any(r["anonymized_text"] == "billing complaint" for r in billing_records)

    negative_records = client.get("/records?sentiment=negative").json()
    assert all(r["sentiment"] == "negative" for r in negative_records)
    assert any(r["anonymized_text"] == "billing complaint" for r in negative_records)

    # combined filter: topic AND sentiment together
    combined = client.get(
        "/records?topic=billing and payments&sentiment=negative"
    ).json()
    assert any(r["anonymized_text"] == "billing complaint" for r in combined)

    # a combination with no matching record returns a clean empty list, not
    # an error - topic/sentiment aren't a fixed enum (custom models can
    # produce arbitrary labels), so "no matches" is a valid, non-error result
    no_match = client.get(
        "/records?topic=billing and payments&sentiment=positive"
    ).json()
    assert no_match == []


def test_records_export_respects_topic_and_sentiment_filters():
    resp = client.get(
        "/records/export?format=json&topic=billing and payments&sentiment=negative"
    )
    assert resp.status_code == 200, resp.text
    import json as _json

    rows = _json.loads(resp.content)
    assert all(r["topic"] == "billing and payments" and r["sentiment"] == "negative" for r in rows)
