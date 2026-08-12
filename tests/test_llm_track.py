"""
Testy LLM větve (`llm_client.pipeline.LLMPipeline`).

Klient je nahrazen atrapou (`FakeClient`), takže testy běží bez API klíče
a bez nákladů. Tím je zároveň ověřeno, že návrh je testovatelný – jedna
z nefunkčních požadavků na systém.

Balíček `llm_client` už neobsahuje vlastní databázi, `evaluate` (srovnávací
metriky BERT vs. LLM) ani vlastní `ingest` parser - srovnání enginů bylo
z rozsahu vyřazeno (testování proběhne ručně a jeho výsledky budou popsány
přímo v diplomové práci) a LLM výsledky se ukládají stejnou cestou
(`db.save_record` hlavní aplikace) jako u lokální větve, viz `main.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from llm_client.client import CallMeta, LLMClient
from llm_client.pipeline import LLMPipeline, _fallback_pseudonymize
from llm_client.schemas import AnalysisResult


class FakeClient(LLMClient):
    """Deterministická atrapa poskytovatele."""

    def __init__(self, payload: Dict[str, Any] | None = None):
        self.payload = payload or {
            "sentiment": {"label": "negative", "score": 0.91},
            "entities": [{"text": "Jan Novák", "label": "PER", "score": 0.95}],
            "topics": [
                {"label": "billing and payments", "score": 0.8},
                {"label": "complaint", "score": 0.6},
            ],
            "pii": [
                {
                    "text": "jan@example.com",
                    "label": "EMAIL_ADDRESS",
                    "placeholder": "<EMAIL_ADDRESS_1>",
                }
            ],
            "pseudonymized_text": "Dobrý den, jsem <PERSON_1>.",
        }
        self.calls = 0

    def analyze(self, text: str) -> Tuple[Dict[str, Any], CallMeta]:
        self.calls += 1
        meta = CallMeta(
            latency_ms=120,
            tokens_in=310,
            tokens_out=140,
            cost_usd=0.0031,
            model="fake-model",
            provider="fake",
        )
        return self.payload, meta


@pytest.fixture
def pipeline() -> LLMPipeline:
    return LLMPipeline(provider="anthropic", mode="pre_pseudonymized", client=FakeClient())


# --------------------------------------------------------------------------

def test_analyze_returns_unified_contract(pipeline: LLMPipeline) -> None:
    result = pipeline.analyze("Dobrý den, jsem Jan Novák, jan@example.com.")
    assert isinstance(result, AnalysisResult)
    assert result.engine == "llm"
    assert result.sentiment.label == "negative"
    assert result.entities[0].label == "PER"
    assert result.top_topic == "billing and payments"
    assert result.cost_usd > 0


def test_labels_outside_taxonomy_are_normalised() -> None:
    client = FakeClient(
        {
            "sentiment": {"label": "POSITIVE", "score": 0.7},
            "entities": [{"text": "Brno", "label": "GPE", "score": 0.9}],
            "topics": [{"label": "refund request", "score": 0.5}],
            "pii": [],
            "pseudonymized_text": "x",
        }
    )
    pipe = LLMPipeline(provider="anthropic", client=client)
    result = pipe.analyze("Brno")
    assert result.sentiment.label == "positive"   # velká písmena
    assert result.entities[0].label == "LOC"      # GPE -> LOC
    assert result.topics[0].label == "other"      # mimo číselník


def test_raw_mode_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError):
        LLMPipeline(provider="anthropic", mode="raw", client=FakeClient())


def test_pii_is_masked_before_leaving_the_system() -> None:
    text = "Napiš mi na jan@example.com nebo na 604 123 456."
    masked, spans = _fallback_pseudonymize(text)
    assert "jan@example.com" not in masked
    assert "<EMAIL_ADDRESS_1>" in masked
    assert {s.label for s in spans} >= {"EMAIL_ADDRESS"}


def test_client_receives_masked_text() -> None:
    client = FakeClient()
    captured: list[str] = []
    original = client.analyze

    def spy(text: str):
        captured.append(text)
        return original(text)

    client.analyze = spy  # type: ignore[method-assign]
    LLMPipeline(provider="anthropic", client=client).analyze("piš na a@b.cz")
    assert "a@b.cz" not in captured[0]


def test_offsets_are_filled_in(pipeline: LLMPipeline) -> None:
    result = pipeline.analyze("Zdravím, tady Jan Novák.")
    entity = result.entities[0]
    assert entity.start is not None and entity.end is not None


def test_analyze_batch_reports_progress_and_preserves_order() -> None:
    client = FakeClient()
    pipe = LLMPipeline(provider="anthropic", client=client)

    calls: list[tuple[int, int]] = []
    texts = ["první", "druhá", "třetí"]
    results = pipe.analyze_batch(texts, on_progress=lambda done, total: calls.append((done, total)))

    assert len(results) == 3
    assert all(isinstance(r, AnalysisResult) for r in results)
    assert len(calls) == 3
    assert calls[-1] == (3, 3)
