"""
Testy pro V8 integraci llm_client do skutečného projektu.

Na rozdíl od test_llm_track.py, tyto testy cíleně ověřují místa, kde byl
`llm_client` upravován, aby odpovídal *reálnému* rozhraní tohoto projektu
(pipeline.process_message, processors.anonymizer, processors.topics),
místo předpokládaného `conversation_system.*` z jiné konverzace.

Srovnávací vrstva (`BertPipelineAdapter`, `execute_run`, vlastní databáze
`llm_client`u) byla z rozsahu odstraněna - LLM výsledky se teď ukládají přes
`db.save_record` hlavní aplikace stejně jako lokální výsledky, viz
`test_main_llm_integration.py`.
"""

from __future__ import annotations

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Pseudonymizace proti reálnému processors.anonymizer (ne fake)
# ---------------------------------------------------------------------------

def test_pipeline_bridges_to_real_anonymizer_module():
    """
    _load_presidio_anonymizer musí umět najít a zavolat SKUTEČNÝ
    processors.anonymizer (anonymize_text + detect_pii), ne fiktivní
    conversation_system.processors.anonymizer.pseudonymize.
    """
    from llm_client.pipeline import _load_presidio_anonymizer

    pseudonymize = _load_presidio_anonymizer()
    assert pseudonymize is not None, (
        "processors.anonymizer by měl být dostupný (je součástí stejného projektu) "
        "- pokud tohle selže, bridge stále odkazuje na starou cestu."
    )

    masked, spans = pseudonymize("Contact John Smith at john@example.com")
    assert "john@example.com" not in masked

    from processors import anonymizer as real_anonymizer

    if real_anonymizer.backend_name() == "presidio":
        # detect_pii() only returns structured spans when Presidio is actually
        # installed; in regex-fallback mode (e.g. this sandboxed test run,
        # where presidio-analyzer isn't installed) anonymize_text() still
        # masks the email via regex, but detect_pii() legitimately returns [].
        assert any(getattr(s, "entity_type", None) == "EMAIL_ADDRESS" for s in spans)
    else:
        assert spans == []


def test_llm_pipeline_uses_real_anonymizer_end_to_end():
    from llm_client.pipeline import LLMPipeline
    from llm_client.client import CallMeta, LLMClient

    class FakeClient(LLMClient):
        def __init__(self):
            self.seen_text = None

        def analyze(self, text):
            self.seen_text = text
            return (
                {
                    "sentiment": {"label": "negative", "score": 0.8},
                    "entities": [],
                    "topics": [{"label": "complaint", "score": 0.5}],
                    "pii": [],
                    "pseudonymized_text": text,
                },
                CallMeta(latency_ms=10, tokens_in=5, tokens_out=5, cost_usd=0.0, model="fake", provider="fake"),
            )

    client = FakeClient()
    pipe = LLMPipeline(provider="anthropic", mode="pre_pseudonymized", client=client)
    pipe.analyze("Email me at real.person@example.com")
    assert "real.person@example.com" not in client.seen_text


# ---------------------------------------------------------------------------
# Sjednocená taxonomie témat
# ---------------------------------------------------------------------------

def test_topic_taxonomy_matches_real_zero_shot_labels():
    from llm_client.config import TOPIC_LABELS
    from processors.topics import DEFAULT_LABELS

    assert TOPIC_LABELS == DEFAULT_LABELS


# ---------------------------------------------------------------------------
# Gemini client (atrapa SDK - žádný reálný klíč/síťové volání)
# ---------------------------------------------------------------------------

class _FakeGeminiResponse:
    def __init__(self, text, prompt_tokens=100, output_tokens=50):
        self.text = text
        self.usage_metadata = types.SimpleNamespace(
            prompt_token_count=prompt_tokens, candidates_token_count=output_tokens
        )


class _FakeGenerativeModel:
    last_instance = None

    def __init__(self, model_name, system_instruction=None):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.last_call = None
        _FakeGenerativeModel.last_instance = self

    def generate_content(self, prompt, generation_config=None):
        self.last_call = {"prompt": prompt, "generation_config": generation_config}
        import json

        payload = {
            "sentiment": {"label": "positive", "score": 0.77},
            "entities": [],
            "topics": [{"label": "praise and positive feedback", "score": 0.9}],
            "pii": [],
            "pseudonymized_text": prompt,
        }
        return _FakeGeminiResponse(json.dumps(payload))


def _install_fake_genai(monkeypatch):
    fake_genai = types.SimpleNamespace(
        configure=lambda api_key=None: None,
        GenerativeModel=_FakeGenerativeModel,
        GenerationConfig=lambda **kw: kw,
    )
    fake_google = types.SimpleNamespace(generativeai=fake_genai)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    return fake_genai


def test_gemini_client_builds_request_and_parses_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    _install_fake_genai(monkeypatch)

    from llm_client.client import GeminiClient
    from llm_client.config import PROVIDERS

    client = GeminiClient(PROVIDERS["google"])
    payload, meta = client.analyze("Great support, thank you!")

    assert payload["sentiment"]["label"] == "positive"
    assert meta.provider == "google"
    assert meta.tokens_in == 100
    assert meta.tokens_out == 50
    assert meta.cost_usd > 0

    call = _FakeGenerativeModel.last_instance.last_call
    assert "Great support" in call["prompt"]
    assert call["generation_config"]["response_mime_type"] == "application/json"
    assert "response_schema" in call["generation_config"]

    schema = call["generation_config"]["response_schema"]

    def _assert_no_minmax(node):
        if isinstance(node, dict):
            assert "minimum" not in node, "Gemini's Schema proto rejects 'minimum'"
            assert "maximum" not in node, "Gemini's Schema proto rejects 'maximum'"
            for v in node.values():
                _assert_no_minmax(v)
        elif isinstance(node, list):
            for item in node:
                _assert_no_minmax(item)

    _assert_no_minmax(schema)


def test_gemini_schema_strips_minmax_but_other_clients_keep_it():
    from llm_client import prompts

    gemini = prompts.gemini_schema()
    anthropic_tool = prompts.anthropic_tool()

    assert "minimum" not in gemini["properties"]["sentiment"]["properties"]["score"]
    assert (
        anthropic_tool["input_schema"]["properties"]["sentiment"]["properties"]["score"]["minimum"]
        == 0
    )


def test_gemini_missing_api_key_raises_clean_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _install_fake_genai(monkeypatch)

    from llm_client.client import LLMError, get_client

    with pytest.raises(LLMError):
        get_client("google")


def test_provider_registry_includes_all_three():
    from llm_client.client import _REGISTRY

    assert set(_REGISTRY) == {"anthropic", "openai", "google"}


def test_google_gemini_is_the_default_llm_provider(monkeypatch):
    """
    Gemini has a usable free tier, unlike Anthropic/OpenAI - it's the
    fallback provider when a caller doesn't specify an ``engine``/``provider``
    explicitly (e.g. a direct ``LLMPipeline()`` call with no arguments).
    ``Settings.default_provider`` reads ``LLM_PROVIDER`` fresh on every
    ``Settings()`` construction (it's a ``default_factory``, not a value
    fixed at import time), so no module reload is needed here.
    """
    from llm_client.config import Settings

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert Settings().default_provider == "google"


def test_hidden_attribute_is_not_overridden_by_component_display_rules():
    """
    Regression guard: `.dropdownMenu`/`.modelPickers`/`.resultRow` each set
    their own `display` (flex/grid), which - per normal CSS cascade rules -
    outranks the browser's default `[hidden] { display: none }` UA rule
    (author styles always win over UA styles at equal specificity). Without
    a global override, toggling `el.hidden = true` in JS had no visual
    effect on any of them - e.g. the records.html Export dropdown stayed
    permanently open, and the analyze-page LLM engine note/model pickers
    never actually hid/showed. A single `[hidden] { display: none !important
    }` rule (added to static/styles.css) fixes all of them at once.
    """
    from pathlib import Path

    css = (Path(__file__).resolve().parent.parent / "static" / "styles.css").read_text(
        encoding="utf-8"
    )
    assert "[hidden]" in css, "styles.css must restore the [hidden] attribute's effect"

    import re

    # Match the rule at the start of a line (not the explanatory comment
    # above it, which also contains the literal text "[hidden] { ... }").
    match = re.search(r"^\[hidden\]\s*\{([^}]*)\}", css, re.MULTILINE)
    assert match, "[hidden] selector must have a rule block"
    assert "display: none" in match.group(1)
    assert "!important" in match.group(1), (
        "must be !important to win over component display rules like "
        "`.dropdownMenu { display: flex }` regardless of source order"
    )
