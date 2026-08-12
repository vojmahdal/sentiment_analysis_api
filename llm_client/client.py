"""
Klienti jednotlivých poskytovatelů LLM.

Abstrakce `LLMClient` skrývá rozdíly mezi Anthropic Messages API a OpenAI
Chat Completions API. Volající kód pracuje pouze s metodou `analyze()`,
která vrací dvojici (data, metadata) – data odpovídají schématu z modulu
`prompts`, metadata obsahují latenci, tokeny a náklad.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from . import prompts
from .config import PROVIDERS, SETTINGS, ProviderConfig


class LLMError(RuntimeError):
    pass


@dataclass
class CallMeta:
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str
    provider: str


class LLMClient:
    """Základní rozhraní."""

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        if not cfg.available:
            raise LLMError(
                f"Missing API key. Set the environment variable {cfg.api_key_env}."
            )

    def analyze(self, text: str) -> Tuple[Dict[str, Any], CallMeta]:
        raise NotImplementedError

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in / 1_000_000 * self.cfg.price_in_per_mtok
            + tokens_out / 1_000_000 * self.cfg.price_out_per_mtok
        )


class AnthropicClient(LLMClient):
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("Install the 'anthropic' package (pip install anthropic).") from exc
        self._sdk = anthropic.Anthropic(
            api_key=cfg.api_key, timeout=SETTINGS.request_timeout_s
        )

    def analyze(self, text: str) -> Tuple[Dict[str, Any], CallMeta]:
        started = time.perf_counter()
        response = self._sdk.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=prompts.SYSTEM_PROMPT,
            tools=[prompts.anthropic_tool()],
            tool_choice={"type": "tool", "name": prompts.TOOL_NAME},
            messages=prompts.build_messages(text),
        )
        latency = int((time.perf_counter() - started) * 1000)

        payload: Dict[str, Any] | None = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                payload = block.input
                break
        if payload is None:
            raise LLMError("Model did not return structured output (missing tool_use block).")

        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        meta = CallMeta(
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost(tokens_in, tokens_out),
            model=self.cfg.model,
            provider="anthropic",
        )
        return payload, meta


class OpenAIClient(LLMClient):
    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("Install the 'openai' package (pip install openai).") from exc
        self._sdk = OpenAI(api_key=cfg.api_key, timeout=SETTINGS.request_timeout_s)

    def analyze(self, text: str) -> Tuple[Dict[str, Any], CallMeta]:
        started = time.perf_counter()
        response = self._sdk.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            response_format=prompts.openai_response_format(),
            messages=[{"role": "system", "content": prompts.SYSTEM_PROMPT}]
            + prompts.build_messages(text),
        )
        latency = int((time.perf_counter() - started) * 1000)

        content = response.choices[0].message.content
        if not content:
            raise LLMError("Model returned an empty response.")
        payload = json.loads(content)

        usage = response.usage
        meta = CallMeta(
            latency_ms=latency,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            cost_usd=self._cost(usage.prompt_tokens, usage.completion_tokens),
            model=self.cfg.model,
            provider="openai",
        )
        return payload, meta


class GeminiClient(LLMClient):
    """
    Google Gemini (`google-generativeai`). Strukturovaný výstup je vynucen
    přes `response_mime_type="application/json"` + `response_schema` -
    schéma vychází ze stejného `prompts.build_input_schema()` jako u
    Anthropic/OpenAI, ale přes `prompts.gemini_schema()`, který navíc
    odstraní klíče `minimum`/`maximum` - Gemini je ve svém `Schema` proto
    nezná a s nimi by volání rovnou odmítlo.
    """

    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "Install the 'google-generativeai' package (pip install google-generativeai)."
            ) from exc
        genai.configure(api_key=cfg.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(
            cfg.model, system_instruction=prompts.SYSTEM_PROMPT
        )

    def analyze(self, text: str) -> Tuple[Dict[str, Any], CallMeta]:
        started = time.perf_counter()
        response = self._model.generate_content(
            prompts.USER_TEMPLATE.format(text=text),
            generation_config=self._genai.GenerationConfig(
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_tokens,
                response_mime_type="application/json",
                response_schema=prompts.gemini_schema(),
            ),
        )
        latency = int((time.perf_counter() - started) * 1000)

        content = response.text
        if not content:
            raise LLMError("Model returned no content.")
        payload = json.loads(content)

        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0
        meta = CallMeta(
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost(tokens_in, tokens_out),
            model=self.cfg.model,
            provider="google",
        )
        return payload, meta


_REGISTRY = {"anthropic": AnthropicClient, "openai": OpenAIClient, "google": GeminiClient}


def get_client(provider: str | None = None) -> LLMClient:
    provider = provider or SETTINGS.default_provider
    if provider not in _REGISTRY:
        raise LLMError(
            f"Unknown provider '{provider}'. Available: {', '.join(_REGISTRY)}."
        )
    return _REGISTRY[provider](PROVIDERS[provider])


def call_with_retry(client: LLMClient, text: str) -> Tuple[Dict[str, Any], CallMeta]:
    """Volání s exponenciálním odstupem – ošetřuje rate limit a výpadky sítě."""
    last: Exception | None = None
    for attempt in range(SETTINGS.max_retries):
        try:
            return client.analyze(text)
        except Exception as exc:  # noqa: BLE001 – záměrně široké, chyba se loguje
            last = exc
            if attempt == SETTINGS.max_retries - 1:
                break
            time.sleep(2**attempt)
    raise LLMError(f"Call failed after {SETTINGS.max_retries} attempts: {last}")
