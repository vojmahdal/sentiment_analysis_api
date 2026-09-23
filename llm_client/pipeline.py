"""
LLM pipeline – protějšek BERT pipeline.

Obě třídy sdílejí metodu `analyze(text) -> AnalysisResult`, takže je lze
v systému zaměnit bez zásahu do volajícího kódu.

Režimy zpracování (parametr `mode`):

* ``pre_pseudonymized`` – výchozí. Text je nejprve lokálně zbaven osobních
  údajů (Presidio) a teprve poté odeslán do API poskytovatele. Ven z
  infrastruktury tak neopouštějí osobní údaje a odpadá nutnost řešit
  poskytovatele jako zpracovatele osobních údajů podle čl. 28 GDPR.
* ``raw`` – originální text jde přímo do API a pseudonymizaci provádí model.
  Slouží výhradně k experimentálnímu porovnání úspěšnosti detekce PII.
  Vyžaduje explicitní povolení proměnnou ``LLM_ALLOW_RAW=true``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

from .client import LLMClient, call_with_retry, get_client
from .config import SETTINGS
from .schemas import AnalysisResult, Entity, PIISpan, Sentiment, TopicScore

log = logging.getLogger(__name__)


def _coerce_topic(item: Any) -> Dict[str, Any]:
    """
    Schéma vynucuje objekt {label, score}, ale ani vynucené schéma není u LLM
    100% záruka - slabší modely občas vrátí pole holých řetězců místo objektů.
    Takový záznam se nezahazuje (ztráta informace), ale doplní o výchozí
    skóre, aby TopicScore(**item) nespadlo na "argument after ** must be a
    mapping, not str".
    """
    if isinstance(item, str):
        return {"label": item, "score": 1.0}
    return item


# --------------------------------------------------------------------------
# Lokální pseudonymizace před odesláním
# --------------------------------------------------------------------------

def _load_presidio_anonymizer() -> Optional[Callable[[str], tuple[str, list]]]:
    """
    Zkusí připojit pseudonymizér z BERT větve systému (`processors.anonymizer`).
    Ten nemá jedinou funkci `pseudonymize()` s obojím výstupem - má
    `anonymize_text(text) -> str` a `detect_pii(text) -> list[dict]` zvlášť
    ({"type","start","end","score","text"}). Zde je spojíme do stejného
    tvaru `(masked_text, spans)`, jaký LLM pipeline očekává, aby šlo použít
    přesně tu anonymizaci (Presidio + regex fallback), kterou používá i
    hlavní BERT větev - žádná druhá, oddělená implementace stejné věci.
    Pokud modul není dostupný, vrátí None a použije se záložní regex níže.
    """
    try:
        from processors.anonymizer import anonymize_text, detect_pii  # type: ignore
    except Exception:  # noqa: BLE001
        log.warning("processors.anonymizer není dostupný, používá se záložní regex.")
        return None

    def pseudonymize(text: str) -> tuple[str, list]:
        masked = anonymize_text(text)
        spans = [
            SimpleNamespace(
                text=item.get("text", ""),
                entity_type=item.get("type", "ID_NUMBER"),
                start=item.get("start"),
                end=item.get("end"),
                placeholder=None,
            )
            for item in detect_pii(text)
        ]
        return masked, spans

    return pseudonymize


_FALLBACK_PATTERNS = [
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE_NUMBER", re.compile(r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){9,12}(?!\d)")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
]


def _fallback_pseudonymize(text: str) -> tuple[str, List[PIISpan]]:
    spans: List[PIISpan] = []
    counters: dict[str, int] = {}
    out = text
    for label, pattern in _FALLBACK_PATTERNS:
        for match in list(pattern.finditer(out)):
            counters[label] = counters.get(label, 0) + 1
            placeholder = f"<{label}_{counters[label]}>"
            spans.append(
                PIISpan(text=match.group(), label=label, placeholder=placeholder)
            )
            out = out.replace(match.group(), placeholder, 1)
    return out, spans


class LLMPipeline:
    """Analýza zprávy pomocí generativního modelu přes API."""

    engine = "llm"

    def __init__(
        self,
        provider: str | None = None,
        mode: str | None = None,
        client: LLMClient | None = None,
    ):
        self.provider = provider or SETTINGS.default_provider
        self.mode = mode or SETTINGS.default_mode
        if self.mode == "raw" and not SETTINGS.allow_raw_mode:
            raise ValueError(
                "The 'raw' mode sends unmasked text outside the system. "
                "Enable it deliberately via LLM_ALLOW_RAW=true."
            )
        self._client = client or get_client(self.provider)
        self._presidio = _load_presidio_anonymizer()

    # ------------------------------------------------------------------
    def _prepare(self, text: str) -> tuple[str, List[PIISpan]]:
        if self.mode != "pre_pseudonymized":
            return text, []
        if self._presidio is not None:
            masked, raw_spans = self._presidio(text)
            spans = [
                PIISpan(
                    text=getattr(s, "text", ""),
                    label=getattr(s, "entity_type", "ID_NUMBER"),
                    start=getattr(s, "start", None),
                    end=getattr(s, "end", None),
                    placeholder=getattr(s, "placeholder", None),
                )
                for s in raw_spans
            ]
            return masked, spans
        return _fallback_pseudonymize(text)

    # ------------------------------------------------------------------
    def analyze(self, text: str) -> AnalysisResult:
        sent_text, local_spans = self._prepare(text)

        result = AnalysisResult(
            engine=self.engine,
            provider=self.provider,
            mode=self.mode,
        )
        try:
            payload, meta = call_with_retry(self._client, sent_text)
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            log.error("Analýza selhala: %s", exc)
            return result

        result.model = meta.model
        result.provider = meta.provider
        result.latency_ms = meta.latency_ms
        result.tokens_in = meta.tokens_in
        result.tokens_out = meta.tokens_out
        result.cost_usd = meta.cost_usd
        result.raw_response = payload

        # I s vynuceným schématem se občas stane, že model (typicky u
        # levnějších/menších modelů) vrátí tvar mírně odlišný od schématu -
        # zpracování takové odpovědi nesmí spadnout na neošetřené výjimce,
        # ale skončit jako čistá chyba stejně jako selhání samotného volání.
        try:
            result.sentiment = Sentiment(**payload.get("sentiment", {}))
            result.entities = [Entity(**e) for e in payload.get("entities", [])]
            result.topics = [TopicScore(**_coerce_topic(t)) for t in payload.get("topics", [])]

            model_spans = [PIISpan(**p) for p in payload.get("pii", [])]
            # V režimu pre_pseudonymized je autoritativní lokální detekce;
            # nálezy modelu se přidávají jako doplněk (reziduální PII).
            result.pii = local_spans + model_spans if local_spans else model_spans
            result.pseudonymized_text = payload.get("pseudonymized_text") or sent_text
        except (TypeError, ValueError) as exc:
            result.error = f"Model returned malformed structured output: {exc}"
            log.error("Zpracování odpovědi selhalo: %s", exc)
            return result

        # Doplnění offsetů, které model nevrací spolehlivě.
        _fill_offsets(result.entities, sent_text)
        return result

    # ------------------------------------------------------------------
    def analyze_batch(
        self,
        texts: Iterable[str],
        concurrency: int | None = None,
        on_progress: Any = None,
    ) -> List[AnalysisResult]:
        """
        Dávkové zpracování s omezenou souběžností (kvůli rate limitům).

        ``on_progress(done, total)`` je zavolán po dokončení každé zprávy,
        stejná konvence jako ``pipeline.process_batch(on_progress=...)`` u
        lokální větve - umožňuje hlásit průběh přes ``jobs.py`` i pro dávkové
        zpracování LLM enginem.
        """
        items = list(texts)
        workers = concurrency or SETTINGS.concurrency
        results: List[AnalysisResult] = [AnalysisResult()] * len(items)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.analyze, t): i for i, t in enumerate(items)}
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if on_progress:
                    on_progress(done, len(items))
        return results


def _fill_offsets(entities: List[Entity], text: str) -> None:
    cursor = 0
    for entity in entities:
        if entity.start is not None and entity.end is not None:
            continue
        idx = text.find(entity.text, cursor)
        if idx == -1:
            idx = text.find(entity.text)
        if idx != -1:
            entity.start, entity.end = idx, idx + len(entity.text)
            cursor = entity.end
