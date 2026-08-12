"""
Konfigurace LLM větve systému.

Veškeré citlivé hodnoty (API klíče) se načítají výhradně z proměnných
prostředí – nikdy nejsou uloženy v repozitáři ani v databázi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------
# Sjednocené taxonomie
#
# Aby bylo srovnání BERT vs. LLM metodicky korektní, musí oba enginy
# produkovat výstupy nad *identickou* množinou tříd. Taxonomie je proto
# definována centrálně a do promptu LLM se vkládá dynamicky.
# --------------------------------------------------------------------------

SENTIMENT_LABELS: List[str] = ["negative", "neutral", "positive"]

# Odpovídá výstupu modelu dslim/bert-base-NER (CoNLL-2003)
ENTITY_LABELS: List[str] = ["PER", "ORG", "LOC", "MISC"]

# Převzato přímo z processors.topics.DEFAULT_LABELS (zero-shot klasifikátor
# BERT větve), aby LLM klasifikoval do přesně stejné množiny témat - jinak
# by srovnání měřilo rozdíl v číselnících, ne rozdíl v modelech.
try:
    from processors.topics import DEFAULT_LABELS as TOPIC_LABELS  # type: ignore
except Exception:  # pragma: no cover - fallback when imported outside the app
    TOPIC_LABELS: List[str] = [
        "billing and payments",
        "technical issue",
        "complaint",
        "product question",
        "account management",
        "cancellation",
        "delivery and shipping",
        "praise and positive feedback",
    ]

# Typy PII sjednocené s Microsoft Presidio
PII_LABELS: List[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "CREDIT_CARD",
    "IBAN_CODE",
    "DATE_TIME",
    "ORGANIZATION",
    "ID_NUMBER",
]


@dataclass
class ProviderConfig:
    """Popis jednoho poskytovatele LLM."""

    name: str
    model: str
    api_key_env: str
    # cena v USD za 1 milion tokenů (vstup, výstup)
    price_in_per_mtok: float
    price_out_per_mtok: float
    max_tokens: int = 2048
    temperature: float = 0.0

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)

    @property
    def available(self) -> bool:
        return bool(self.api_key)


# Ceny je nutné před finálním měřením ověřit v aktuálním ceníku poskytovatele;
# jsou zde uvedeny jako výchozí hodnoty pro výpočet nákladovosti.
PROVIDERS: Dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        api_key_env="ANTHROPIC_API_KEY",
        price_in_per_mtok=float(os.getenv("ANTHROPIC_PRICE_IN", "3.00")),
        price_out_per_mtok=float(os.getenv("ANTHROPIC_PRICE_OUT", "15.00")),
    ),
    "openai": ProviderConfig(
        name="openai",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key_env="OPENAI_API_KEY",
        price_in_per_mtok=float(os.getenv("OPENAI_PRICE_IN", "0.15")),
        price_out_per_mtok=float(os.getenv("OPENAI_PRICE_OUT", "0.60")),
    ),
    "google": ProviderConfig(
        name="google",
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        api_key_env="GEMINI_API_KEY",
        price_in_per_mtok=float(os.getenv("GEMINI_PRICE_IN", "0.30")),
        price_out_per_mtok=float(os.getenv("GEMINI_PRICE_OUT", "2.50")),
    ),
}


@dataclass
class Settings:
    default_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "google")
    )
    # 'pre_pseudonymized' = text je před odesláním do API zbaven PII lokálně
    # 'raw'               = do API jde originální text (pouze pro srovnávací experiment)
    default_mode: str = field(default_factory=lambda: os.getenv("LLM_MODE", "pre_pseudonymized"))
    max_retries: int = 3
    request_timeout_s: int = 60
    # maximální počet souběžných požadavků při dávkovém zpracování
    concurrency: int = int(os.getenv("LLM_CONCURRENCY", "4"))
    # tvrdá pojistka: bez explicitního souhlasu nelze poslat surový text ven
    allow_raw_mode: bool = os.getenv("LLM_ALLOW_RAW", "false").lower() == "true"


SETTINGS = Settings()
