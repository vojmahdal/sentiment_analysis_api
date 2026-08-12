"""
Datový kontrakt sdílený oběma enginy.

Klíčová myšlenka: BERT pipeline i LLM pipeline vracejí *stejnou* strukturu
`AnalysisResult`. Díky tomu je porovnání otázkou porovnání dvou instancí
téhož typu, nikoli ad-hoc mapování mezi nekompatibilními výstupy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .config import ENTITY_LABELS, PII_LABELS, SENTIMENT_LABELS, TOPIC_LABELS


class Entity(BaseModel):
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[float] = None

    @field_validator("label")
    @classmethod
    def _norm_label(cls, v: str) -> str:
        v = v.strip().upper()
        alias = {
            "PERSON": "PER",
            "PEOPLE": "PER",
            "ORGANIZATION": "ORG",
            "ORGANISATION": "ORG",
            "LOCATION": "LOC",
            "GPE": "LOC",
            "PLACE": "LOC",
        }
        v = alias.get(v, v)
        return v if v in ENTITY_LABELS else "MISC"


class TopicScore(BaseModel):
    label: str
    score: float = 0.0

    @field_validator("label")
    @classmethod
    def _norm(cls, v: str) -> str:
        v = v.strip().lower()
        return v if v in TOPIC_LABELS else "other"


class PIISpan(BaseModel):
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None
    placeholder: Optional[str] = None

    @field_validator("label")
    @classmethod
    def _norm(cls, v: str) -> str:
        v = v.strip().upper().replace(" ", "_")
        alias = {"EMAIL": "EMAIL_ADDRESS", "PHONE": "PHONE_NUMBER", "NAME": "PERSON"}
        v = alias.get(v, v)
        return v if v in PII_LABELS else "ID_NUMBER"


class Sentiment(BaseModel):
    label: str = "neutral"
    score: float = 0.0

    @field_validator("label")
    @classmethod
    def _norm(cls, v: str) -> str:
        v = v.strip().lower()
        alias = {
            "pos": "positive",
            "neg": "negative",
            "neu": "neutral",
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
        }
        v = alias.get(v, v)
        return v if v in SENTIMENT_LABELS else "neutral"


class AnalysisResult(BaseModel):
    """Jednotný výstup jednoho enginu nad jednou zprávou."""

    engine: str = "unknown"          # 'bert' | 'llm'
    provider: str = "local"          # 'local' | 'anthropic' | 'openai'
    model: str = ""
    mode: str = "pre_pseudonymized"

    sentiment: Sentiment = Field(default_factory=Sentiment)
    entities: List[Entity] = Field(default_factory=list)
    topics: List[TopicScore] = Field(default_factory=list)
    pii: List[PIISpan] = Field(default_factory=list)
    pseudonymized_text: str = ""

    # provozní metriky
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def top_topic(self) -> str:
        if not self.topics:
            return "other"
        return max(self.topics, key=lambda t: t.score).label
