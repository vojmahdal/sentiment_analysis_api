"""
Prompty a schéma strukturovaného výstupu.

Volný text z LLM je pro měření nepoužitelný. Výstup je proto vynucen
schématem: u Anthropic přes `tools` + `tool_choice`, u OpenAI přes
`response_format: json_schema`. Model tedy nemůže vrátit nic jiného než
validní objekt s pevně danými poli a povolenými hodnotami.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .config import ENTITY_LABELS, PII_LABELS, SENTIMENT_LABELS, TOPIC_LABELS

TOOL_NAME = "record_analysis"

SYSTEM_PROMPT = """\
Jsi analytický nástroj pro zpracování zákaznických konverzací s podpůrnými \
službami. Analyzuješ jednu zprávu a vracíš strukturovaný výsledek.

Pravidla:
1. Používej výhradně hodnoty z povolených číselníků. Nevymýšlej nové třídy.
2. Sentiment posuzuj z pohledu pisatele zprávy, ne z pohledu operátora.
3. Entity vyznač přesně tak, jak se vyskytují v textu (doslovný podřetězec).
4. Skóre je tvoje míra jistoty v intervalu 0.0 až 1.0.
5. U témat vrať skóre pro všechna témata z číselníku, ne jen pro vítězné.
6. Pseudonymizovaný text vznikne nahrazením každého osobního údaje \
zástupným tokenem ve tvaru <TYP_N>, kde TYP je typ údaje a N je pořadové \
číslo výskytu daného typu (např. <PERSON_1>, <EMAIL_ADDRESS_1>). Stejná \
hodnota vyskytující se opakovaně dostane vždy stejný token.
7. Nepřidávej žádný komentář ani vysvětlení mimo strukturovaný výstup.
"""

USER_TEMPLATE = """\
Analyzuj následující zprávu.

<zprava>
{text}
</zprava>
"""


def build_input_schema() -> Dict[str, Any]:
    """JSON schéma vynuceného výstupu."""
    return {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": SENTIMENT_LABELS},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["label", "score"],
            },
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string", "enum": ENTITY_LABELS},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["text", "label", "score"],
                },
            },
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": TOPIC_LABELS},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["label", "score"],
                },
            },
            "pii": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string", "enum": PII_LABELS},
                        "placeholder": {"type": "string"},
                    },
                    "required": ["text", "label", "placeholder"],
                },
            },
            "pseudonymized_text": {"type": "string"},
        },
        "required": ["sentiment", "entities", "topics", "pii", "pseudonymized_text"],
    }


def anthropic_tool() -> Dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": "Zaznamená výsledek analýzy jedné zprávy.",
        "input_schema": build_input_schema(),
    }


def openai_response_format() -> Dict[str, Any]:
    schema = build_input_schema()
    _strictify(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": TOOL_NAME, "strict": True, "schema": schema},
    }


def _strictify(node: Any) -> None:
    """OpenAI strict mode vyžaduje additionalProperties=false u všech objektů."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node.setdefault("required", list(node.get("properties", {}).keys()))
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


def gemini_schema() -> Dict[str, Any]:
    """
    Schéma pro Gemini `response_schema`. Gemini přijímá jen omezenou
    podmnožinu JSON Schema (Google `Schema` proto) - na rozdíl od
    Anthropic/OpenAI nezná klíče `minimum`/`maximum`, jejich přítomnost ve
    schématu volání rovnou odmítne chybou "Unknown field for Schema:
    minimum". Číselné rozmezí skóre (0.0-1.0) proto hlídá jen instrukce v
    `SYSTEM_PROMPT`, ne schéma samotné.
    """
    schema = build_input_schema()
    _strip_unsupported_gemini_keys(schema)
    return schema


def _strip_unsupported_gemini_keys(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("minimum", None)
        node.pop("maximum", None)
        for value in node.values():
            _strip_unsupported_gemini_keys(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported_gemini_keys(item)


def build_messages(text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": USER_TEMPLATE.format(text=text)}]


def schema_fingerprint() -> str:
    """Otisk schématu – ukládá se k běhu kvůli reprodukovatelnosti."""
    import hashlib

    payload = json.dumps(build_input_schema(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
