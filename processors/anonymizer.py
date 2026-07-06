"""
Anonymization / pseudonymization of personal data (PII).

Primary strategy: Microsoft Presidio (NER via spaCy + regex recognizers + context).
Fallback strategy: regular expressions only (used if Presidio/spaCy are unavailable).

Per GDPR terminology this is *pseudonymization*: identifiers are replaced by
placeholders and the original text is not stored in readable form (only a hash
is kept for deduplication). The data therefore remains personal data and must be
handled accordingly.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Try to load Presidio. If it is not installed (or the spaCy model is missing),
# the module gracefully degrades to a regex-only anonymizer so the system never
# crashes on startup.
# ---------------------------------------------------------------------------
_PRESIDIO_AVAILABLE = False
_analyzer = None
_anonymizer = None

# spaCy models to try, from most to least accurate. The Docker image installs
# en_core_web_lg; the smaller ones are accepted as fallbacks.
_SPACY_MODELS = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]


def _find_spacy_model() -> str | None:
    try:
        import spacy
    except Exception:
        return None
    for name in _SPACY_MODELS:
        try:
            spacy.load(name)
            return name
        except Exception:
            continue
    return None


try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    _model = _find_spacy_model()
    if _model is None:
        raise RuntimeError("no spaCy model available")

    _provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": _model}],
        }
    )
    _analyzer = AnalyzerEngine(nlp_engine=_provider.create_engine())
    _anonymizer = AnonymizerEngine()
    _PRESIDIO_AVAILABLE = True
    print(f"[anonymizer] Presidio loaded (spaCy model: {_model}).")
except Exception as e:  # pragma: no cover - depends on runtime environment
    print(f"[anonymizer] Presidio unavailable, falling back to regex only: {e}")


# ---------------------------------------------------------------------------
# Regex fallback (also used to enrich Presidio for strictly formatted IDs)
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

# Map Presidio entity types -> placeholder labels we expose to users.
_PLACEHOLDERS = {
    "PERSON": "[NAME]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "LOCATION": "[LOCATION]",
    "CREDIT_CARD": "[CARD]",
    "IBAN_CODE": "[IBAN]",
    "IP_ADDRESS": "[IP]",
    "URL": "[URL]",
    "DATE_TIME": "[DATE]",
    "NRP": "[ID]",
    "US_SSN": "[ID]",
}


def _regex_anonymize(text: str) -> str:
    """Pure-regex anonymization for the formatted identifiers we can match safely."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _URL_RE.sub("[URL]", text)
    text = _CREDIT_CARD_RE.sub("[CARD]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text


def anonymize_text(text: str, language: str = "en") -> str:
    """
    Return a pseudonymized version of ``text``.

    With Presidio: NER-based detection of names/locations + built-in regex
    recognizers for emails, phones, cards, IBANs, etc.
    Without Presidio: regex-only fallback (emails, phones, URLs, cards).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if not _PRESIDIO_AVAILABLE:
        return _regex_anonymize(text)

    try:
        results = _analyzer.analyze(text=text, language=language)
        operators = {
            entity: OperatorConfig("replace", {"new_value": placeholder})
            for entity, placeholder in _PLACEHOLDERS.items()
        }
        # default operator for anything detected but not explicitly mapped
        operators["DEFAULT"] = OperatorConfig("replace", {"new_value": "[REDACTED]"})

        anonymized = _anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        return anonymized.text
    except Exception as e:  # pragma: no cover
        print(f"[anonymizer] Presidio failed at runtime, using regex fallback: {e}")
        return _regex_anonymize(text)


def detect_pii(text: str, language: str = "en") -> list[dict[str, Any]]:
    """
    Return the list of detected PII spans (for evaluation / debugging).
    Empty list if Presidio is unavailable.
    """
    if not _PRESIDIO_AVAILABLE or not isinstance(text, str) or not text.strip():
        return []
    try:
        results = _analyzer.analyze(text=text, language=language)
        return [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": float(r.score),
                "text": text[r.start : r.end],
            }
            for r in results
        ]
    except Exception:
        return []


def backend_name() -> str:
    """Report which anonymization backend is active (for /health and the UI)."""
    return "presidio" if _PRESIDIO_AVAILABLE else "regex-fallback"
