"""
llm_client – druhá větev systému pro analýzu konverzací.

Zatímco původní větev provádí NER, klasifikaci témat, analýzu sentimentu
a pseudonymizaci lokálními modely typu BERT, tato větev řeší tytéž úlohy
jediným voláním generativního modelu přes API - vždy až nad textem, který
už prošel stejnou lokální pseudonymizací jako BERT větev. Výsledky obou
větví se ukládají do stejné databáze (`db.py` hlavní aplikace) a zobrazují
ve stejném seznamu záznamů.
"""

from .config import SETTINGS, PROVIDERS
from .pipeline import LLMPipeline
from .schemas import AnalysisResult

__version__ = "1.0.0"

__all__ = [
    "SETTINGS",
    "PROVIDERS",
    "LLMPipeline",
    "AnalysisResult",
]
