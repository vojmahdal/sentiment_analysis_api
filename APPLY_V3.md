# Jak aplikovat V3 do repozitáře `sentiment_analysis_api`

Tento balíček je kompletní snapshot projektu ve stavu V3 - staví na V2 a
přidává dynamický výběr sentiment modelu z Hugging Face Hubu a XML export
uložených záznamů. Aplikuje se **po** V2 (tj. repozitář by měl už mít commit
a tag `v2.0`).

## Co je nové oproti V2

- `processors/model_registry.py` - loader/cache pro libovolný HF
  `text-classification` model (max 3 modely v paměti, `trust_remote_code`
  nikdy zapnuto).
- `processors/sentiment.py`, `pipeline.py`, `main.py` - `sentiment_model`
  parametr v `/analyze`, `/predict`, `/ingest` (form pole).
- Nový endpoint `GET /models` (výchozí/doporučené/cachované modely).
- Nový endpoint `GET /records/export.xml` a `db.export_xml()` (stdlib
  `xml.etree.ElementTree`, žádná nová závislost).
- `static/index.html` + `app.js` - pole pro zadání modelu (s návrhy z
  `/models`).
- `static/records.html` - tlačítko „Export XML".
- Aktualizovaný `README.md` a `docs/class_diagram.md`.

## Postup

1. Zkopíruj **veškerý obsah** této složky do kořene repozitáře (přepiš
   `main.py`, `db.py`, `pipeline.py`, `README.md`, `docs/class_diagram.md`,
   `static/*`; přidej nový `processors/model_registry.py`).
2. `git status` a commit:

   ```bash
   git add -A
   git commit -m "V3: dynamic Hugging Face model selection for sentiment analysis + XML export of stored records"
   git tag -a v3.0 -m "V3: dynamic HF model selection + XML export"
   ```

3. Push, až budeš chtít:

   ```bash
   git push origin main
   git push origin v3.0
   ```

## Lokální ověření před commitem (doporučeno)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
uvicorn main:app --reload --port 8000
```

Pak vyzkoušej:
- `GET /models` - měl by vrátit výchozí + doporučené modely
- `POST /analyze` s `"sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english"`
  (první request tento model stáhne, může chvíli trvat)
- `POST /analyze` s neplatným `sentiment_model` (např. `"neexistujici/model"`)
  - očekávej HTTP 400 se srozumitelnou chybou, ne pád serveru
- `GET /records/export.xml` - stažení/zobrazení platného XML se záznamy
- na stránce `/static/records.html` tlačítko „Export XML"
