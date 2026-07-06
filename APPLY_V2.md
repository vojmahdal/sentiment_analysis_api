# Jak aplikovat V2 do repozitáře `sentiment_analysis_api`

Tento balíček je kompletní snapshot projektu ve stavu V2 (plná extrakční
pipeline: ingest, NER, topic classification, Presidio anonymizace,
dashboard). Postup:

1. Rozbal zip do dočasné složky (nebo pracuj přímo z rozbaleného obsahu).
2. Ve svém repozitáři `sentiment_analysis_api` smaž staré soubory, které V2
   nahrazuje:
   - `static/logs.html`
   - `static/logs.js`
   (nahrazují je `static/records.html` a `static/records.js`)
3. Zkopíruj **veškerý obsah** této složky do kořene repozitáře
   `sentiment_analysis_api` (přepiš `main.py`, `db.py`, `Dockerfile`,
   `README.md`, `requirements.txt`, `static/*`; přidej nové soubory
   `pipeline.py`, `ingest.py`, `processors/`, `evaluate.py`,
   `prepare_dataset.py`, `sample_chats.csv`, `requirements-dev.txt`,
   `docs/class_diagram.md`, `.gitignore`).
4. Starý `sentiment_logs.db` (z V1) můžeš v repu nechat jako historický
   artefakt, nebo smazat - V2 už používá nový soubor `conversation_logs.db`
   (běhový, není součástí tohoto zipu a je v `.gitignore`, takže se
   nebude commitovat).
5. Zkontroluj `git status`, přidej a commitni:

   ```bash
   git add -A
   git commit -m "V2: add ingest pipeline, NER, zero-shot topic classification and Presidio anonymization"
   git tag -a v2.0 -m "V2: full extraction pipeline (ingest, NER, topics, sentiment, anonymization)"
   ```

6. Push, až budeš chtít:

   ```bash
   git push origin main
   git push origin v2.0
   ```

## Lokální ověření před commitem (doporučeno)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
uvicorn main:app --reload --port 8000
```

Pak vyzkoušej:
- `GET /health` - měly by být `ready: true` u ner/topics/sentiment
- `POST /analyze` s nějakým textem obsahujícím jméno/e-mail
- `POST /ingest` se souborem `sample_chats.csv`
- `GET /records`, `GET /stats`
- stránku `/` a `/static/records.html`
