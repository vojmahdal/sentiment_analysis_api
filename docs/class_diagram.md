# Class diagram - V3

Modules are shown as facade classes over their public functions. GitHub
renders this Mermaid diagram directly. Compared to V2, this adds
`ModelRegistry` (dynamic Hugging Face model loading/caching for sentiment
analysis) and `export_xml()` on `Database`.

```mermaid
classDiagram
    class API {
        <<main.py>>
        +home() FileResponse
        +health() dict
        +list_models() dict
        +analyze(payload) dict
        +predict(payload) dict
        +ingest_file(file, sentiment_model) dict
        +records(limit) list
        +export_records_xml(limit) Response
        +get_stats() dict
    }

    class IngestService {
        <<ingest.py>>
        +parse_csv(raw) list
        +parse_json(raw) list
        +parse_upload(filename, raw) list
    }

    class Pipeline {
        <<pipeline.py>>
        +process_message(text, topic_labels, sentiment_model) dict
        +process_batch(messages, topic_labels, sentiment_model) list
    }

    class NERProcessor {
        <<processors/ner.py>>
        -model_name: str = "dslim/bert-base-NER"
        +extract_entities(text) list
        +is_ready() bool
    }

    class TopicClassifier {
        <<processors/topics.py>>
        -model_name: str = "facebook/bart-large-mnli"
        +classify_topic(text, labels) dict
        +is_ready() bool
    }

    class SentimentAnalyzer {
        <<processors/sentiment.py>>
        -default_model_name: str = "vojmahdal/roberta-sentiment-3labels"
        +analyze_sentiment(text, model_id) dict
        +is_ready() bool
    }

    class ModelRegistry {
        <<processors/model_registry.py>>
        -cache: dict~str, Pipeline~
        -max_cached_models: int = 3
        +get_pipeline(model_id, task) Pipeline
        +cached_models() list
    }

    class Anonymizer {
        <<processors/anonymizer.py>>
        -backend: presidio | regex-fallback
        +anonymize_text(text) str
        +detect_pii(text) list
        +backend_name() str
    }

    class Database {
        <<db.py>>
        +save_record(result, source) void
        +get_records(limit) list
        +stats() dict
        +export_xml(limit) bytes
    }

    API --> IngestService : parses uploaded files
    API --> Pipeline : runs analysis
    API --> Database : reads/writes/export records
    API --> ModelRegistry : lists cached models
    Pipeline --> NERProcessor
    Pipeline --> TopicClassifier
    Pipeline --> SentimentAnalyzer
    Pipeline --> Anonymizer
    SentimentAnalyzer --> ModelRegistry : loads non-default HF models
```
