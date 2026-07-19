# Class diagram - V4

Modules are shown as facade classes over their public functions. GitHub
renders this Mermaid diagram directly. Compared to V3, `Database.export_xml()`
is generalized into `export_records(fmt, limit)`, dispatching to one exporter
function per format (`xml`, `json`, `csv` - listed in `db.EXPORT_FORMATS`).

```mermaid
classDiagram
    class API {
        <<main.py>>
        +home() FileResponse
        +health() dict
        +list_models() dict
        +analyze(payload) dict
        +predict(payload) dict
        +ingest_file(file, sentiment_model, ner_model, topic_model) dict
        +records(limit) list
        +export_records(format, limit) Response
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
        +process_message(text, topic_labels, sentiment_model, ner_model, topic_model) dict
        +process_batch(messages, topic_labels, sentiment_model, ner_model, topic_model) list
    }

    class NERProcessor {
        <<processors/ner.py>>
        -default_model_name: str = "dslim/bert-base-NER"
        +extract_entities(text, model_id) list
        +is_ready() bool
    }

    class TopicClassifier {
        <<processors/topics.py>>
        -default_model_name: str = "facebook/bart-large-mnli"
        +classify_topic(text, labels, model_id) dict
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
        -cache: dict~"task::model_id", Pipeline~
        -max_cached_pipelines: int = 6
        +get_pipeline(model_id, task, **kwargs) Pipeline
        +cached_models(task) dict
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
        -export_formats: dict = {xml, json, csv}
        +save_record(result, source) void
        +get_records(limit) list
        +stats() dict
        +export_records(fmt, limit) bytes
    }

    API --> IngestService : parses uploaded files
    API --> Pipeline : runs analysis
    API --> Database : reads/writes/export records
    API --> ModelRegistry : lists cached models
    Pipeline --> NERProcessor
    Pipeline --> TopicClassifier
    Pipeline --> SentimentAnalyzer
    Pipeline --> Anonymizer
    NERProcessor --> ModelRegistry : loads non-default HF models
    TopicClassifier --> ModelRegistry : loads non-default HF models
    SentimentAnalyzer --> ModelRegistry : loads non-default HF models
```
