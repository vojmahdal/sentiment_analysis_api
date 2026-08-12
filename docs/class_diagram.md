# Class diagram - V8

Modules are shown as facade classes over their public functions. GitHub
renders this Mermaid diagram directly. Compared to V3, `Database.export_xml()`
is generalized into `export_records(fmt, limit)`, dispatching to one exporter
function per format (`xml`, `json`, `csv` - listed in `db.EXPORT_FORMATS`).
V5 only changed the dashboard's model picker (dropdown instead of radio
buttons) and the export button (dropdown menu) - no backend classes changed
there. V6 added a batched method to `NERProcessor`, `TopicClassifier` and
`SentimentAnalyzer` (`*_batch`, taking a list of texts), and `Pipeline.
process_batch` calls those directly instead of looping over
`process_message` - see the README section "Batched inference for large
/ingest batches". V7 adds a new `JobTracker` class (`jobs.py`) and makes
`API.ingest_file` start a background thread instead of processing
synchronously - see "Background ingest jobs (progress bar + timer)".

V8 adds a second, independent branch alongside `Pipeline`: the `llm_client`
package, which resolves the same three tasks (sentiment, NER, topic) with a
single call to an LLM provider instead of three local models. `API` gains
`list_engines()` and branches `analyze()`/`ingest_file()` on an `engine`
field between `Pipeline` (local, unchanged) and `LLMPipeline` (new). Both
branches always go through `Anonymizer` first - `LLMPipeline` wraps the
same module via `_load_presidio_anonymizer()` - so pseudonymization happens
locally before any text reaches a provider API. `LLMPipeline` delegates the
actual model call to one of three `LLMClient` subclasses (`AnthropicClient`,
`OpenAIClient`, `GeminiClient`) chosen by `get_client(provider)`, each
returning the same `AnalysisResult` contract regardless of provider. `API`
maps that result onto the same shape `Pipeline.process_message` returns and
persists it via the same `Database.save_record` used by the local engine,
tagged with `engine`/`provider`/`model`/`latency_ms`/`cost_usd` - both
engines' results therefore live in the same `records` table and are listed/
exported together, see the README section "Optional LLM engine (Claude /
Gemini / OpenAI)". A dedicated BERT-vs-LLM comparison layer (its own
database, agreement metrics, dashboard) was considered but dropped from
this version; that comparison is instead run manually and written up in
the thesis.

```mermaid
classDiagram
    class API {
        <<main.py>>
        +home() FileResponse
        +health() dict
        +list_models() dict
        +list_engines() dict
        +analyze(payload) dict
        +predict(payload) dict
        +ingest_file(file, engine, llm_mode, sentiment_model, ner_model, topic_model) dict
        +ingest_status(job_id) dict
        +records(limit, engine, topic, sentiment) list
        +export_records(format, limit, engine, topic, sentiment) Response
        +get_stats() dict
    }

    class IngestService {
        <<ingest.py>>
        +parse_csv(raw) list
        +parse_json(raw) list
        +parse_upload(filename, raw) list
    }

    class JobTracker {
        <<jobs.py>>
        -jobs: dict~job_id, dict~
        +create_job(total) str
        +progress_callback(job_id) Callable
        +finish_job(job_id, result) void
        +fail_job(job_id, error) void
        +get_job(job_id) dict
    }

    class Pipeline {
        <<pipeline.py>>
        +process_message(text, topic_labels, sentiment_model, ner_model, topic_model) dict
        +process_batch(messages, topic_labels, sentiment_model, ner_model, topic_model, on_progress) list
    }

    class NERProcessor {
        <<processors/ner.py>>
        -default_model_name: str = "dslim/bert-base-NER"
        +extract_entities(text, model_id) list
        +extract_entities_batch(texts, model_id) list~list~
        +is_ready() bool
    }

    class TopicClassifier {
        <<processors/topics.py>>
        -default_model_name: str = "facebook/bart-large-mnli"
        +classify_topic(text, labels, model_id) dict
        +classify_topic_batch(texts, labels, model_id) list~dict~
        +is_ready() bool
    }

    class SentimentAnalyzer {
        <<processors/sentiment.py>>
        -default_model_name: str = "vojmahdal/roberta-sentiment-3labels"
        +analyze_sentiment(text, model_id) dict
        +analyze_sentiment_batch(texts, model_id) list~dict~
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
        +get_records(limit, engine, topic, sentiment) list
        +stats() dict
        +export_records(fmt, limit, engine, topic, sentiment) bytes
    }
    note for Database "records columns since V8:\nengine, provider, model,\nlatency_ms, cost_usd\n(NULL for the local engine)\nget_records/export_records share a\n_build_filter() helper: optional\nengine ('local'|'llm'), topic and\nsentiment filters, combinable"

    class LLMPipeline {
        <<llm_client/pipeline.py>>
        -provider: str
        -mode: "pre_pseudonymized" | "raw"
        +analyze(text) AnalysisResult
        +analyze_batch(texts, on_progress) list~AnalysisResult~
    }

    class LLMClient {
        <<llm_client/client.py>>
        <<abstract>>
        +analyze(text) tuple~dict, CallMeta~
    }

    class AnthropicClient {
        <<llm_client/client.py>>
    }

    class OpenAIClient {
        <<llm_client/client.py>>
    }

    class GeminiClient {
        <<llm_client/client.py>>
    }

    API --> IngestService : parses uploaded files
    API --> Pipeline : runs local analysis (in a background thread)
    API --> LLMPipeline : runs LLM analysis (engine != local)
    API --> JobTracker : creates/polls ingest jobs
    API --> Database : reads/writes/export records
    API --> ModelRegistry : lists cached models
    Pipeline --> JobTracker : reports progress via on_progress
    Pipeline --> NERProcessor
    Pipeline --> TopicClassifier
    Pipeline --> SentimentAnalyzer
    Pipeline --> Anonymizer
    NERProcessor --> ModelRegistry : loads non-default HF models
    TopicClassifier --> ModelRegistry : loads non-default HF models
    SentimentAnalyzer --> ModelRegistry : loads non-default HF models
    LLMPipeline --> Anonymizer : pseudonymizes before any provider call
    LLMPipeline --> LLMClient : delegates the analysis call
    LLMClient <|-- AnthropicClient
    LLMClient <|-- OpenAIClient
    LLMClient <|-- GeminiClient
    LLMPipeline --> JobTracker : reports progress via on_progress (ingest)
    API --> Database : maps LLMPipeline result onto save_record too
```
