const textEl = document.getElementById("text");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const sentimentEl = document.getElementById("sentiment");
const topicEl = document.getElementById("topic");
const entitiesEl = document.getElementById("entities");
const anonEl = document.getElementById("anon");
const errorEl = document.getElementById("error");

const fileEl = document.getElementById("file");
const ingestBtn = document.getElementById("ingestBtn");
const ingestStatusEl = document.getElementById("ingestStatus");
const ingestErrorEl = document.getElementById("ingestError");
const ingestProgressEl = document.getElementById("ingestProgress");
const ingestProgressBarEl = document.getElementById("ingestProgressBar");
const ingestProgressLabelEl = document.getElementById("ingestProgressLabel");
const ingestTimerEl = document.getElementById("ingestTimer");

// ---------------------------------------------------------------------------
// Model pickers: a <select> dropdown listing the suggested models plus a
// "Custom model…" option. The free-text input for a custom Hugging Face
// repo id is hidden by default and only appears once "Custom model…" is
// selected in the dropdown.
// ---------------------------------------------------------------------------
const modelPickers = {}; // task -> { getValue() }

function buildModelPicker(containerEl, task, defaultModel, suggestedModels) {
  containerEl.innerHTML = "";

  const models = suggestedModels && suggestedModels.length ? suggestedModels : [defaultModel];

  const select = document.createElement("select");
  select.className = "modelSelect";
  select.setAttribute("aria-label", `${task} model`);

  for (const modelId of models) {
    const opt = document.createElement("option");
    opt.value = modelId;
    opt.textContent = modelId;
    select.appendChild(opt);
  }

  const customOpt = document.createElement("option");
  customOpt.value = "__custom__";
  customOpt.textContent = "Custom model (Hugging Face Hub id)…";
  select.appendChild(customOpt);

  if (models.includes(defaultModel)) {
    select.value = defaultModel;
  }

  const customInput = document.createElement("input");
  customInput.type = "text";
  customInput.className = "modelInput";
  customInput.placeholder = "e.g. some-namespace/some-model";
  customInput.hidden = true;

  select.addEventListener("change", () => {
    const isCustom = select.value === "__custom__";
    customInput.hidden = !isCustom;
    if (isCustom) customInput.focus();
  });

  containerEl.appendChild(select);
  containerEl.appendChild(customInput);

  return {
    getValue() {
      if (select.value === "__custom__") {
        return (customInput.value || "").trim() || null;
      }
      return select.value;
    },
  };
}

async function loadModelPickers() {
  const pickerContainers = {
    sentiment: document.getElementById("sentimentModelPicker"),
    ner: document.getElementById("nerModelPicker"),
    topics: document.getElementById("topicModelPicker"),
  };

  let data = null;
  try {
    const res = await fetch("/models");
    if (res.ok) data = await res.json();
  } catch (e) {
    // fall back to an empty picker (default model only) below
  }

  for (const [task, containerEl] of Object.entries(pickerContainers)) {
    if (!containerEl) continue;
    const info = data?.[task] || {};
    modelPickers[task] = buildModelPicker(
      containerEl,
      task,
      info.default || "",
      info.suggested || (info.default ? [info.default] : [])
    );
  }
}

function setStatus(msg) {
  statusEl.textContent = msg || "";
}
function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = !msg;
}

function sentimentClass(label) {
  const l = (label || "").toLowerCase();
  if (l.includes("pos")) return "pill pos";
  if (l.includes("neg")) return "pill neg";
  return "pill neu";
}

function renderResult(data) {
  // sentiment
  const sLabel = data.sentiment ?? "—";
  const sScore =
    typeof data.sentiment_score === "number"
      ? ` (${data.sentiment_score.toFixed(2)})`
      : "";
  sentimentEl.innerHTML = `<span class="${sentimentClass(sLabel)}">${sLabel}${sScore}</span>`;

  // topic
  const tScore =
    typeof data.topic_score === "number" ? ` (${data.topic_score.toFixed(2)})` : "";
  topicEl.textContent = (data.topic ?? "—") + (data.topic ? tScore : "");

  // entities
  entitiesEl.innerHTML = "";
  if (Array.isArray(data.entities) && data.entities.length) {
    for (const ent of data.entities) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerHTML = `<span class="chipType">${ent.type}</span>${ent.text}`;
      entitiesEl.appendChild(chip);
    }
  } else {
    entitiesEl.innerHTML = `<span class="muted">No entities detected</span>`;
  }

  // anonymized text
  anonEl.textContent = data.anonymized_text ?? "";

  resultEl.hidden = false;
}

function resetOutput() {
  resultEl.hidden = true;
  showError("");
  setStatus("");
}

async function analyze() {
  resetOutput();
  const text = (textEl.value || "").trim();
  if (!text) {
    showError("Message cannot be empty.");
    return;
  }

  analyzeBtn.disabled = true;
  setStatus("Analyzing…");

  const sentimentModel = modelPickers.sentiment?.getValue() ?? null;
  const nerModel = modelPickers.ner?.getValue() ?? null;
  const topicModel = modelPickers.topics?.getValue() ?? null;

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        sentiment_model: sentimentModel,
        ner_model: nerModel,
        topic_model: topicModel,
      }),
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const detail = data?.detail ? `\n\n${JSON.stringify(data.detail)}` : "";
      throw new Error(`API error (${res.status} ${res.statusText}).${detail}`);
    }
    renderResult(data);
    setStatus("Done.");
  } catch (e) {
    showError(e?.message || String(e));
    setStatus("");
  } finally {
    analyzeBtn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Batch ingest: /ingest now starts a background job and returns immediately
// (see jobs.py). We poll /ingest/status/{job_id} for progress and drive a
// progress bar + elapsed-time timer while it runs.
// ---------------------------------------------------------------------------
let ingestTimerHandle = null;
let ingestPollHandle = null;

function stopIngestTimers() {
  if (ingestTimerHandle) clearInterval(ingestTimerHandle);
  if (ingestPollHandle) clearInterval(ingestPollHandle);
  ingestTimerHandle = null;
  ingestPollHandle = null;
}

function startIngestTimer(startedAt) {
  ingestTimerEl.textContent = "0.0 s";
  ingestTimerHandle = setInterval(() => {
    const elapsed = (Date.now() - startedAt) / 1000;
    ingestTimerEl.textContent = `${elapsed.toFixed(1)} s`;
  }, 100);
}

function updateIngestProgress(processed, total) {
  ingestProgressBarEl.max = Math.max(total, 1);
  ingestProgressBarEl.value = processed;
  ingestProgressLabelEl.textContent = `Zpracováno ${processed} z ${total}`;
}

function finishIngest(message, isError) {
  stopIngestTimers();
  ingestBtn.disabled = false;
  ingestStatusEl.textContent = isError ? "" : message;
  if (isError) {
    ingestErrorEl.textContent = message;
    ingestErrorEl.hidden = false;
  }
}

async function pollIngestJob(jobId) {
  ingestPollHandle = setInterval(async () => {
    let res, status;
    try {
      res = await fetch(`/ingest/status/${jobId}`);
      status = await res.json();
    } catch (e) {
      finishIngest(`Ztraceno spojení se serverem: ${e?.message || e}`, true);
      return;
    }

    if (!res.ok) {
      finishIngest(status?.detail || `Neznámá úloha (HTTP ${res.status}).`, true);
      return;
    }

    updateIngestProgress(status.processed, status.total);

    if (status.status === "done") {
      const note = status.result.truncated ? " (dávka omezena na 200 zpráv)" : "";
      finishIngest(
        `Zpracováno ${status.result.processed}, uloženo ${status.result.stored}${note}. Viz Records.`,
        false
      );
    } else if (status.status === "error") {
      finishIngest(`Zpracování selhalo: ${status.error}`, true);
    }
  }, 1000);
}

async function ingest() {
  ingestErrorEl.hidden = true;
  const file = fileEl.files?.[0];
  if (!file) {
    ingestErrorEl.textContent = "Please choose a CSV or JSON file first.";
    ingestErrorEl.hidden = false;
    return;
  }

  ingestBtn.disabled = true;
  ingestStatusEl.textContent = "Nahrávání…";
  ingestProgressEl.hidden = false;
  updateIngestProgress(0, 0);

  try {
    const form = new FormData();
    form.append("file", file);
    const sentimentModel = modelPickers.sentiment?.getValue();
    const nerModel = modelPickers.ner?.getValue();
    const topicModel = modelPickers.topics?.getValue();
    if (sentimentModel) form.append("sentiment_model", sentimentModel);
    if (nerModel) form.append("ner_model", nerModel);
    if (topicModel) form.append("topic_model", topicModel);
    const res = await fetch("/ingest", { method: "POST", body: form });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const detail = data?.detail ? `: ${JSON.stringify(data.detail)}` : "";
      throw new Error(`API error (${res.status})${detail}`);
    }

    ingestStatusEl.textContent = "Zpracovávání…";
    updateIngestProgress(0, data.total);
    startIngestTimer(Date.now());
    pollIngestJob(data.job_id);
  } catch (e) {
    ingestProgressEl.hidden = true;
    ingestErrorEl.textContent = e?.message || String(e);
    ingestErrorEl.hidden = false;
    ingestStatusEl.textContent = "";
    ingestBtn.disabled = false;
  }
}

analyzeBtn.addEventListener("click", analyze);
clearBtn.addEventListener("click", () => {
  textEl.value = "";
  resetOutput();
  textEl.focus();
});
textEl.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
});
ingestBtn.addEventListener("click", ingest);

loadModelPickers();
