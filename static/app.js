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

// ---------------------------------------------------------------------------
// Model pickers: one radio button per suggested model, plus a "Custom model"
// radio that reveals a free-text input for any other Hugging Face repo id.
// ---------------------------------------------------------------------------
const modelPickers = {}; // task -> { getValue() }

function buildModelPicker(containerEl, task, defaultModel, suggestedModels) {
  containerEl.innerHTML = "";
  const groupName = `${task}ModelChoice`;

  const models = suggestedModels && suggestedModels.length ? suggestedModels : [defaultModel];

  models.forEach((modelId, i) => {
    const optionRow = document.createElement("label");
    optionRow.className = "modelOption";

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = groupName;
    radio.value = modelId;
    if (modelId === defaultModel || (i === 0 && !models.includes(defaultModel))) {
      radio.checked = true;
    }

    optionRow.appendChild(radio);
    optionRow.appendChild(document.createTextNode(" " + modelId));
    containerEl.appendChild(optionRow);
  });

  const customRow = document.createElement("label");
  customRow.className = "modelOption";
  const customRadio = document.createElement("input");
  customRadio.type = "radio";
  customRadio.name = groupName;
  customRadio.value = "__custom__";
  customRow.appendChild(customRadio);
  customRow.appendChild(document.createTextNode(" Custom model (Hugging Face Hub id):"));
  containerEl.appendChild(customRow);

  const customInput = document.createElement("input");
  customInput.type = "text";
  customInput.className = "modelInput";
  customInput.placeholder = "e.g. some-namespace/some-model";
  customInput.disabled = true;
  containerEl.appendChild(customInput);

  containerEl.addEventListener("change", (e) => {
    if (e.target.name !== groupName) return;
    customInput.disabled = e.target.value !== "__custom__";
    if (!customInput.disabled) customInput.focus();
  });

  return {
    getValue() {
      const checked = containerEl.querySelector(`input[name="${groupName}"]:checked`);
      if (!checked) return null;
      if (checked.value === "__custom__") {
        return (customInput.value || "").trim() || null;
      }
      return checked.value;
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

async function ingest() {
  ingestErrorEl.hidden = true;
  const file = fileEl.files?.[0];
  if (!file) {
    ingestErrorEl.textContent = "Please choose a CSV or JSON file first.";
    ingestErrorEl.hidden = false;
    return;
  }

  ingestBtn.disabled = true;
  ingestStatusEl.textContent = "Uploading and processing…";

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
    const note = data.truncated ? " (batch truncated to 200)" : "";
    ingestStatusEl.textContent = `Processed ${data.processed}, stored ${data.stored}${note}. See Records.`;
  } catch (e) {
    ingestErrorEl.textContent = e?.message || String(e);
    ingestErrorEl.hidden = false;
    ingestStatusEl.textContent = "";
  } finally {
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
