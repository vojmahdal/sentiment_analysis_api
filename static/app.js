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

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
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
