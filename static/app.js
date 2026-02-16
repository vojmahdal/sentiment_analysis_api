const textEl = document.getElementById("text");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const labelEl = document.getElementById("label");
const scoreEl = document.getElementById("score");
const errorEl = document.getElementById("error");

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = !msg;
}

function showResult({ label, score }) {
  labelEl.textContent = label ?? "";
  scoreEl.textContent =
    typeof score === "number" ? score.toFixed(4) : String(score ?? "");
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
    showError("Text cannot be empty.");
    return;
  }

  analyzeBtn.disabled = true;
  setStatus("Analyzing…");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok) {
      const detail = data?.detail ? `\n\n${JSON.stringify(data.detail)}` : "";
      throw new Error(`Error API (${res.status} ${res.statusText}).${detail}`);
    }

    showResult(data);
    setStatus("Done.");
  } catch (e) {
    showError(e?.message || String(e));
    setStatus("");
  } finally {
    analyzeBtn.disabled = false;
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
