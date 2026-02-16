const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const emptyEl = document.getElementById("empty");
const tableWrapper = document.getElementById("tableWrapper");
const bodyEl = document.getElementById("logBody");
const reloadBtn = document.getElementById("reloadBtn");

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = !msg;
}

function renderRows(items) {
  bodyEl.innerHTML = "";
  for (const item of items) {
    const tr = document.createElement("tr");

    const tdTime = document.createElement("td");
    tdTime.textContent = item.created_at || "";
    tr.appendChild(tdTime);

    const tdText = document.createElement("td");
    tdText.textContent = item.anonymized_text || "";
    tr.appendChild(tdText);

    const tdLabel = document.createElement("td");
    tdLabel.textContent = item.label || "";
    tr.appendChild(tdLabel);

    const tdScore = document.createElement("td");
    tdScore.textContent =
      typeof item.score === "number" ? item.score.toFixed(4) : String(item.score ?? "");
    tr.appendChild(tdScore);

    bodyEl.appendChild(tr);
  }
}

async function loadLogs() {
  showError("");
  setStatus("Loading logs…");

  try {
    const res = await fetch("/logs?limit=100");
    const data = await res.json().catch(() => null);

    if (!res.ok) {
      throw new Error(
        `API error (${res.status} ${res.statusText})` +
          (data?.detail ? `: ${JSON.stringify(data.detail)}` : "")
      );
    }

    if (!Array.isArray(data) || data.length === 0) {
      emptyEl.hidden = false;
      tableWrapper.hidden = true;
      setStatus("No logs yet.");
      return;
    }

    renderRows(data);
    emptyEl.hidden = true;
    tableWrapper.hidden = false;
    setStatus(`Loaded ${data.length} records.`);
  } catch (e) {
    showError(e?.message || String(e));
    setStatus("");
  }
}

reloadBtn.addEventListener("click", loadLogs);

// auto-load on page open
loadLogs();

