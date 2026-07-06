const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const emptyEl = document.getElementById("empty");
const tableWrapper = document.getElementById("tableWrapper");
const bodyEl = document.getElementById("logBody");
const reloadBtn = document.getElementById("reloadBtn");
const statsEl = document.getElementById("stats");

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

function renderStats(stats) {
  if (!stats || !stats.total) {
    statsEl.innerHTML = `<span class="muted">No statistics yet.</span>`;
    return;
  }
  const sentiments = Object.entries(stats.by_sentiment || {})
    .map(([k, v]) => `<span class="${sentimentClass(k)}">${k}: ${v}</span>`)
    .join(" ");
  const topics = Object.entries(stats.by_topic || {})
    .map(([k, v]) => `<span class="chip">${k}: ${v}</span>`)
    .join(" ");
  statsEl.innerHTML = `
    <div class="statsRow"><span class="k">Total records</span><span class="v">${stats.total}</span></div>
    <div class="statsRow column"><span class="k">By sentiment</span><div class="chips">${sentiments}</div></div>
    <div class="statsRow column"><span class="k">Top topics</span><div class="chips">${topics}</div></div>
  `;
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

    const tdTopic = document.createElement("td");
    tdTopic.textContent = item.topic || "—";
    tr.appendChild(tdTopic);

    const tdSent = document.createElement("td");
    const sLabel = item.sentiment || "—";
    tdSent.innerHTML = `<span class="${sentimentClass(sLabel)}">${sLabel}</span>`;
    tr.appendChild(tdSent);

    const tdEnt = document.createElement("td");
    const ents = Array.isArray(item.entities) ? item.entities : [];
    tdEnt.textContent = ents.length
      ? ents.map((e) => `${e.type}:${e.text}`).join(", ")
      : "—";
    tr.appendChild(tdEnt);

    const tdSrc = document.createElement("td");
    tdSrc.textContent = item.source || "";
    tr.appendChild(tdSrc);

    bodyEl.appendChild(tr);
  }
}

async function load() {
  showError("");
  setStatus("Loading…");
  try {
    const [recRes, statRes] = await Promise.all([
      fetch("/records?limit=100"),
      fetch("/stats"),
    ]);
    const records = await recRes.json().catch(() => null);
    const stats = await statRes.json().catch(() => null);

    if (!recRes.ok) {
      throw new Error(`API error (${recRes.status} ${recRes.statusText})`);
    }

    renderStats(stats);

    if (!Array.isArray(records) || records.length === 0) {
      emptyEl.hidden = false;
      tableWrapper.hidden = true;
      setStatus("No records yet.");
      return;
    }

    renderRows(records);
    emptyEl.hidden = true;
    tableWrapper.hidden = false;
    setStatus(`Loaded ${records.length} records.`);
  } catch (e) {
    showError(e?.message || String(e));
    setStatus("");
  }
}

reloadBtn.addEventListener("click", load);
load();
