const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const emptyEl = document.getElementById("empty");
const tableWrapper = document.getElementById("tableWrapper");
const bodyEl = document.getElementById("logBody");
const reloadBtn = document.getElementById("reloadBtn");
const statsEl = document.getElementById("stats");

const exportBtn = document.getElementById("exportBtn");
const exportMenu = document.getElementById("exportMenu");
const engineFilterEl = document.getElementById("engineFilter");
const topicFilterEl = document.getElementById("topicFilter");
const sentimentFilterEl = document.getElementById("sentimentFilter");

function setStatus(msg) {
  statusEl.textContent = msg || "";
}
function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = !msg;
}

// ---------------------------------------------------------------------------
// Export dropdown button: one menu item per supported export format.
// ---------------------------------------------------------------------------
function setExportMenuOpen(open) {
  exportMenu.hidden = !open;
  exportBtn.setAttribute("aria-expanded", String(open));
}

function buildExportMenu(formats) {
  exportMenu.innerHTML = "";
  for (const fmt of formats) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "dropdownItem";
    item.setAttribute("role", "menuitem");
    item.dataset.format = fmt;
    item.textContent = fmt.toUpperCase();
    exportMenu.appendChild(item);
  }
}

exportBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  setExportMenuOpen(exportMenu.hidden);
});

exportMenu.addEventListener("click", (e) => {
  const fmt = e.target.dataset.format;
  if (!fmt) return;
  setExportMenuOpen(false);
  const params = new URLSearchParams({ format: fmt });
  if (engineFilterEl.value) params.set("engine", engineFilterEl.value);
  if (topicFilterEl.value) params.set("topic", topicFilterEl.value);
  if (sentimentFilterEl.value) params.set("sentiment", sentimentFilterEl.value);
  window.location.href = `/records/export?${params.toString()}`;
});

document.addEventListener("click", () => setExportMenuOpen(false));
exportMenu.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setExportMenuOpen(false);
});

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

// ---------------------------------------------------------------------------
// Topic/sentiment filter dropdowns: rebuilt from the labels /stats reports
// as actually in use, so they never list a topic/sentiment with 0 records.
// The current selection is preserved across a rebuild when still valid.
// ---------------------------------------------------------------------------
function populateFilterOptions(selectEl, keys) {
  const current = selectEl.value;
  selectEl.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "";
  allOpt.textContent = "All";
  selectEl.appendChild(allOpt);
  for (const key of keys) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = key;
    selectEl.appendChild(opt);
  }
  if (keys.includes(current)) selectEl.value = current;
}

function engineLabel(item) {
  if (!item.provider) return "local";
  const parts = [item.provider];
  if (item.model) parts.push(item.model);
  if (typeof item.latency_ms === "number") parts.push(`${item.latency_ms} ms`);
  if (typeof item.cost_usd === "number") parts.push(`$${item.cost_usd.toFixed(4)}`);
  return parts.join(" · ");
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

    const tdEngine = document.createElement("td");
    tdEngine.textContent = engineLabel(item);
    tr.appendChild(tdEngine);

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
    const recParams = new URLSearchParams({ limit: "100" });
    if (engineFilterEl.value) recParams.set("engine", engineFilterEl.value);
    if (topicFilterEl.value) recParams.set("topic", topicFilterEl.value);
    if (sentimentFilterEl.value) recParams.set("sentiment", sentimentFilterEl.value);

    const [recRes, statRes] = await Promise.all([
      fetch(`/records?${recParams.toString()}`),
      fetch("/stats"),
    ]);
    const records = await recRes.json().catch(() => null);
    const stats = await statRes.json().catch(() => null);

    if (!recRes.ok) {
      throw new Error(`API error (${recRes.status} ${recRes.statusText})`);
    }

    renderStats(stats);
    buildExportMenu(stats?.export_formats || ["xml", "json", "csv"]);
    populateFilterOptions(topicFilterEl, Object.keys(stats?.by_topic || {}));
    populateFilterOptions(sentimentFilterEl, Object.keys(stats?.by_sentiment || {}));

    const activeFilters = [engineFilterEl.value, topicFilterEl.value, sentimentFilterEl.value].filter(
      Boolean
    );
    if (!Array.isArray(records) || records.length === 0) {
      emptyEl.textContent = activeFilters.length
        ? `No records match the current filter (${activeFilters.join(", ")}).`
        : "No records yet. Analyze a message or ingest a file first.";
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
engineFilterEl.addEventListener("change", load);
topicFilterEl.addEventListener("change", load);
sentimentFilterEl.addEventListener("change", load);
load();
