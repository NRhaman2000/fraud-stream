"use strict";
// Dashboard client: Server-Sent Events for live stats + alerts, REST for the review queue.
// All untrusted text (merchant names etc.) is set via textContent, never parsed as HTML.

const $ = (id) => document.getElementById(id);
const HISTORY = 120; // points kept per chart (one per stats push)
const series = { tps: [], p50: [], p95: [] };
let filter = "open";
let cursor = null;

const fmt = {
  int: (n) => (n == null ? "–" : Math.round(n).toLocaleString()),
  ms: (n) => (n == null ? "–" : n < 10 ? n.toFixed(1) : Math.round(n).toString()),
  ratio: (n) => (n == null ? "–" : n.toFixed(3)),
  money: (n) => "$" + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  time: (iso) => new Date(iso).toLocaleTimeString(),
};

function cell(text, cls) {
  const td = document.createElement("td");
  td.textContent = text;
  if (cls) td.className = cls;
  return td;
}

function truthCell(label) {
  const td = document.createElement("td");
  td.dataset.truth = "";
  if (label === null || label === undefined) { td.textContent = "pending"; td.className = "muted"; return td; }
  const span = document.createElement("span");
  span.className = "tag " + (label ? "fraud" : "ok");
  span.textContent = label ? "fraud" : "legit";
  td.appendChild(span);
  return td;
}

function reviewCell(alert) {
  const td = document.createElement("td");
  if (alert.status && alert.status !== "open") {
    td.textContent = alert.status === "confirmed_fraud" ? "confirmed fraud" : "false positive";
    td.className = "muted";
    return td;
  }
  for (const [label, decision, cls] of [["Fraud", "confirmed_fraud", "fraud"], ["Not fraud", "false_positive", "fp"]]) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label; b.className = cls;
    b.addEventListener("click", () => review(alert.tx_id, decision, td));
    td.appendChild(b);
    td.appendChild(document.createTextNode(" "));
  }
  return td;
}

function row(alert, isNew) {
  const tr = document.createElement("tr");
  tr.dataset.tx = alert.tx_id;
  if (isNew) tr.className = "new";
  tr.append(
    cell(fmt.time(alert.flagged_at)),
    cell(alert.card_id.slice(0, 8) + "…"),
    cell(alert.merchant, "merchant"),
    cell(alert.category.replace("_", " ")),
    cell(fmt.money(alert.amount), "num"),
    cell(alert.score.toFixed(3), "num"),
    truthCell(alert.true_label),
    reviewCell(alert),
  );
  return tr;
}

async function review(txId, decision, td) {
  td.textContent = "saving…";
  const res = await fetch(`/api/alerts/${encodeURIComponent(txId)}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status: decision }),
  });
  if (res.ok) {
    const alert = await res.json();
    td.replaceWith(reviewCell(alert));
    if (filter === "open") document.querySelector(`tr[data-tx="${txId}"]`)?.remove();
  } else {
    const body = await res.json().catch(() => ({}));
    td.textContent = body?.error?.message || `error ${res.status}`;
  }
}

async function loadAlerts(reset) {
  if (reset) { $("alerts").replaceChildren(); cursor = null; }
  const params = new URLSearchParams({ limit: "50" });
  if (filter) params.set("status", filter);
  if (cursor) { params.set("before_flagged_at", cursor.at); params.set("before_tx_id", cursor.tx); }
  const res = await fetch(`/api/alerts?${params}`);
  if (!res.ok) { $("msg").textContent = "Could not load alerts."; return; }
  const page = await res.json();
  for (const a of page.items) $("alerts").appendChild(row(a, false));
  cursor = page.next_before_tx_id ? { at: page.next_before_flagged_at, tx: page.next_before_tx_id } : null;
  $("more").hidden = !cursor;
  $("msg").textContent = $("alerts").children.length ? "" : "No alerts yet.";
}

function drawChart(canvas, lines) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr; ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const all = lines.flatMap((l) => l.data).filter((v) => v != null);
  const max = Math.max(1, ...all) * 1.15;
  ctx.strokeStyle = "#262c38"; ctx.fillStyle = "#8a93a6"; ctx.font = "11px system-ui"; ctx.lineWidth = 1;
  for (const f of [0, 0.5, 1]) {
    const y = h - 16 - f * (h - 24);
    ctx.beginPath(); ctx.moveTo(36, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.fillText(Math.round(max * f).toLocaleString(), 0, y + 4);
  }
  for (const line of lines) {
    ctx.strokeStyle = line.color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    line.data.forEach((v, i) => {
      if (v == null) return;
      const x = 36 + (i / (HISTORY - 1)) * (w - 40);
      const y = h - 16 - (v / max) * (h - 24);
      if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
    });
    ctx.stroke();
  }
}

function push(arr, v) { arr.push(v); if (arr.length > HISTORY) arr.shift(); }

function onStats(s) {
  $("tps").textContent = fmt.int(s.tps);
  $("p95").textContent = fmt.ms(s.p95_ms);
  $("scored").textContent = fmt.int(s.scored_total);
  $("flagged").textContent = fmt.int(s.flagged_total);
  $("precision").textContent = fmt.ratio(s.precision);
  $("recall").textContent = fmt.ratio(s.recall);
  $("labelled").textContent = fmt.int(s.labelled_total);
  $("model").textContent = s.model_version || "–";
  $("window").textContent = s.window_s;
  push(series.tps, s.tps); push(series.p50, s.p50_ms); push(series.p95, s.p95_ms);
  drawChart($("chart-tps"), [{ data: series.tps, color: "#4f8cff" }]);
  drawChart($("chart-lat"), [{ data: series.p50, color: "#3ecf8e" }, { data: series.p95, color: "#9b7bff" }]);
}

function onAlert(a) {
  if (filter !== "open" && filter !== "") return;
  if (document.querySelector(`tr[data-tx="${a.tx_id}"]`)) return;
  // Live alerts carry the scorer's clock (ns); stored ones carry flagged_at.
  const flaggedAt = new Date(a.scored_at_ns / 1e6).toISOString();
  $("alerts").prepend(row({ ...a, flagged_at: flaggedAt, status: "open", true_label: null }, true));
  while ($("alerts").children.length > 200) $("alerts").lastChild.remove();
  $("msg").textContent = "";
}

// Labels (chargebacks) arrive after the alert: fill in "pending" truth cells every few seconds.
async function refreshTruth() {
  const pending = document.querySelectorAll("#alerts td.muted[data-truth]");
  if (!pending.length) return;
  const params = new URLSearchParams({ limit: "200" });
  if (filter) params.set("status", filter);
  const res = await fetch(`/api/alerts?${params}`);
  if (!res.ok) return;
  const labels = new Map((await res.json()).items
    .filter((a) => a.true_label !== null).map((a) => [a.tx_id, a.true_label]));
  for (const td of pending) {
    const tx = td.parentElement?.dataset.tx;
    if (tx && labels.has(tx)) td.replaceWith(truthCell(labels.get(tx)));
  }
}

function connect() {
  const es = new EventSource("/api/stream");
  es.onopen = () => { $("conn").textContent = "live"; $("conn").className = "pill on"; };
  es.onerror = () => { $("conn").textContent = "reconnecting…"; $("conn").className = "pill off"; };
  es.addEventListener("stats", (e) => onStats(JSON.parse(e.data)));
  es.addEventListener("alert", (e) => onAlert(JSON.parse(e.data)));
}

$("filter").addEventListener("change", (e) => { filter = e.target.value; loadAlerts(true); });
$("more").addEventListener("click", () => loadAlerts(false));
loadAlerts(true);
connect();
setInterval(refreshTruth, 3000);
