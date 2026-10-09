/* KB Copilot — chat, decision-path trace, sources, sync button. */
"use strict";

const $ = (id) => document.getElementById(id);
const chatEl = $("chat"), traceEl = $("trace"), sourcesEl = $("sources");
const auditEl = $("audit"), questionEl = $("question"), askBtn = $("ask-btn");
const reindexBtn = $("reindex-btn"), toastEl = $("toast"), chipsEl = $("status-chips");

let toastTimer = null;
function toast(text, kind = "") {
  toastEl.textContent = text;
  toastEl.className = "toast " + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add("hidden"), 6000);
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}

/* Render a bot answer with lightweight markdown: **bold**, [n] citations kept, links. */
function renderAnswer(text) {
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  html = html.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  return html;
}

function addMsg(kind, html, metaHtml) {
  const div = document.createElement("div");
  div.className = "msg " + kind;
  div.innerHTML = html + (metaHtml ? `<div class="meta">${metaHtml}</div>` : "");
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

/* ── Decision-path timeline ── */
function renderTrace(events) {
  if (!events || !events.length) { traceEl.className = "trace empty"; traceEl.textContent = "No trace."; return; }
  traceEl.className = "trace";
  traceEl.innerHTML = "";
  for (const ev of events) {
    const cls = /PASS|GOOD|draft|results|revision|RETRY/.test(ev.decision) ? "good"
              : /WEAK|SKIPPED|REVISE/.test(ev.decision) ? "weak"
              : /ERROR/.test(ev.decision) ? "error" : "";
    const step = document.createElement("div");
    step.className = "trace-step " + cls;
    step.innerHTML =
      `<div class="trace-head"><span class="trace-agent">${escapeHtml(ev.agent)}</span>` +
      `<span class="trace-decision">${escapeHtml(ev.decision)}</span></div>` +
      (ev.detail ? `<div class="trace-detail">${escapeHtml(ev.detail)}</div>` : "") +
      `<div class="trace-ms">${ev.latency_ms ?? 0} ms</div>`;
    traceEl.appendChild(step);
  }
}

/* ── Sources ── */
function renderSources(data) {
  sourcesEl.className = "sources";
  sourcesEl.innerHTML = "";
  const cards = [];
  for (const s of data.kb_sources || []) {
    cards.push(
      `<div class="source-card"><span class="tag kbreader">kb</span> <span class="score">${s.score}</span>` +
      `<div class="path">${escapeHtml(s.path)}</div>` +
      `<div class="trace-detail">${escapeHtml(s.branches)}</div></div>`
    );
  }
  for (const s of data.glean_sources || []) {
    cards.push(
      `<div class="source-card"><span class="tag glean">Glean</span>` +
      `<div>${escapeHtml(s.title)}</div>` +
      (s.url ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.url)}</a>` : "") +
      `</div>`
    );
  }
  sourcesEl.innerHTML = cards.join("") || "No sources used.";
  if (!cards.length) sourcesEl.className = "sources empty";
}

/* ── Chat flow ── */
async function ask() {
  const q = questionEl.value.trim();
  if (q.length < 2) return;
  questionEl.value = "";
  askBtn.disabled = true;
  addMsg("user", escapeHtml(q));

  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.innerHTML = "<span></span><span></span><span></span>";
  chatEl.appendChild(thinking);
  chatEl.scrollTop = chatEl.scrollHeight;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const data = await resp.json();

    const tags = [];
    if ((data.kb_sources || []).length) tags.push('<span class="tag kbreader">kb</span>');
    if (data.used_glean) tags.push('<span class="tag glean">Glean fallback</span>');
    if (data.revisions > 0) tags.push(`<span class="tag revised">${data.revisions} revision(s)</span>`);
    addMsg(
      "bot",
      renderAnswer(data.final_answer),
      `<span>${(data.latency_ms / 1000).toFixed(1)}s</span>` + tags.join("")
    );
    renderTrace(data.events);
    renderSources(data);
  } catch (err) {
    addMsg("bot", `⚠️ ${escapeHtml(String(err.message || err))}`);
    toast("Request failed — check the server logs", "error");
  } finally {
    thinking.remove();
    askBtn.disabled = false;
    questionEl.focus();
  }
}
askBtn.addEventListener("click", ask);
questionEl.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });

/* ── Sync (reindex) button ── */
reindexBtn.addEventListener("click", async () => {
  if (reindexBtn.classList.contains("busy")) return;
  reindexBtn.classList.add("busy");
  try {
    const resp = await fetch("/api/reindex", { method: "POST" });
    const data = await resp.json();
    toast(data.message || "Sync started…");
    pollReindex();
  } catch {
    toast("Sync failed to start", "error");
    reindexBtn.classList.remove("busy");
  }
});

let reindexPoll = null;
function pollReindex() {
  clearInterval(reindexPoll);
  reindexPoll = setInterval(async () => {
    try {
      const s = (await (await fetch("/api/status")).json()).reindex;
      if (!s.running) {
        clearInterval(reindexPoll);
        reindexBtn.classList.remove("busy");
        if (s.error) toast("Sync failed: " + s.error, "error");
        else if (s.last) {
          const r = s.last;
          toast(`Synced: ${r.documents_new} new · ${r.documents_removed} removed · ` +
                `${r.documents_unchanged} unchanged · ${r.chunks_added} chunks · ${r.duration_s}s`, "ok");
        }
        refreshStatus();
      }
    } catch { /* transient */ }
  }, 1500);
}

/* ── Status chips + audit ── */
async function refreshStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    chipsEl.innerHTML =
      `<span class="chip ${s.glean_configured ? "ok" : "warn"}">Glean ${s.glean_configured ? "ready" : "no cookies"}</span>` +
      `<span class="chip">${s.chunks.toLocaleString()} chunks</span>` +
      `<span class="chip">${escapeHtml(s.llm_model)}</span>`;
    const a = s.audit;
    auditEl.innerHTML =
      `Queries: <b>${a.total_queries}</b><br>` +
      `Glean fallbacks: <b>${a.glean_fallbacks}</b><br>` +
      `Revised answers: <b>${a.revised_answers}</b>`;
    if (s.reindex.running) { reindexBtn.classList.add("busy"); pollReindex(); }
  } catch { /* server offline */ }
}
refreshStatus();

questionEl.focus();
