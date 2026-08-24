"use strict";

const $ = (id) => document.getElementById(id);
const convList = $("convList");
const thread = $("thread");
const input = $("input");
const sendBtn = $("send");
const composer = $("composer");

let conversations = [];
let activeId = null;
let busy = false;

/* ── helpers ─────────────────────────────────────────────── */

const escapeHtml = (s) =>
  s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

/** Minimal, deliberate subset: the model writes **bold** and [1] citations. */
function render(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d{1,2})\]/g, '<span class="cite">$1</span>')
    .replace(/^[*-]\s+/gm, "· ");
}

function initials(title) {
  const word = (title || "?").trim().split(/\s+/)[0];
  return word.slice(0, 2).toUpperCase();
}

/* Stable per-conversation tint so the list stays scannable. */
function tint(id) {
  let hash = 0;
  for (const ch of id) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
  return `linear-gradient(145deg,hsl(${hash} 26% 62%),hsl(${(hash + 40) % 360} 24% 44%))`;
}

/* ── rendering ───────────────────────────────────────────── */

function drawConversations() {
  convList.innerHTML = "";
  for (const conv of conversations) {
    const item = document.createElement("button");
    item.className = "conv" + (conv.id === activeId ? " active" : "");
    item.innerHTML = `
      <div class="avatar" style="background:${tint(conv.id)}">${escapeHtml(initials(conv.title))}</div>
      <div class="conv-body">
        <div class="conv-name">${escapeHtml(conv.title)}</div>
        <div class="conv-preview">${escapeHtml(conv.preview)}</div>
      </div>
      <div class="conv-time">${escapeHtml(conv.time)}</div>`;
    item.onclick = () => {
      if (busy) return;
      activeId = conv.id;
      drawConversations();
      drawThread();
    };
    convList.appendChild(item);
  }
}

function sourcesHtml(sources) {
  if (!sources || !sources.length) return "";
  const rows = sources
    .map(
      (s) => `
      <div class="src">
        <div class="src-n">${s.n}</div>
        <div>
          <div class="src-file">${escapeHtml(s.source)} · หน้า ${s.line_start}${
        s.course_code ? " · " + escapeHtml(s.course_code) : ""
      }${s.pinned ? '<span class="tag">ปักหมุด</span>' : ""}</div>
          <div class="src-text">${escapeHtml(s.text.split("\n")[0])}</div>
        </div>
      </div>`
    )
    .join("");
  return `<details class="sources"><summary>อ้างอิง ${sources.length} รายการ</summary>${rows}</details>`;
}

function drawThread() {
  const conv = conversations.find((c) => c.id === activeId);
  thread.innerHTML = "";
  if (!conv) return;

  if (!conv.messages.length) {
    const hint = document.createElement("div");
    hint.className = "day";
    hint.textContent = "ถามเกี่ยวกับหลักสูตร รายวิชา CLO หรือ PLO ได้เลย";
    thread.appendChild(hint);
  }

  for (const msg of conv.messages) {
    const row = document.createElement("div");
    row.className = "row " + (msg.role === "user" ? "out" : "in");
    row.innerHTML = `
      <div class="msg">
        <div class="bubble${msg.error ? " error" : ""}">${render(msg.text)}</div>
        ${msg.role === "assistant" ? sourcesHtml(msg.sources) : ""}
        <div class="stamp">${escapeHtml(msg.time)}</div>
      </div>`;
    thread.appendChild(row);
  }
  thread.scrollTop = thread.scrollHeight;
}

function showTyping() {
  const row = document.createElement("div");
  row.className = "row in";
  row.id = "typingRow";
  row.innerHTML = '<div class="typing"><i></i><i></i><i></i></div>';
  thread.appendChild(row);
  thread.scrollTop = thread.scrollHeight;
}

const hideTyping = () => $("typingRow")?.remove();

function setStatus(state, text) {
  $("statusDot").className = "status " + state;
  $("statusText").textContent = text;
}

/* ── data ────────────────────────────────────────────────── */

async function loadConversations() {
  conversations = await (await fetch("/api/conversations")).json();
  if (!conversations.length) return newConversation();
  if (!conversations.some((c) => c.id === activeId)) {
    activeId = conversations[conversations.length - 1].id;
  }
  drawConversations();
  drawThread();
}

async function newConversation() {
  const conv = await (await fetch("/api/conversations", { method: "POST" })).json();
  conversations.push(conv);
  activeId = conv.id;
  drawConversations();
  drawThread();
  input.focus();
}

async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    $("peerSub").textContent = `${info.model} · ${info.chunks} chunks · top-${info.top_k}`;
    $("infoPanel").innerHTML = `
      <div><b>โมเดลตอบ</b>${escapeHtml(info.model)}</div>
      <div><b>โมเดล embedding</b>${escapeHtml(info.embedding_model)}</div>
      <div><b>chunks ในคลัง</b>${info.chunks}</div>
      <div><b>top-k</b>${info.top_k}</div>
      <div><b>reranker</b>${info.reranker ? escapeHtml(info.reranker) : "ปิดอยู่"}</div>
      <div><b>หมวดข้อมูล</b>${escapeHtml((info.categories || ["ทั้งหมด"]).join(", "))}</div>`;
    setStatus("on", "พร้อมใช้งาน");
  } catch {
    setStatus("err", "เชื่อมต่อไม่ได้");
  }
}

/* ── events ──────────────────────────────────────────────── */

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || busy) return;

  busy = true;
  sendBtn.disabled = true;
  input.value = "";

  const conv = conversations.find((c) => c.id === activeId);
  // Show the question immediately; the server echoes the authoritative copy
  // back with the answer, which then replaces this optimistic one.
  conv.messages.push({
    role: "user",
    text,
    time: new Date().toTimeString().slice(0, 5),
  });
  if (conv.title === "การสนทนาใหม่") conv.title = text.slice(0, 38);
  drawConversations();
  drawThread();
  showTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: activeId, message: text }),
    });
    const data = await response.json();
    hideTyping();
    if (data.conversation) {
      Object.assign(conv, data.conversation);
      drawConversations();
      drawThread();
    }
    setStatus(response.ok ? "on" : "err", response.ok ? "พร้อมใช้งาน" : "เกิดข้อผิดพลาด");
  } catch (err) {
    hideTyping();
    conv.messages.push({
      role: "assistant",
      text: "ติดต่อเซิร์ฟเวอร์ไม่ได้: " + err.message,
      time: new Date().toTimeString().slice(0, 5),
      error: true,
      sources: [],
    });
    drawThread();
    setStatus("err", "เชื่อมต่อไม่ได้");
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
});

$("newChat").onclick = () => !busy && newConversation();
$("infoBtn").onclick = () => ($("infoPanel").hidden = !$("infoPanel").hidden);

loadInfo();
loadConversations();
input.focus();
