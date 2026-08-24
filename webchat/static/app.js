"use strict";

const $ = (id) => document.getElementById(id);
const convList = $("convList");
const thread = $("thread");
const input = $("input");
const sendBtn = $("send");
const composer = $("composer");
const drawer = $("drawer");

let conversations = [];
let activeId = null;
let busy = false;
let lastPinned = false;

/* ── helpers ─────────────────────────────────────────────── */

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

/** Minimal, deliberate subset: the model writes **bold** and [1] citations. */
function render(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d{1,2})\]/g, '<span class="cite">$1</span>')
    .replace(/^\s*[*-]\s+/gm, "· ");
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

const clock = () => new Date().toTimeString().slice(0, 5);

/* ── conversations ───────────────────────────────────────── */

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

/* ── thread ──────────────────────────────────────────────── */

function sourcesHtml(sources) {
  if (!sources || !sources.length) return "";
  const rows = sources
    .map(
      (s) => `
      <div class="src">
        <div class="src-n">${s.n}</div>
        <div>
          <div class="src-file">${escapeHtml(s.source)} · หน้า ${escapeHtml(s.line_start)}${
        s.course_code ? " · " + escapeHtml(s.course_code) : ""
      }${s.pinned ? '<span class="tag">ปักหมุด</span>' : ""}</div>
          <div class="src-text">${escapeHtml(s.text.split("\n")[0])}</div>
        </div>
      </div>`
    )
    .join("");
  return `<details class="sources"><summary>อ้างอิง ${sources.length} รายการ</summary>${rows}</details>`;
}

function metaHtml(msg) {
  const bits = [escapeHtml(msg.time)];
  if (msg.latency != null) bits.push(`<span class="lat">${msg.latency}s</span>`);
  if (msg.sources && msg.sources.length) bits.push(`${msg.sources.length} อ้างอิง`);
  return `<div class="meta">${bits.join('<span class="dot"></span>')}</div>`;
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
        ${metaHtml(msg)}
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

/* ── details drawer ──────────────────────────────────────── */

const kv = (k, v) =>
  `<div class="kv"><span class="k">${escapeHtml(k)}</span><span class="v">${escapeHtml(v)}</span></div>`;

/* Filenames are what the pipeline stores; these are what a person calls them. */
const DOC_LABELS = [
  [/^CLOs/i, "ตาราง CLO–PLO ของทุกรายวิชา"],
  [/^หลักสูตร-683/, "หลักสูตรวิศวกรรมคอมพิวเตอร์ ฉบับปรับปรุง 2568"],
  [/^หลักสูตร63/, "หลักสูตรวิศวกรรมคอมพิวเตอร์ ฉบับปรับปรุง 2563"],
];

const docLabel = (file) =>
  (DOC_LABELS.find(([re]) => re.test(file)) || [null, file.replace(/\.pdf$/i, "")])[1];

function thaiDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleDateString("th-TH", { day: "numeric", month: "short", year: "numeric" });
}

/* Three shapes of question the system genuinely answers — verified against
   the live index, because a suggestion that returns "ไม่พบข้อมูล" is worse
   than no suggestion at all.  Ask by code, by name, and in reverse. */
const EXAMPLES = [
  "วิชา 04-620-201 มี CLO อะไรบ้าง",
  "วิชาระบบฐานข้อมูลมี CLO อะไรบ้าง",
  "วิชาไหนบ้างที่สอดคล้องกับ PLO8",
];

function drawDrawer(info) {
  const { index, chunking, retrieval, generation, stats } = info;

  const docs = index.sources
    .map((f) => `<div class="doc">${escapeHtml(docLabel(f))}</div>`)
    .join("");

  const examples = EXAMPLES.map(
    (q) => `<button class="try" data-q="${escapeHtml(q)}">${escapeHtml(q)}</button>`
  ).join("");

  $("drawerBody").innerHTML = `
    <div class="sec">
      <div class="tiles">
        <div class="tile"><b>${index.sources.length}</b><span>เอกสาร</span></div>
        <div class="tile"><b>${index.chunks.toLocaleString()}</b><span>ท่อนข้อความ</span></div>
        <div class="tile"><b>${retrieval.top_k}</b><span>อ้างอิง/คำตอบ</span></div>
      </div>
    </div>

    <div class="sec">
      <h3>ตอบจากเอกสารเหล่านี้</h3>
      ${docs}
      <p style="margin-top:9px">อัปเดตล่าสุด ${escapeHtml(thaiDate(index.built_at))}</p>
    </div>

    <div class="sec">
      <h3>ทำงานอย่างไร</h3>
      <ol class="how">
        <li>ค้นหาข้อความที่เกี่ยวข้องกับคำถามจากเอกสารทั้งหมด</li>
        <li>เลือกมา ${retrieval.top_k} ท่อนที่ตรงที่สุด ส่งให้ AI อ่าน</li>
        <li>AI ตอบจากข้อความเหล่านั้นเท่านั้น พร้อมใส่เลขอ้างอิง</li>
      </ol>
      <p>ตัวเลข <span class="cite">1</span> ในคำตอบคือแหล่งที่มา กดดูได้ที่ปุ่ม "อ้างอิง" ใต้คำตอบ</p>
    </div>

    <div class="sec">
      <h3>ลองถามดู</h3>
      ${examples}
    </div>

    <div class="sec">
      <h3>สิ่งที่ยังตอบไม่ได้</h3>
      <div class="note">
        ตอนนี้มีเฉพาะเอกสารหลักสูตร — ยังไม่มีแบบฟอร์มคำร้อง
        และข้อกำหนดสหกิจศึกษา ถ้าถามเรื่องเหล่านี้ระบบจะตอบว่าไม่พบข้อมูล
      </div>
    </div>

    <details class="tech">
      <summary>ข้อมูลทางเทคนิค</summary>
      ${kv("โมเดลค้นหา", retrieval.embedding_model)}
      ${kv("มิติเวกเตอร์", index.dim)}
      ${kv("ขนาด chunk", `${chunking.size} / ซ้อน ${chunking.overlap}`)}
      ${kv("candidate_k", retrieval.candidate_k)}
      ${kv("RRF k", retrieval.rrf_k)}
      ${kv("reranker", retrieval.reranker || "ปิด")}
      ${kv("ปักหมุดรหัสวิชา", lastPinned ? "ใช้ในคำถามล่าสุด" : "เปิดอยู่")}
      ${kv("โมเดลตอบ", generation.model)}
      ${kv("จำบทสนทนา", `${generation.memory_turns} เทิร์น`)}
      ${kv("ถามไปแล้ว", `${stats.questions} ครั้ง`)}
      ${kv("เวลาตอบเฉลี่ย", stats.avg_latency != null ? stats.avg_latency + " วินาที" : "—")}
      ${kv("ข้อผิดพลาด", stats.errors)}
    </details>`;

  for (const button of $("drawerBody").querySelectorAll(".try")) {
    button.onclick = () => {
      input.value = button.dataset.q;
      input.focus();
      composer.requestSubmit();
    };
  }
}

async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    $("peerSub").textContent =
      `${info.generation.model} · ${info.index.chunks.toLocaleString()} chunks · top-${info.retrieval.top_k}`;
    $("footStat").textContent = info.stats.questions
      ? `${info.stats.questions} คำถาม · เฉลี่ย ${info.stats.avg_latency}s`
      : "";
    drawDrawer(info);
    setStatus("on", "พร้อมใช้งาน");
  } catch {
    setStatus("err", "เชื่อมต่อไม่ได้");
  }
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
  conv.messages.push({ role: "user", text, time: clock() });
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
    if (data.message) lastPinned = !!data.message.pinned_hit;
    if (data.conversation) {
      Object.assign(conv, data.conversation);
      drawConversations();
      drawThread();
    }
    setStatus(response.ok ? "on" : "err", response.ok ? "พร้อมใช้งาน" : "เกิดข้อผิดพลาด");
    loadInfo();
  } catch (err) {
    hideTyping();
    conv.messages.push({
      role: "assistant",
      text: "ติดต่อเซิร์ฟเวอร์ไม่ได้: " + err.message,
      time: clock(),
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

function toggleDrawer(open) {
  const show = open ?? drawer.hidden;
  drawer.hidden = !show;
  // The class widens the grid track; the chat shrinks rather than being
  // covered, so the composer stays reachable with the drawer open.
  document.querySelector(".glass").classList.toggle("with-drawer", show);
  $("infoBtn").classList.toggle("on", show);
  if (show) loadInfo();
}

$("newChat").onclick = () => !busy && newConversation();
$("infoBtn").onclick = () => toggleDrawer();
$("closeDrawer").onclick = () => toggleDrawer(false);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !drawer.hidden) toggleDrawer(false);
});

loadInfo();
loadConversations();
input.focus();
