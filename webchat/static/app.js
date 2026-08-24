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

function hashOf(id) {
  let hash = 0;
  for (const ch of String(id)) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return hash;
}

/* Stable per-conversation tint so the list stays scannable. */
function tint(id) {
  const hue = hashOf(id) % 360;
  return `linear-gradient(145deg,hsl(${hue} 30% 64%),hsl(${(hue + 44) % 360} 26% 44%))`;
}

/* Initials were useless here: nearly every conversation starts with "วิชา",
   so the list was a column of identical "วิ". These abstract marks are
   derived from the id instead, so two conversations never look alike. */
const MARKS = [
  '<circle cx="12" cy="12" r="5.6"/>',
  '<circle cx="12" cy="12" r="5.4" fill="none" stroke="currentColor" stroke-width="2.3"/>',
  '<path d="M12 6 18 17.4H6z"/>',
  '<path d="M12 5.4 18.6 12 12 18.6 5.4 12z"/>',
  '<path d="M6.2 12a5.8 5.8 0 0 1 11.6 0z"/>',
  '<rect x="6.4" y="6.4" width="11.2" height="11.2" rx="3"/>',
  '<circle cx="8.6" cy="12" r="2.5"/><circle cx="15.4" cy="12" r="2.5"/>',
  '<rect x="5.6" y="9.1" width="12.8" height="2.5" rx="1.25"/><rect x="5.6" y="13.4" width="12.8" height="2.5" rx="1.25"/>',
];

function avatarMark(id) {
  const hash = hashOf(id);
  const shape = MARKS[hash % MARKS.length];
  const spin = (hash >> 3) % 4 * 45;
  return `<svg class="mark" viewBox="0 0 24 24" aria-hidden="true">
    <g transform="rotate(${spin} 12 12)">${shape}</g></svg>`;
}

/* The assistant gets one fixed mark so it never reads as "just another
   conversation" in the header. */
const ASSISTANT_MARK = `<svg class="mark" viewBox="0 0 24 24" aria-hidden="true">
  <circle cx="12" cy="12" r="8.2" fill="none" stroke="currentColor" stroke-width="1.7"/>
  <circle cx="12" cy="12" r="3.1"/></svg>`;

const clock = () => new Date().toTimeString().slice(0, 5);

/* ── conversations ───────────────────────────────────────── */

function drawConversations() {
  convList.innerHTML = "";
  for (const conv of conversations) {
    const item = document.createElement("button");
    item.className = "conv" + (conv.id === activeId ? " active" : "");
    item.innerHTML = `
      <div class="avatar" style="background:${tint(conv.id)}">${avatarMark(conv.id)}</div>
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

/* Every one of these was run against the live index before being offered —
   a suggested question that answers "ไม่พบข้อมูล" teaches the user that the
   system does not work. */
const SUGGESTIONS = [
  {
    group: "รายวิชาและ CLO",
    items: [
      "วิชา 04-620-201 มี CLO อะไรบ้าง",
      "วิชาระบบฐานข้อมูลมี CLO อะไรบ้าง",
      "วิชาไหนสอนเรื่องปัญญาประดิษฐ์",
    ],
  },
  {
    group: "ภาพรวมหลักสูตร",
    items: [
      "หลักสูตรมีกี่หน่วยกิต",
      "อาชีพที่ทำได้หลังจบหลักสูตรนี้มีอะไรบ้าง",
      "วิชาไหนบ้างที่สอดคล้องกับ PLO8",
    ],
  },
];

function emptyState() {
  const wrap = document.createElement("div");
  wrap.className = "empty";
  wrap.innerHTML = `
    <div class="empty-head">
      <div class="avatar xl">${ASSISTANT_MARK}</div>
      <h2>ถามเกี่ยวกับหลักสูตรได้เลย</h2>
      <p>ตอบจากเอกสารหลักสูตรวิศวกรรมคอมพิวเตอร์ พร้อมอ้างอิงกลับไปที่เอกสารทุกครั้ง</p>
    </div>
    ${SUGGESTIONS.map(
      (s) => `
      <div class="sug-group">
        <h3>${escapeHtml(s.group)}</h3>
        <div class="sug-list">
          ${s.items
            .map((q) => `<button class="sug" data-q="${escapeHtml(q)}">${escapeHtml(q)}</button>`)
            .join("")}
        </div>
      </div>`
    ).join("")}`;

  for (const button of wrap.querySelectorAll(".sug")) {
    button.onclick = () => {
      input.value = button.dataset.q;
      composer.requestSubmit();
    };
  }
  return wrap;
}

function drawThread() {
  const conv = conversations.find((c) => c.id === activeId);
  thread.innerHTML = "";
  if (!conv) return;

  if (!conv.messages.length) {
    thread.appendChild(emptyState());
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
      ${kv("โมเดลที่ตั้งไว้", generation.model)}
      ${kv("โมเดลที่ตอบจริง", generation.active_model || "ยังไม่ได้ถาม")}
      ${kv("โมเดลสำรอง", generation.fallbacks.length)}
      ${kv("พักอยู่ (โควตาหมด)", generation.resting.length ? generation.resting.join(", ") : "ไม่มี")}
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
    const gen = info.generation;
    const model = gen.active_model || gen.model;
    // Say so when a fallback is carrying the load — otherwise the header
    // claims a model that is actually sitting out a quota cooldown.
    const swapped = gen.active_model && gen.active_model !== gen.model ? " · สำรอง" : "";
    $("peerSub").textContent =
      `${model}${swapped} · ${info.index.chunks.toLocaleString()} chunks · top-${info.retrieval.top_k}`;
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

$("peerAvatar").innerHTML = ASSISTANT_MARK;

loadInfo();
loadConversations();
input.focus();
