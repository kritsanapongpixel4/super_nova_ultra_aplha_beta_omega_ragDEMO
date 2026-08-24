"""Local web chat for the RAG system.

    python webchat/app.py            then open http://127.0.0.1:5000

Kept separate from main.py on purpose: the CLI stays dependency-free, and
this only adds Flask on top of the same RAGPipeline.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

import config  # noqa: E402
from src.memory import ConversationMemory  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = Flask(__name__, static_folder="static", static_url_path="/static")

_pipeline: RAGPipeline | None = None
# One pipeline, one embedding model, one FAISS index — serialise access so two
# browser tabs cannot interleave inside a single retrieve/generate cycle.
_lock = threading.Lock()

# conversation id -> {"title", "created", "memory", "messages"}
_conversations: dict[str, dict] = {}

# Live counters for the details drawer — a config dump alone says what the
# system is set to, not how it is actually behaving.
_stats: dict[str, object] = {
    "questions": 0,
    "errors": 0,
    "last_latency": None,
    "avg_latency": None,
    "_total_latency": 0.0,
}


def _now() -> str:
    return datetime.now().strftime("%H:%M")


def _new_conversation() -> dict:
    conversation = {
        "id": uuid.uuid4().hex[:12],
        "title": "การสนทนาใหม่",
        "created": datetime.now().isoformat(timespec="seconds"),
        "memory": ConversationMemory(config.MEMORY_MAX_TURNS),
        "messages": [],
    }
    _conversations[conversation["id"]] = conversation
    return conversation


def _public(conversation: dict) -> dict:
    """The shape the browser sees — everything except the memory object."""
    return {
        "id": conversation["id"],
        "title": conversation["title"],
        "created": conversation["created"],
        "messages": conversation["messages"],
        "preview": (
            conversation["messages"][-1]["text"][:44]
            if conversation["messages"]
            else "ยังไม่มีข้อความ"
        ),
        "time": (
            conversation["messages"][-1]["time"]
            if conversation["messages"]
            else _now()
        ),
    }


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    from src.index_meta import read_meta

    meta = read_meta(config.INDEX_META_FILE) or {}
    store = _pipeline.retriever.store if _pipeline else None

    return jsonify(
        {
            "index": {
                "chunks": len(store) if store else 0,
                "dim": store.dim if store else config.EMBEDDING_DIM,
                "built_at": meta.get("built_at"),
                "sources": meta.get("sources", []),
                "categories": config.SOURCE_CATEGORIES or ["ทั้งหมด"],
            },
            "chunking": {
                "size": config.CHUNK_SIZE,
                "overlap": config.CHUNK_OVERLAP,
                "min_letters": config.MIN_CHUNK_LETTERS,
            },
            "retrieval": {
                "embedding_model": config.EMBEDDING_MODEL,
                "top_k": config.TOP_K,
                "candidate_k": config.CANDIDATE_K,
                "rrf_k": config.RRF_K,
                "reranker": config.RERANKER_MODEL if config.USE_RERANKER else None,
            },
            "generation": {
                "provider": config.LLM_PROVIDER,
                "model": config.LLM_MODEL,
                "max_tokens": config.LLM_MAX_TOKENS,
                "memory_turns": config.MEMORY_MAX_TURNS,
            },
            # Underscore keys are the running totals behind the averages —
            # bookkeeping, not something the page should have to ignore.
            "stats": {k: v for k, v in _stats.items() if not k.startswith("_")},
        }
    )


@app.get("/api/conversations")
def list_conversations():
    return jsonify([_public(c) for c in _conversations.values()])


@app.post("/api/conversations")
def create_conversation():
    return jsonify(_public(_new_conversation()))


@app.delete("/api/conversations/<conversation_id>")
def delete_conversation(conversation_id: str):
    _conversations.pop(conversation_id, None)
    return jsonify({"ok": True})


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "ข้อความว่าง"}), 400

    conversation = _conversations.get(payload.get("conversation_id") or "")
    if conversation is None:
        conversation = _new_conversation()

    conversation["messages"].append(
        {"role": "user", "text": question, "time": _now()}
    )
    if conversation["title"] == "การสนทนาใหม่":
        conversation["title"] = question[:38]

    started = time.perf_counter()
    try:
        with _lock:
            _pipeline.memory = conversation["memory"]
            result = _pipeline.answer(question)
    except Exception as exc:  # keep the tab alive and say what broke
        _stats["errors"] += 1
        logging.exception("chat failed")
        detail = str(exc)
        # The raw SDK errors are a wall of JSON; say the useful part instead.
        if "RESOURCE_EXHAUSTED" in detail or "429" in detail:
            text = (
                "โควตา Gemini หมดชั่วคราว (429) — รอสักครู่แล้วลองใหม่ "
                "หรือเปลี่ยน LLM_MODEL ใน config.py เป็นโมเดลอื่น"
            )
        elif "UNAVAILABLE" in detail or "503" in detail:
            text = "โมเดลมีผู้ใช้งานหนาแน่น (503) — ลองใหม่อีกครั้ง"
        else:
            text = f"เกิดข้อผิดพลาด: {detail[:300]}"
        message = {
            "role": "assistant",
            "text": text,
            "time": _now(),
            "sources": [],
            "error": True,
        }
        conversation["messages"].append(message)
        return jsonify({"conversation": _public(conversation), "message": message}), 500

    sources = [
        {
            "n": index,
            "source": chunk.get("source", ""),
            "line_start": chunk.get("line_start"),
            "line_end": chunk.get("line_end"),
            "course_code": chunk.get("course_code"),
            "text": chunk.get("text", ""),
            "pinned": bool(chunk.get("pinned")),
        }
        for index, chunk in enumerate(result["sources"], start=1)
    ]
    latency = round(time.perf_counter() - started, 2)
    _stats["questions"] += 1
    _stats["last_latency"] = latency
    _stats["_total_latency"] += latency
    _stats["avg_latency"] = round(_stats["_total_latency"] / _stats["questions"], 2)

    message = {
        "role": "assistant",
        "text": result["answer"],
        "time": _now(),
        "sources": sources,
        "latency": latency,
        "pinned_hit": any(s["pinned"] for s in sources),
    }
    conversation["messages"].append(message)
    return jsonify({"conversation": _public(conversation), "message": message})


def main() -> None:
    global _pipeline

    print("กำลังโหลด RAG pipeline (ครั้งแรกใช้เวลาสักครู่)...")
    started = datetime.now()
    try:
        _pipeline = RAGPipeline.from_config(use_memory=True)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"เริ่มไม่สำเร็จ: {exc}")
        sys.exit(1)

    # Warm the embedding model now rather than making the first question wait
    # ~30s for a model load it did not ask for.
    _pipeline.retriever.embedder.load()

    elapsed = (datetime.now() - started).total_seconds()
    print(f"พร้อมใช้งานใน {elapsed:.0f}s")
    print(f"  chunks   : {len(_pipeline.retriever.store)}")
    print(f"  โมเดลตอบ : {config.LLM_MODEL}")
    print("\n  http://127.0.0.1:5000\n")

    _new_conversation()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
