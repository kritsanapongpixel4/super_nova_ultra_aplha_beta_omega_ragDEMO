"""Query transformations that run before retrieval.

- rewrite      : turn a context-dependent follow-up into a standalone question
- multi_query  : generate several phrasings and union their results
- hyde         : draft a hypothetical answer and search with that embedding
"""

from __future__ import annotations

import logging

from .generator import load_api_key

logger = logging.getLogger(__name__)

DEFAULT_REWRITE_MODEL = "gemini-3.6-flash"

_REWRITE_PROMPT = """คุณคือตัวช่วยเขียนคำถามใหม่ให้สมบูรณ์ในตัวเอง

ประวัติการสนทนา:
{history}

คำถามล่าสุด: {query}

เขียนคำถามล่าสุดใหม่ให้เข้าใจได้โดยไม่ต้องอ่านประวัติ โดยแทนคำอ้างอิง
เช่น "วิชานั้น" "อันแรก" ด้วยชื่อหรือรหัสจริงจากประวัติ
ถ้าคำถามสมบูรณ์อยู่แล้วให้ตอบกลับมาเหมือนเดิมทุกตัวอักษร
ตอบเป็นคำถามเดียวเท่านั้น ห้ามอธิบายหรือใส่เครื่องหมายคำพูด"""


_shared_client = None


def _client():
    """Return a cached Gen AI client.

    Cached, not built per call: a client created inline as
    ``genai.Client(...).models.generate_content(...)`` is garbage-collected
    while the request is still in flight, and its transport closes underneath
    it — "Cannot send a request, as the client has been closed."
    """
    global _shared_client
    if _shared_client is None:
        from google import genai

        _shared_client = genai.Client(api_key=load_api_key())
    return _shared_client


def rewrite_query(
    query: str,
    history: list[dict[str, str]] | None = None,
    model: str = DEFAULT_REWRITE_MODEL,
    client=None,
) -> str:
    """Rewrite a follow-up ("what about that one?") into a standalone question.

    Retrieval never sees the conversation — only this one string is embedded
    and BM25-matched.  So "แล้ว CLO ข้อแรกของวิชานั้นคืออะไร" retrieves nothing
    useful no matter how good the index is: the course it refers to is in the
    previous turn, not in the query.

    Falls back to the original query if the rewrite fails; a chat that answers
    from a slightly worse query beats a chat that crashes.
    """
    if not history:
        return query

    from google.genai import types

    transcript = "\n".join(
        f"{'ผู้ใช้' if turn['role'] == 'user' else 'ระบบ'}: {turn['content']}"
        for turn in history
    )
    prompt = _REWRITE_PROMPT.format(history=transcript, query=query)

    try:
        response = (client or _client()).models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(max_output_tokens=500),
        )
    except Exception:
        logger.warning("เขียนคำถามใหม่ไม่สำเร็จ — ใช้คำถามเดิม", exc_info=True)
        return query

    rewritten = (response.text or "").strip().strip('"').strip()
    if not rewritten:
        return query
    if rewritten != query:
        logger.info("🔁 เขียนคำถามใหม่: %s", rewritten)
    return rewritten


def multi_query(query: str, n: int = 3) -> list[str]:
    """Return n alternative phrasings of the query (the original included)."""
    raise NotImplementedError


def hyde(query: str) -> str:
    """Generate a hypothetical answer document to embed instead of the question."""
    raise NotImplementedError
