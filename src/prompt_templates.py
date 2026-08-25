"""Prompt templates for grounded answer generation."""

SYSTEM_PROMPT = """You answer questions using ONLY the context provided below.

Rules:
- If the context does not contain the answer, say so plainly. Never invent facts.
- Cite the source number of every chunk you use, like [1] or [2].
- Each chunk is labelled with the document it came from. When two documents
  disagree, name them instead of picking one silently.
- Answer in the same language the user asked in.
- Keep the tone factual, clear, and non-judgmental."""

ANSWER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

# The document name, not the line range.  line_start/line_end belong to the
# record a chunk was split out of, so every chunk from the same record repeats
# the same span — 31% of chunks share theirs with another — which told the
# model nothing and told it wrongly.  The filename is what distinguishes two
# chunks that disagree, e.g. คู่มือนักศึกษา67 against คู่มือนักศึกษา69.
CHUNK_TEMPLATE = "[{index}] {source}\n{text}"

NO_CONTEXT_MESSAGE = "I could not find anything about that in the documents."


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into the numbered block the prompt expects."""
    return "\n\n".join(
        CHUNK_TEMPLATE.format(
            index=i,
            source=chunk.get("source") or "ไม่ทราบแหล่งที่มา",
            text=chunk["text"],
        )
        for i, chunk in enumerate(chunks, start=1)
    )


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build the user-turn text from a question and its retrieved chunks."""
    return ANSWER_TEMPLATE.format(context=format_context(chunks), question=question)
