"""Prompt templates for grounded answer generation."""

SYSTEM_PROMPT = """You answer questions using ONLY the context provided below.

Rules:
- If the context does not contain the answer, say so plainly. Never invent facts.
- Cite the source number of every chunk you use, like [1] or [2].
- Answer in the same language the user asked in.
- Keep the tone factual, clear, and non-judgmental."""

ANSWER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

CHUNK_TEMPLATE = "[{index}] (lines {line_start}-{line_end})\n{text}"

NO_CONTEXT_MESSAGE = "I could not find anything about that in the documents."


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into the numbered block the prompt expects."""
    return "\n\n".join(
        CHUNK_TEMPLATE.format(
            index=i,
            line_start=chunk.get("line_start", "?"),
            line_end=chunk.get("line_end", "?"),
            text=chunk["text"],
        )
        for i, chunk in enumerate(chunks, start=1)
    )


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build the user-turn text from a question and its retrieved chunks."""
    return ANSWER_TEMPLATE.format(context=format_context(chunks), question=question)
