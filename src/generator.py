"""LLM answer generation - turns retrieved chunks into a grounded answer.

Uses the Google Gen AI SDK (pip install google-genai).  The API key is read
from GEMINI_API_KEY, which .env supplies - never hardcode a key here, and
never commit .env (it is gitignored).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import prompt_templates

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

# Conversation history is stored in the neutral {"role": "user"|"assistant"}
# shape (see memory.ConversationMemory); Gemini calls the assistant "model".
_ROLE_MAP = {"user": "user", "assistant": "model", "model": "model"}


def _load_api_key() -> str:
    """Read GEMINI_API_KEY from the environment, falling back to .env."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key and _ENV_PATH.exists():
        from dotenv import load_dotenv

        load_dotenv(_ENV_PATH)
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "ไม่พบ GEMINI_API_KEY — คัดลอก .env.example เป็น .env แล้วใส่ key "
            "(ขอได้ที่ https://aistudio.google.com/apikey)"
        )
    return key


class Generator:
    """Wraps a single Gemini call over the retrieved context."""

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        max_tokens: int = 16000,
        system_prompt: str = prompt_templates.SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._client = None  # built on first use so importing stays cheap

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=_load_api_key())
        return self._client

    def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Answer `question` using only `chunks`, optionally with prior turns."""
        if not chunks:
            return prompt_templates.NO_CONTEXT_MESSAGE

        from google.genai import types

        contents = [
            types.Content(
                role=_ROLE_MAP.get(turn["role"], "user"),
                parts=[types.Part(text=turn["content"])],
            )
            for turn in (history or [])
        ]
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(text=prompt_templates.build_prompt(question, chunks))
                ],
            )
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=self.max_tokens,
            ),
        )

        # response.text is None when the model returns no text part at all —
        # a safety block or a response cut off before it wrote anything.
        return (response.text or "").strip()
