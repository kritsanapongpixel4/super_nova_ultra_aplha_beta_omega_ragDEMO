"""LLM answer generation - turns retrieved chunks into a grounded answer.

Uses the Google Gen AI SDK (pip install google-genai).  The API key is read
from GEMINI_API_KEY, which .env supplies - never hardcode a key here, and
never commit .env (it is gitignored).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from . import prompt_templates

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

# Quota exhaustion and capacity pressure are both "try a different model";
# anything else (a bad key, a malformed request) must surface as-is.
_EXHAUSTED_MARKERS = ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503")


def _is_exhausted(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in _EXHAUSTED_MARKERS)

# Conversation history is stored in the neutral {"role": "user"|"assistant"}
# shape (see memory.ConversationMemory); Gemini calls the assistant "model".
_ROLE_MAP = {"user": "user", "assistant": "model", "model": "model"}


def load_api_key() -> str:
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
    """Wraps a Gemini call over the retrieved context, with model fallback."""

    def __init__(
        self,
        # Mirrors config.LLM_MODEL.  Kept as a literal rather than importing
        # config, to keep this module usable on its own — but it had drifted
        # to 3.6-flash, which the latency benchmark puts at 9.5-31s TTFT
        # against 3.9s for the model config actually selects.
        model: str = "gemini-3.5-flash",
        max_tokens: int = 16000,
        system_prompt: str = prompt_templates.SYSTEM_PROMPT,
        fallback_models: tuple[str, ...] = (),
        cooldown_seconds: float = 600.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        # Free-tier quota is counted per model and the models do not reset
        # together — on any given day some are exhausted while others are
        # untouched.  Pinning one model therefore breaks at random intervals.
        self.models = [model, *(m for m in fallback_models if m != model)]
        self.cooldown_seconds = cooldown_seconds
        self._cooldown: dict[str, float] = {}  # model -> retry-after timestamp
        self.last_model: str | None = None
        self._client = None  # built on first use so importing stays cheap

    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            # The SDK's default retry policy keeps re-trying a 429 with
            # growing backoff, so a request that has simply run out of free
            # quota hangs for minutes instead of failing.  Behind a chat UI
            # that reads as a frozen page.  Fail fast and let the caller show
            # the real reason.
            self._client = genai.Client(
                api_key=load_api_key(),
                http_options=types.HttpOptions(
                    timeout=60_000,  # milliseconds
                    retry_options=types.HttpRetryOptions(
                        attempts=2,
                        initial_delay=1.0,
                        max_delay=4.0,
                        http_status_codes=[503, 504],  # transient only, not 429
                    ),
                ),
            )
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

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            max_output_tokens=self.max_tokens,
        )

        now = time.monotonic()
        available = [m for m in self.models if self._cooldown.get(m, 0) <= now]
        # Everything is cooling down — try them all anyway rather than refuse
        # outright, since the cooldown is only a guess at when quota returns.
        for model in available or self.models:
            try:
                response = self.client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as exc:
                if not _is_exhausted(exc):
                    raise
                self._cooldown[model] = time.monotonic() + self.cooldown_seconds
                logger.warning("โมเดล %s ใช้ไม่ได้ชั่วคราว — ลองตัวถัดไป", model)
                continue

            self.last_model = model
            # response.text is None when the model returns no text part at
            # all — a safety block, or output cut off before anything was
            # written.
            return (response.text or "").strip()

        raise RuntimeError(
            "โมเดลทั้งหมดใช้ไม่ได้ในตอนนี้ (" + ", ".join(self.models) + ")"
        )
