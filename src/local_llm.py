"""Talk to a model running locally under Ollama.

Exists for one caller: the LLM judge in ``evaluation/eval_generation.py``.
Grading 205 answers needs 205 requests, the Gemini free tier allows 20 per
model per day, and a local model has no quota at all.

Deliberately *not* folded into ``Generator``.  That class is built around the
things a hosted free tier forces on you — a fallback chain across five models,
a per-model cooldown after a 429, a retry policy tuned to Gemini's error
codes.  None of it applies to a model running on this machine, and a provider
switch inside it would mean carrying that machinery down a path that can never
use it.  What the judge actually needs is one method that takes a prompt and
returns text, so that is all this is.

The answering side stays on Gemini on purpose: it is the system under
evaluation, and swapping the model would measure a different one.

Uses urllib rather than the openai or ollama packages — one HTTP POST does not
justify a dependency, and requirements.txt stays as it is.

    ollama serve                 # ปกติ installer ตั้งให้รันเป็น service อยู่แล้ว
    ollama pull <model>
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"


class OllamaUnavailable(RuntimeError):
    """Ollama is not reachable, or the model is not pulled."""


class OllamaGenerator:
    """Minimal stand-in for Generator.complete() backed by a local model."""

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        temperature: float = 0.2,
        timeout: float = 600.0,
        think_reserve: int = 3000,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        # Generous: an 8B model on a laptop GPU takes tens of seconds for a
        # long grading prompt, and a timeout here reads as a judging failure.
        self.timeout = timeout

        # Reasoning models (Qwen3, and others like it) think before they
        # answer.  Ollama keeps that thinking out of ``response`` but it still
        # spends the ``num_predict`` budget, so a caller asking for 1000
        # tokens got the thinking and 14 characters of a JSON object cut off
        # mid-key — which reads as "the judge failed" when the judge was fine
        # and the budget was not.  Reserve room on top of what the caller asks
        # for.  Turning thinking off instead is worse: the same question the
        # model scores correctly with thinking on, it gets wrong without it.
        self.think_reserve = think_reserve
        self.last_model = model

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaUnavailable(
                f"ติดต่อ ollama ที่ {self.host} ไม่ได้ ({exc}) — "
                "ตรวจว่ารันอยู่หรือยัง: ollama serve"
            ) from exc

    def available(self) -> list[str]:
        """Model names this host has pulled, or raise if it is not running."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as r:
                tags = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaUnavailable(
                f"ติดต่อ ollama ที่ {self.host} ไม่ได้ ({exc})"
            ) from exc
        return [m["name"] for m in tags.get("models", [])]

    def check(self) -> None:
        """Fail early and clearly, rather than mid-run on question 40."""
        names = self.available()
        if not any(n == self.model or n.startswith(f"{self.model}:") for n in names):
            raise OllamaUnavailable(
                f"ยังไม่มีโมเดล {self.model} — รัน: ollama pull {self.model}\n"
                f"   ที่มีอยู่: {', '.join(names) or '(ยังไม่มีเลย)'}"
            )

    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Same signature as Generator.complete, so the judge cannot tell."""
        data = self._post(
            "/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens + self.think_reserve,
                },
            },
        )
        text = (data.get("response") or "").strip()
        if data.get("done_reason") == "length":
            # Truncated output is usually unparseable rather than merely
            # short, and silently returning half a JSON object sends the
            # caller hunting for a parsing bug that is not there.
            logger.warning(
                "%s ตอบไม่จบ (num_predict=%d หมด) — ได้ %d ตัวอักษร",
                self.model, max_tokens + self.think_reserve, len(text),
            )
        return text
