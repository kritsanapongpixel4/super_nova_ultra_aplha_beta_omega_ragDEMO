"""LLM answer generation - turns retrieved chunks into a grounded answer.

Uses the Anthropic SDK (pip install anthropic). Credentials are resolved from
the environment (ANTHROPIC_API_KEY, or an `ant auth login` profile) - never
hardcode a key here.
"""

from typing import Any

import anthropic

from . import prompt_templates


class Generator:
    """Wraps a single Claude call over the retrieved context."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        max_tokens: int = 16000,
        system_prompt: str = prompt_templates.SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic()

    def generate(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Answer `question` using only `chunks`, optionally with prior turns."""
        if not chunks:
            return prompt_templates.NO_CONTEXT_MESSAGE

        messages = list(history or [])
        messages.append(
            {"role": "user", "content": prompt_templates.build_prompt(question, chunks)}
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=messages,
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
