"""Conversation history - keeps the last few turns so follow-ups make sense."""


class ConversationMemory:
    """A bounded list of {"role": ..., "content": ...} message dicts."""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self.messages: list[dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self._add("user", content)

    def add_assistant(self, content: str) -> None:
        self._add("assistant", content)

    def _add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # one turn = one user message + one assistant message
        limit = self.max_turns * 2
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def history(self) -> list[dict[str, str]]:
        """Messages to prepend to the next request."""
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
