"""Shared data models for aipou — provider-agnostic."""

from dataclasses import dataclass, field
from typing import Literal, Optional


Role = Literal["user", "assistant", "system"]


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)


@dataclass
class TokenUsage:
    """Token usage statistics from the API response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __repr__(self) -> str:
        return (
            f"TokenUsage(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens})"
        )


@dataclass
class AIResponse:
    """A complete response from an AI provider."""
    content: str
    model: str
    provider: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: Optional[str] = None
    raw: Optional[dict] = field(default=None, repr=False)

    def __str__(self) -> str:
        return self.content


@dataclass
class StreamChunk:
    """A single streamed chunk from an AI provider."""
    delta: str           # The new text fragment
    model: str
    provider: str
    finish_reason: Optional[str] = None
    is_final: bool = False

    def __str__(self) -> str:
        return self.delta