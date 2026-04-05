"""Base abstract provider for all AI API integrations."""

from abc import ABC, abstractmethod
from typing import Iterator, Optional
from aipou.models import ChatMessage, AIResponse, StreamChunk


class BaseProvider(ABC):
    """Abstract base class that every AI provider must implement."""

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.default_model = kwargs.get("default_model", self._default_model())
        self.timeout = kwargs.get("timeout", 30)

    @abstractmethod
    def _default_model(self) -> str:
        """Return the default model identifier for this provider."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: Optional[str] = None,
        **kwargs,
    ) -> AIResponse:
        """Send a chat request and return a complete response."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: Optional[str] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Send a chat request and yield response chunks as they arrive."""
        ...

    def _resolve_model(self, model: Optional[str]) -> str:
        return model or self.default_model

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.default_model}>"