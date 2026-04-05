"""
aipou.client — Unified AI client.

Usage:
    from aipou import AIClient, ChatMessage

    client = AIClient("openai", api_key="sk-...")
    response = client.chat("Explique-moi les transformers en 3 phrases.")
    print(response)

    # Streaming
    for chunk in client.stream("Raconte une histoire courte."):
        print(chunk.delta, end="", flush=True)
"""

from typing import Iterator, Optional, Union
import time
import logging

from aipou.models import ChatMessage, AIResponse, StreamChunk, TokenUsage
from aipou.providers.base import BaseProvider
from aipou.exceptions import APIError, RateLimitError, ProviderNotFoundError

logger = logging.getLogger(__name__)

# Registry of built-in providers
_PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {}

def _load_providers():
    """Lazy-load providers to keep imports fast."""
    from aipou.providers.openai import OpenAIProvider
    from aipou.providers.anthropic import AnthropicProvider
    from aipou.providers.mistral import MistralProvider
    from aipou.providers.gemini import GeminiProvider
    _PROVIDER_REGISTRY.update({
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mistral": MistralProvider,
        "gemini": GeminiProvider,
    })


class AIClient:
    """
    Provider-agnostic AI client.

    Parameters
    ----------
    provider : str
        One of "openai", "anthropic", "mistral".
    api_key : str
        Your API key for the chosen provider.
    model : str, optional
        Override the default model for this provider.
    max_retries : int
        Number of automatic retries on rate-limit / transient errors (default 3).
    retry_delay : float
        Base delay in seconds between retries (exponential backoff, default 1.0).
    **provider_kwargs
        Extra keyword arguments forwarded to the provider constructor.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        *,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **provider_kwargs,
    ):
        if not _PROVIDER_REGISTRY:
            _load_providers()

        provider_key = provider.lower()
        if provider_key not in _PROVIDER_REGISTRY:
            raise ProviderNotFoundError(
                f"Provider '{provider}' inconnu. Disponibles : {list(_PROVIDER_REGISTRY)}"
            )

        self._provider: BaseProvider = _PROVIDER_REGISTRY[provider_key](
            api_key=api_key,
            default_model=model,
            **provider_kwargs,
        )
        self.provider_name = provider_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Cumulative token tracking across all calls in this session
        self._total_usage = TokenUsage()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: Union[str, list[ChatMessage]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        **kwargs,
    ) -> AIResponse:
        """
        Send a message and get a complete response.

        Parameters
        ----------
        prompt : str or list[ChatMessage]
            A simple string prompt OR a pre-built list of messages.
        system : str, optional
            System instructions (injected as the first message / system block).
        history : list[ChatMessage], optional
            Previous conversation turns prepended before `prompt`.
        """
        messages = self._build_messages(prompt, history)
        response = self._with_retry(
            self._provider.chat,
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            **kwargs,
        )
        self._accumulate_usage(response.usage)
        logger.info(
            "[%s] %s | tokens: %s",
            self.provider_name,
            response.model,
            response.usage,
        )
        return response

    def stream(
        self,
        prompt: Union[str, list[ChatMessage]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: Optional[str] = None,
        history: Optional[list[ChatMessage]] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """
        Stream the response token by token.

        Yields StreamChunk objects. Use `chunk.delta` for the new text,
        `chunk.is_final` to detect the last chunk.

        Example
        -------
            for chunk in client.stream("Bonjour !"):
                print(chunk.delta, end="", flush=True)
        """
        messages = self._build_messages(prompt, history)
        yield from self._provider.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            **kwargs,
        )

    def stream_text(
        self,
        prompt: Union[str, list[ChatMessage]],
        **kwargs,
    ) -> Iterator[str]:
        """Convenience wrapper — yields plain strings instead of StreamChunk."""
        for chunk in self.stream(prompt, **kwargs):
            if chunk.delta:
                yield chunk.delta

    @property
    def session_usage(self) -> TokenUsage:
        """Total token usage accumulated since this client was created."""
        return self._total_usage

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: Union[str, list[ChatMessage]],
        history: Optional[list[ChatMessage]],
    ) -> list[ChatMessage]:
        if isinstance(prompt, str):
            messages = list(history or []) + [ChatMessage.user(prompt)]
        else:
            messages = list(history or []) + list(prompt)
        return messages

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens

    def _with_retry(self, fn, *args, **kwargs):
        last_exc: Exception = APIError("Unreachable")
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except RateLimitError as exc:
                last_exc = exc
                wait = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit atteint (tentative %d/%d). Pause de %.1fs.",
                    attempt, self.max_retries, wait,
                )
                time.sleep(wait)
            except APIError:
                raise  # Don't retry non-transient errors
        raise last_exc

    def __repr__(self) -> str:
        return f"<AIClient provider={self.provider_name} model={self._provider.default_model}>"