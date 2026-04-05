"""Anthropic provider (claude-3-5-sonnet, claude-3-opus, etc.)."""

from typing import Iterator, Optional
import requests

from aipou.providers.base import BaseProvider
from aipou.models import ChatMessage, AIResponse, StreamChunk, TokenUsage
from aipou.exceptions import AuthenticationError, APIError, RateLimitError


_BASE_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude APIs."""

    def _default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Anthropic doesn't accept 'system' role inside messages array."""
        return [m.to_dict() for m in messages if m.role != "system"]

    def _extract_system(self, messages: list[ChatMessage], system: Optional[str]) -> Optional[str]:
        """Merge explicit system param with any system-role messages."""
        system_messages = [m.content for m in messages if m.role == "system"]
        if system:
            system_messages.insert(0, system)
        return "\n\n".join(system_messages) if system_messages else None

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
        resolved_model = self._resolve_model(model)
        system_prompt = self._extract_system(messages, system)

        payload: dict = {
            "model": resolved_model,
            "messages": self._build_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = requests.post(
            f"{_BASE_URL}/messages",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        data = resp.json()

        content = "".join(
            block.get("text", "") for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage_data = data.get("usage", {})
        return AIResponse(
            content=content,
            model=data.get("model", resolved_model),
            provider="anthropic",
            usage=TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
            ),
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

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
        import json

        resolved_model = self._resolve_model(model)
        system_prompt = self._extract_system(messages, system)

        payload: dict = {
            "model": resolved_model,
            "messages": self._build_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }
        if system_prompt:
            payload["system"] = system_prompt

        with requests.post(
            f"{_BASE_URL}/messages",
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            self._raise_for_status(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "content_block_delta":
                    delta = event.get("delta", {}).get("text", "")
                    yield StreamChunk(delta=delta, model=resolved_model, provider="anthropic")
                elif event_type == "message_delta":
                    stop_reason = event.get("delta", {}).get("stop_reason")
                    yield StreamChunk(
                        delta="",
                        model=resolved_model,
                        provider="anthropic",
                        finish_reason=stop_reason,
                        is_final=True,
                    )

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Anthropic invalide ou manquante.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Anthropic atteinte. Réessayez plus tard.")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur Anthropic {resp.status_code} : {detail}")