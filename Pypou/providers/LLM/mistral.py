"""Mistral AI provider (mistral-large, mistral-medium, mistral-small, etc.)."""

from typing import Iterator, Optional
import requests

from aipou.providers.base import BaseProvider
from aipou.models import ChatMessage, AIResponse, StreamChunk, TokenUsage
from aipou.exceptions import AuthenticationError, APIError, RateLimitError


_BASE_URL = "https://api.mistral.ai/v1"


class MistralProvider(BaseProvider):
    """Provider for Mistral AI chat completion APIs."""

    def _default_model(self) -> str:
        return "mistral-large-latest"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(
        self, messages: list[ChatMessage], system: Optional[str]
    ) -> list[dict]:
        result = []
        if system:
            result.append({"role": "system", "content": system})
        result.extend(m.to_dict() for m in messages)
        return result

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
        payload = {
            "model": resolved_model,
            "messages": self._build_messages(messages, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        resp = requests.post(
            f"{_BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        data = resp.json()

        choice = data["choices"][0]
        usage_data = data.get("usage", {})
        return AIResponse(
            content=choice["message"]["content"],
            model=data.get("model", resolved_model),
            provider="mistral",
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
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
        payload = {
            "model": resolved_model,
            "messages": self._build_messages(messages, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }
        with requests.post(
            f"{_BASE_URL}/chat/completions",
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
                if line == "[DONE]":
                    yield StreamChunk(delta="", model=resolved_model, provider="mistral", is_final=True)
                    return
                try:
                    data = json.loads(line)
                    choice = data["choices"][0]
                    delta = choice.get("delta", {}).get("content") or ""
                    finish_reason = choice.get("finish_reason")
                    yield StreamChunk(
                        delta=delta,
                        model=data.get("model", resolved_model),
                        provider="mistral",
                        finish_reason=finish_reason,
                        is_final=finish_reason is not None,
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Mistral invalide ou manquante.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Mistral atteinte. Réessayez plus tard.")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur Mistral {resp.status_code} : {detail}")