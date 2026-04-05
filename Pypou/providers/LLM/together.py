"""Together AI provider — accès à +100 modèles open-source hébergés.

Clé API : https://api.together.xyz/settings/api-keys
Format  : compatible OpenAI

Modèles populaires :
  - meta-llama/Llama-3.3-70B-Instruct-Turbo
  - mistralai/Mixtral-8x22B-Instruct-v0.1
  - Qwen/Qwen2.5-72B-Instruct-Turbo
  - deepseek-ai/DeepSeek-R1
  - google/gemma-2-27b-it
"""

from typing import Iterator, Optional
import requests

from aipou.providers.base import BaseProvider
from aipou.models import ChatMessage, AIResponse, StreamChunk, TokenUsage
from aipou.exceptions import AuthenticationError, APIError, RateLimitError


_BASE_URL = "https://api.together.xyz/v1"


class TogetherProvider(BaseProvider):
    """Provider pour Together AI (format OpenAI-compatible, +100 modèles open-source)."""

    def _default_model(self) -> str:
        return "meta-llama/Llama-3.3-70B-Instruct-Turbo"

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
            provider="together",
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
                    yield StreamChunk(delta="", model=resolved_model, provider="together", is_final=True)
                    return
                try:
                    data = json.loads(line)
                    choice = data["choices"][0]
                    delta = choice.get("delta", {}).get("content") or ""
                    finish_reason = choice.get("finish_reason")
                    yield StreamChunk(
                        delta=delta,
                        model=data.get("model", resolved_model),
                        provider="together",
                        finish_reason=finish_reason,
                        is_final=finish_reason is not None,
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

    def list_models(self, model_type: str = "chat") -> list[dict]:
        """
        Liste les modèles disponibles sur Together AI.

        Parameters
        ----------
        model_type : str
            Filtre par type : "chat", "language", "image", "embedding", "code".

        Returns
        -------
        list[dict]
            Chaque entrée contient au moins "id", "display_name", "context_length".
        """
        resp = requests.get(
            f"{_BASE_URL}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        models = resp.json()
        if model_type:
            models = [m for m in models if m.get("type", "").lower() == model_type]
        return [
            {
                "id": m["id"],
                "name": m.get("display_name", m["id"]),
                "context_length": m.get("context_length", "?"),
            }
            for m in models
        ]

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Together AI invalide ou manquante.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Together AI atteinte. Réessayez plus tard.")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur Together AI {resp.status_code} : {detail}")