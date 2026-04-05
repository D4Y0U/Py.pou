"""Ollama provider — modèles IA en local, aucune donnée envoyée sur Internet.

Prérequis : Ollama installé et en cours d'exécution (https://ollama.com)
  $ ollama pull llama3.2        # télécharge un modèle
  $ ollama serve                # lance le serveur (par défaut sur :11434)

Modèles populaires : llama3.2, mistral, phi4, gemma3, qwen2.5, deepseek-r1
"""

from typing import Iterator, Optional
import requests

from aipou.providers.base import BaseProvider
from aipou.models import ChatMessage, AIResponse, StreamChunk, TokenUsage
from aipou.exceptions import APIError


_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """Provider pour Ollama — inférence locale sans clé API."""

    def __init__(self, api_key: str = "", **kwargs):
        # Ollama n'a pas besoin de clé API — on accepte une chaîne vide
        self.base_url = kwargs.pop("base_url", _DEFAULT_BASE_URL).rstrip("/")
        super().__init__(api_key=api_key, **kwargs)

    def _default_model(self) -> str:
        return "llama3.2"

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

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
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs.get("options", {}),
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            raise APIError(
                f"Impossible de joindre Ollama sur {self.base_url}. "
                "Vérifiez qu'Ollama est bien lancé (`ollama serve`)."
            )
        self._raise_for_status(resp)
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        usage_data = data.get("usage", {})

        return AIResponse(
            content=content,
            model=data.get("model", resolved_model),
            provider="ollama",
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
            ),
            finish_reason=data.get("done_reason"),
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
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs.get("options", {}),
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            raise APIError(
                f"Impossible de joindre Ollama sur {self.base_url}. "
                "Vérifiez qu'Ollama est bien lancé (`ollama serve`)."
            )
        self._raise_for_status(resp)

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            delta = event.get("message", {}).get("content", "")
            done = event.get("done", False)
            yield StreamChunk(
                delta=delta,
                model=resolved_model,
                provider="ollama",
                finish_reason="stop" if done else None,
                is_final=done,
            )

    def list_models(self) -> list[str]:
        """Retourne la liste des modèles disponibles localement."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            raise APIError(f"Impossible de lister les modèles Ollama : {exc}") from exc

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur Ollama {resp.status_code} : {detail}")