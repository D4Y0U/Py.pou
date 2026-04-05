"""Cohere provider — Command R+, RAG natif, et embeddings.

Clé API : https://dashboard.cohere.com/api-keys

Particularités Cohere :
  - Format de message différent : "CHATBOT" / "USER" (pas role/content)
  - Le message courant est séparé du historique (paramètre "message")
  - Support natif des embeddings vectoriels via embed()
  - Modèles chat : command-r-plus, command-r, command-light
  - Modèles embed : embed-multilingual-v3.0, embed-english-v3.0
"""

from typing import Iterator, Optional
import requests

from aipou.providers.base import BaseProvider
from aipou.models import ChatMessage, AIResponse, StreamChunk, TokenUsage
from aipou.exceptions import AuthenticationError, APIError, RateLimitError


_BASE_URL = "https://api.cohere.com/v2"


class CohereProvider(BaseProvider):
    """Provider pour l'API Cohere (chat + embeddings)."""

    def _default_model(self) -> str:
        return "command-r-plus"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Client-Name": "aipou",
        }

    def _split_messages(
        self, messages: list[ChatMessage], system: Optional[str]
    ) -> tuple[Optional[str], list[dict], str]:
        """
        Sépare les messages en :
          - system_prompt (str | None)
          - chat_history  (list de tours précédents)
          - message       (dernier message utilisateur)
        """
        # Collecte le system
        system_parts = [m.content for m in messages if m.role == "system"]
        if system:
            system_parts.insert(0, system)
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # Filtre les messages non-system
        non_system = [m for m in messages if m.role != "system"]

        if not non_system:
            raise APIError("Au moins un message utilisateur est requis.")

        # Le dernier message doit être de l'utilisateur
        last = non_system[-1]
        if last.role != "user":
            raise APIError("Le dernier message doit être de rôle 'user' pour Cohere.")

        history = []
        for m in non_system[:-1]:
            role = "CHATBOT" if m.role == "assistant" else "USER"
            history.append({"role": role, "content": m.content})

        return system_prompt, history, last.content

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
        system_prompt, history, message = self._split_messages(messages, system)

        payload: dict = {
            "model": resolved_model,
            "message": message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if history:
            payload["chat_history"] = history
        if system_prompt:
            payload["preamble"] = system_prompt

        resp = requests.post(
            f"{_BASE_URL}/chat",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        data = resp.json()

        content = data.get("message", {}).get("content", [{}])[0].get("text", "")
        usage_data = data.get("usage", {}).get("tokens", {})

        return AIResponse(
            content=content,
            model=resolved_model,
            provider="cohere",
            usage=TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
            ),
            finish_reason=data.get("finish_reason"),
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
        system_prompt, history, message = self._split_messages(messages, system)

        payload: dict = {
            "model": resolved_model,
            "message": message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }
        if history:
            payload["chat_history"] = history
        if system_prompt:
            payload["preamble"] = system_prompt

        with requests.post(
            f"{_BASE_URL}/chat",
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            self._raise_for_status(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "content-delta":
                    delta = event.get("delta", {}).get("message", {}).get("content", {}).get("text", "")
                    yield StreamChunk(delta=delta, model=resolved_model, provider="cohere")
                elif event_type == "message-end":
                    yield StreamChunk(
                        delta="",
                        model=resolved_model,
                        provider="cohere",
                        finish_reason=event.get("delta", {}).get("finish_reason"),
                        is_final=True,
                    )

    def embed(
        self,
        texts: list[str],
        *,
        model: str = "embed-multilingual-v3.0",
        input_type: str = "search_document",
    ) -> list[list[float]]:
        """
        Génère des embeddings vectoriels pour une liste de textes.

        Parameters
        ----------
        texts : list[str]
            Les textes à transformer en vecteurs.
        model : str
            Modèle d'embedding (embed-multilingual-v3.0 ou embed-english-v3.0).
        input_type : str
            "search_document" pour indexation, "search_query" pour les requêtes,
            "classification" ou "clustering" pour d'autres usages.

        Returns
        -------
        list[list[float]]
            Un vecteur par texte.
        """
        payload = {
            "model": model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        resp = requests.post(
            f"{_BASE_URL}/embed",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return resp.json()["embeddings"]["float"]

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Cohere invalide ou manquante.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Cohere atteinte. Réessayez plus tard.")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur Cohere {resp.status_code} : {detail}")