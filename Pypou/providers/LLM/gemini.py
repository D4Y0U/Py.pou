"""Google Gemini provider (gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash…).

Authentication : API key via Google AI Studio (https://aistudio.google.com/app/apikey)

Particularités Gemini vs OpenAI/Anthropic :
  - Les rôles sont "user" / "model" (pas "assistant")
  - Le system prompt passe dans un champ "system_instruction" séparé
  - La réponse est dans candidates[0].content.parts[].text
  - Le streaming retourne des événements JSON ligne par ligne (format SSE allégé)
"""

from typing import Iterator, Optional
import requests

from aipou.providers.base import BaseProvider
from aipou.models import ChatMessage, AIResponse, StreamChunk, TokenUsage
from aipou.exceptions import AuthenticationError, APIError, RateLimitError


_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseProvider):
    """Provider pour l'API Google Gemini (Generative Language API)."""

    def _default_model(self) -> str:
        return "gemini-2.0-flash"

    def _endpoint(self, model: str, streaming: bool = False) -> str:
        action = "streamGenerateContent" if streaming else "generateContent"
        return f"{_BASE_URL}/{model}:{action}?key={self.api_key}"

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _build_contents(self, messages: list[ChatMessage]) -> list[dict]:
        """Convertit les ChatMessage au format Gemini (role: user/model)."""
        contents = []
        for m in messages:
            if m.role == "system":
                continue  # Géré séparément via system_instruction
            role = "model" if m.role == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": m.content}],
            })
        return contents

    def _extract_system(
        self, messages: list[ChatMessage], system: Optional[str]
    ) -> Optional[str]:
        """Fusionne les messages système et le paramètre system."""
        parts = [m.content for m in messages if m.role == "system"]
        if system:
            parts.insert(0, system)
        return "\n\n".join(parts) if parts else None

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        system: Optional[str],
        **kwargs,
    ) -> dict:
        payload: dict = {
            "contents": self._build_contents(messages),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                **kwargs.get("generationConfig", {}),
            },
        }
        system_prompt = self._extract_system(messages, system)
        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }
        return payload

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
        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            **kwargs,
        )
        resp = requests.post(
            self._endpoint(resolved_model),
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        data = resp.json()

        content = self._extract_text(data)
        usage_data = data.get("usageMetadata", {})

        return AIResponse(
            content=content,
            model=resolved_model,
            provider="gemini",
            usage=TokenUsage(
                prompt_tokens=usage_data.get("promptTokenCount", 0),
                completion_tokens=usage_data.get("candidatesTokenCount", 0),
            ),
            finish_reason=self._extract_finish_reason(data),
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
        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            **kwargs,
        )
        # Le streaming Gemini ajoute &alt=sse pour forcer le format SSE
        url = self._endpoint(resolved_model, streaming=True) + "&alt=sse"

        with requests.post(
            url,
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

                delta = self._extract_text(event)
                finish_reason = self._extract_finish_reason(event)
                is_final = finish_reason is not None

                yield StreamChunk(
                    delta=delta,
                    model=resolved_model,
                    provider="gemini",
                    finish_reason=finish_reason,
                    is_final=is_final,
                )

    # ------------------------------------------------------------------
    # Helpers d'extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extrait le texte depuis candidates[0].content.parts."""
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _extract_finish_reason(data: dict) -> Optional[str]:
        try:
            return data["candidates"][0].get("finishReason")
        except (KeyError, IndexError):
            return None

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 400:
            try:
                detail = resp.json()["error"]["message"]
            except Exception:
                detail = resp.text
            raise APIError(f"Requête invalide Gemini : {detail}")
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Google invalide ou manquante.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Gemini atteinte. Réessayez plus tard.")
        if resp.status_code >= 400:
            try:
                detail = resp.json()["error"]["message"]
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur Gemini {resp.status_code} : {detail}")