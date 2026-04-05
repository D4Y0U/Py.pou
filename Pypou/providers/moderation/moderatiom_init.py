"""aipou.moderation — détection de contenu sensible.

Deux providers :
  - OpenAI Moderation API  (gratuit, rapide)
  - Anthropic via prompt   (utilise Claude pour une analyse plus nuancée)
"""

from typing import Optional
import requests

from aipou.models import ModerationResult
from aipou.exceptions import AuthenticationError, APIError, RateLimitError


# ---------------------------------------------------------------------------
# OpenAI Moderation
# ---------------------------------------------------------------------------

class OpenAIModerationProvider:
    """
    Modération via l'API OpenAI — gratuite, ultra-rapide.

    Catégories détectées :
      harassment, harassment/threatening, hate, hate/threatening,
      self-harm, self-harm/instructions, self-harm/intent,
      sexual, sexual/minors, violence, violence/graphic
    """

    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout

    def check(
        self,
        text: str,
        *,
        model: str = "omni-moderation-latest",
    ) -> ModerationResult:
        """
        Analyse un texte et retourne les catégories de contenu déclenchées.

        Parameters
        ----------
        model : str
            "omni-moderation-latest" (plus précis) ou "text-moderation-latest" (legacy).
        """
        resp = requests.post(
            "https://api.openai.com/v1/moderations",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"input": text, "model": model},
            timeout=self.timeout,
        )
        if resp.status_code == 401:
            raise AuthenticationError("Clé API OpenAI invalide.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur OpenAI Moderation {resp.status_code}: {resp.text}")

        data = resp.json()
        result = data["results"][0]
        return ModerationResult(
            flagged=result["flagged"],
            categories=result["categories"],
            scores=result["category_scores"],
            provider="openai-moderation",
            raw=data,
        )

    def check_batch(self, texts: list[str], **kwargs) -> list[ModerationResult]:
        """Analyse plusieurs textes en une seule requête."""
        resp = requests.post(
            "https://api.openai.com/v1/moderations",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"input": texts, **kwargs},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise APIError(f"Erreur OpenAI Moderation: {resp.text}")
        data = resp.json()
        return [
            ModerationResult(
                flagged=r["flagged"],
                categories=r["categories"],
                scores=r["category_scores"],
                provider="openai-moderation",
            )
            for r in data["results"]
        ]


# ---------------------------------------------------------------------------
# Anthropic Moderation (via Claude)
# ---------------------------------------------------------------------------

class AnthropicModerationProvider:
    """
    Modération nuancée via Claude — plus contextuelle qu'une API de classification.

    Utile pour des cas complexes : sarcasme, contexte éducatif/médical,
    contenu ambigu, langues moins courantes.
    """

    _SYSTEM = """You are a content moderation assistant. Analyze the given text and respond ONLY with a JSON object (no markdown, no explanation) with this exact structure:
{
  "flagged": true/false,
  "categories": {
    "harassment": true/false,
    "hate_speech": true/false,
    "violence": true/false,
    "sexual": true/false,
    "self_harm": true/false,
    "spam": true/false,
    "misinformation": true/false
  },
  "scores": {
    "harassment": 0.0-1.0,
    "hate_speech": 0.0-1.0,
    "violence": 0.0-1.0,
    "sexual": 0.0-1.0,
    "self_harm": 0.0-1.0,
    "spam": 0.0-1.0,
    "misinformation": 0.0-1.0
  },
  "reason": "brief explanation if flagged, empty string otherwise"
}"""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001", timeout: int = 15):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def check(self, text: str) -> ModerationResult:
        """
        Analyse un texte avec Claude pour une modération contextuelle.

        Utilise claude-haiku par défaut (rapide + économique).
        """
        import json

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 256,
                "system": self._SYSTEM,
                "messages": [{"role": "user", "content": f"Analyze this text:\n\n{text}"}],
            },
            timeout=self.timeout,
        )
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Anthropic invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Anthropic atteinte.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Anthropic Moderation {resp.status_code}: {resp.text}")

        raw_text = resp.json()["content"][0]["text"]
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback si Claude sort du JSON imparfait
            payload = {"flagged": False, "categories": {}, "scores": {}}

        return ModerationResult(
            flagged=payload.get("flagged", False),
            categories=payload.get("categories", {}),
            scores=payload.get("scores", {}),
            provider="anthropic-moderation",
            raw=payload,
        )


# ---------------------------------------------------------------------------
# Client unifié
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai": OpenAIModerationProvider,
    "anthropic": AnthropicModerationProvider,
}


class ModerationClient:
    """
    Client de modération unifié.

    Exemple
    -------
        from aipou.moderation import ModerationClient

        client = ModerationClient("openai", api_key="sk-...")
        result = client.check("Ce texte est inoffensif.")
        print(result.flagged)              # False
        print(result.triggered_categories) # []
    """

    def __init__(self, provider: str, api_key: str, **kwargs):
        key = provider.lower()
        if key not in _PROVIDERS:
            raise ValueError(f"Provider modération inconnu. Disponibles : {list(_PROVIDERS)}")
        self._provider = _PROVIDERS[key](api_key=api_key, **kwargs)
        self.provider_name = key

    def check(self, text: str, **kwargs) -> ModerationResult:
        return self._provider.check(text, **kwargs)