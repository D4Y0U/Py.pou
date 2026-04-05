"""DALL·E 3 — génération d'images via l'API OpenAI.

Clé API : même clé que le provider OpenAI chat.
"""

import base64
from typing import Literal, Optional
import requests

from aipou.models import ImageResponse
from aipou.exceptions import AuthenticationError, APIError, RateLimitError

_BASE_URL = "https://api.openai.com/v1"


class DalleProvider:
    """Génération d'images avec DALL·E 3."""

    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate(
        self,
        prompt: str,
        *,
        model: str = "dall-e-3",
        size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] = "vivid",
        n: int = 1,
        response_format: Literal["url", "b64_json"] = "url",
    ) -> ImageResponse:
        """
        Génère une image depuis un prompt texte.

        Parameters
        ----------
        prompt : str
            Description de l'image à générer (max 4000 caractères pour DALL·E 3).
        size : str
            "1024x1024" (carré), "1792x1024" (paysage), "1024x1792" (portrait).
        quality : str
            "standard" ou "hd" (plus de détails, plus cher).
        style : str
            "vivid" (couleurs saturées, dramatique) ou "natural" (réaliste).
        response_format : str
            "url" (lien temporaire ~1h) ou "b64_json" (données brutes).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "style": style,
            "response_format": response_format,
        }
        resp = requests.post(
            f"{_BASE_URL}/images/generations",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        data = resp.json()

        item = data["data"][0]
        image_data = base64.b64decode(item["b64_json"]) if "b64_json" in item else None

        return ImageResponse(
            url=item.get("url"),
            data=image_data,
            model=model,
            provider="dalle",
            prompt=prompt,
            revised_prompt=item.get("revised_prompt"),
            raw=data,
        )

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API OpenAI invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit DALL·E atteinte.")
        if resp.status_code >= 400:
            try:
                detail = resp.json()["error"]["message"]
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur DALL·E {resp.status_code} : {detail}")