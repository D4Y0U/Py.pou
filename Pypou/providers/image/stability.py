"""Stability AI — Stable Diffusion Ultra, Core, SD3.

Clé API : https://platform.stability.ai/account/keys
"""

from typing import Literal, Optional
import requests

from aipou.models import ImageResponse
from aipou.exceptions import AuthenticationError, APIError, RateLimitError

_BASE_URL = "https://api.stability.ai/v2beta"

Model = Literal["ultra", "core", "sd3-large", "sd3-large-turbo", "sd3-medium"]


class StabilityProvider:
    """Génération d'images avec Stability AI (Stable Diffusion 3)."""

    def __init__(self, api_key: str, timeout: int = 120):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "image/*"}

    def generate(
        self,
        prompt: str,
        *,
        model: Model = "core",
        negative_prompt: Optional[str] = None,
        aspect_ratio: Literal["1:1", "16:9", "9:16", "3:2", "2:3", "4:5", "5:4", "21:9"] = "1:1",
        output_format: Literal["jpeg", "png", "webp"] = "jpeg",
        seed: int = 0,
        style_preset: Optional[str] = None,
    ) -> ImageResponse:
        """
        Génère une image avec Stable Diffusion 3.

        Parameters
        ----------
        model : str
            "ultra" (meilleure qualité), "core" (équilibré), "sd3-large-turbo" (rapide).
        negative_prompt : str, optional
            Ce que l'image ne doit PAS contenir.
        aspect_ratio : str
            Ratio largeur:hauteur. "16:9" pour paysage, "9:16" pour portrait.
        style_preset : str, optional
            Parmi : "3d-model", "analog-film", "anime", "cinematic", "comic-book",
            "digital-art", "fantasy-art", "isometric", "line-art", "low-poly",
            "neon-punk", "origami", "photographic", "pixel-art", "tile-texture".
        """
        endpoint_map = {
            "ultra": f"{_BASE_URL}/stable-image/generate/ultra",
            "core": f"{_BASE_URL}/stable-image/generate/core",
            "sd3-large": f"{_BASE_URL}/stable-image/generate/sd3",
            "sd3-large-turbo": f"{_BASE_URL}/stable-image/generate/sd3",
            "sd3-medium": f"{_BASE_URL}/stable-image/generate/sd3",
        }
        url = endpoint_map.get(model, endpoint_map["core"])

        form: dict = {
            "prompt": (None, prompt),
            "aspect_ratio": (None, aspect_ratio),
            "output_format": (None, output_format),
            "seed": (None, str(seed)),
        }
        if negative_prompt:
            form["negative_prompt"] = (None, negative_prompt)
        if style_preset:
            form["style_preset"] = (None, style_preset)
        if model.startswith("sd3"):
            form["model"] = (None, model)

        resp = requests.post(
            url, headers=self._headers(), files=form, timeout=self.timeout
        )
        self._raise_for_status(resp)

        return ImageResponse(
            url=None,
            data=resp.content,
            model=model,
            provider="stability",
            prompt=prompt,
            raw={"content_type": resp.headers.get("content-type")},
        )

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Stability AI invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Stability AI atteinte.")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("errors", [resp.text])[0]
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur Stability AI {resp.status_code} : {detail}")