"""ElevenLabs — TTS ultra-réaliste avec clonage de voix.

Clé API : https://elevenlabs.io/app/settings/api-keys
Voix par défaut disponibles sans abonnement :
  - Rachel   : 21m00Tcm4TlvDq8ikWAM
  - Domi     : AZnzlk1XvdvUeBnXmlld
  - Bella    : EXAVITQu4vr4xnSDxMaL
  - Antoni   : ErXwobaYiN019PkySvjV
  - Elli     : MF3mGyEYCl7XYWbV9V6O
  - Josh     : TxGEqnHWrfWFTfGW9XjX
  - Arnold   : VR6AewLTigWG4xSOukaG
  - Adam     : pNInz6obpgDQGcFmaJgB
  - Sam      : yoZ06aMxZJJ28mfd3POQ
"""

from typing import Iterator, Literal, Optional
import requests

from aipou.models import TTSResponse
from aipou.exceptions import AuthenticationError, APIError, RateLimitError

_BASE_URL = "https://api.elevenlabs.io/v1"

# Voix Rachel par défaut — naturelle, polyvalente
_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"


class ElevenLabsProvider:
    """Synthèse vocale ultra-réaliste avec ElevenLabs."""

    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self, content_type: str = "application/json") -> dict:
        return {"xi-api-key": self.api_key, "Content-Type": content_type}

    def speak(
        self,
        text: str,
        *,
        voice_id: str = _DEFAULT_VOICE,
        model: str = "eleven_multilingual_v2",
        output_format: Literal["mp3_44100_128", "mp3_44100_64", "pcm_16000", "pcm_24000"] = "mp3_44100_128",
        stability: float = 0.5,                 # 0.0 (expressif) → 1.0 (stable)
        similarity_boost: float = 0.75,
        style: float = 0.0,                     # Exagération du style (0-1)
        use_speaker_boost: bool = True,
    ) -> TTSResponse:
        """
        Convertit du texte en audio ultra-réaliste.

        Parameters
        ----------
        voice_id : str
            ID de la voix (voir liste en en-tête de fichier ou list_voices()).
        model : str
            "eleven_multilingual_v2" (multilingue, recommandé),
            "eleven_turbo_v2_5" (rapide, faible latence),
            "eleven_monolingual_v1" (anglais uniquement, économique).
        stability : float
            0.0 = très expressif/émotionnel, 1.0 = monotone/stable.
        """
        payload = {
            "text": text,
            "model_id": model,
            "output_format": output_format,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": use_speaker_boost,
            },
        }
        resp = requests.post(
            f"{_BASE_URL}/text-to-speech/{voice_id}",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        fmt = "mp3" if output_format.startswith("mp3") else "pcm"
        return TTSResponse(audio=resp.content, format=fmt, provider="elevenlabs")

    def stream(
        self,
        text: str,
        *,
        voice_id: str = _DEFAULT_VOICE,
        model: str = "eleven_turbo_v2_5",        # Turbo recommandé pour le streaming
        output_format: str = "mp3_44100_128",
        chunk_size: int = 4096,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
    ) -> Iterator[bytes]:
        """Stream l'audio par chunks pour une lecture en temps réel."""
        payload = {
            "text": text,
            "model_id": model,
            "output_format": output_format,
            "voice_settings": {"stability": stability, "similarity_boost": similarity_boost},
        }
        with requests.post(
            f"{_BASE_URL}/text-to-speech/{voice_id}/stream",
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            self._raise_for_status(resp)
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    yield chunk

    def list_voices(self) -> list[dict]:
        """Retourne la liste de toutes les voix disponibles (prédéfinies + clonées)."""
        resp = requests.get(
            f"{_BASE_URL}/voices",
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return [
            {"id": v["voice_id"], "name": v["name"], "category": v.get("category", "?")}
            for v in resp.json().get("voices", [])
        ]

    def clone_voice(self, name: str, audio_files: list[str], description: str = "") -> str:
        """
        Clone une voix à partir d'échantillons audio.

        Parameters
        ----------
        name : str
            Nom de la voix clonée.
        audio_files : list[str]
            Chemins vers les fichiers audio d'échantillons (5-25 fichiers, ~1min chacun).

        Returns
        -------
        str
            L'ID de la nouvelle voix clonée.
        """
        files = [("files", (f, open(f, "rb"), "audio/mpeg")) for f in audio_files]
        data = {"name": name, "description": description}
        resp = requests.post(
            f"{_BASE_URL}/voices/add",
            headers={"xi-api-key": self.api_key},
            files=files,
            data=data,
            timeout=120,
        )
        self._raise_for_status(resp)
        return resp.json()["voice_id"]

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API ElevenLabs invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit ElevenLabs atteinte.")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur ElevenLabs {resp.status_code} : {detail}")