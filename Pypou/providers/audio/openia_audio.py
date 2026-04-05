"""OpenAI Audio — Whisper (transcription) + TTS (text-to-speech).

Clé API : même clé que le provider OpenAI chat.
"""

from pathlib import Path
from typing import IO, Literal, Optional, Union
import requests

from aipou.models import TranscriptionResponse, TTSResponse
from aipou.exceptions import AuthenticationError, APIError, RateLimitError

_BASE_URL = "https://api.openai.com/v1"

WhisperModel = Literal["whisper-1"]
TTSModel = Literal["tts-1", "tts-1-hd"]
Voice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class WhisperProvider:
    """Transcription audio → texte avec OpenAI Whisper."""

    def __init__(self, api_key: str, timeout: int = 120):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def transcribe(
        self,
        audio: Union[str, Path, bytes, IO],
        *,
        model: WhisperModel = "whisper-1",
        language: Optional[str] = None,       # Code ISO-639-1, ex. "fr", "en"
        prompt: Optional[str] = None,          # Contexte pour améliorer la précision
        response_format: Literal["json", "text", "srt", "vtt"] = "json",
        temperature: float = 0.0,
    ) -> TranscriptionResponse:
        """
        Transcrit un fichier audio en texte.

        Parameters
        ----------
        audio : str | Path | bytes | IO
            Chemin vers un fichier audio, bytes bruts, ou objet fichier.
            Formats acceptés : mp3, mp4, mpeg, mpga, m4a, wav, webm.
        language : str, optional
            Code langue ISO-639-1 pour améliorer la précision (ex. "fr").
        prompt : str, optional
            Texte de contexte pour guider la transcription (vocabulaire spécifique).
        """
        if isinstance(audio, (str, Path)):
            p = Path(audio)
            audio_file = ("audio" + p.suffix, p.read_bytes(), "audio/mpeg")
        elif isinstance(audio, bytes):
            audio_file = ("audio.mp3", audio, "audio/mpeg")
        else:
            audio_file = ("audio.mp3", audio, "audio/mpeg")

        files = {"file": audio_file}
        data: dict = {"model": model, "response_format": response_format, "temperature": temperature}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        resp = requests.post(
            f"{_BASE_URL}/audio/transcriptions",
            headers=self._headers(),
            files=files,
            data=data,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)

        if response_format == "json":
            payload = resp.json()
            return TranscriptionResponse(
                text=payload["text"],
                language=payload.get("language"),
                duration=payload.get("duration"),
                provider="openai-whisper",
                raw=payload,
            )
        return TranscriptionResponse(text=resp.text, provider="openai-whisper")

    def translate(
        self,
        audio: Union[str, Path, bytes, IO],
        *,
        model: WhisperModel = "whisper-1",
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcrit ET traduit l'audio directement en anglais."""
        if isinstance(audio, (str, Path)):
            p = Path(audio)
            audio_file = ("audio" + p.suffix, p.read_bytes(), "audio/mpeg")
        elif isinstance(audio, bytes):
            audio_file = ("audio.mp3", audio, "audio/mpeg")
        else:
            audio_file = ("audio.mp3", audio, "audio/mpeg")

        files = {"file": audio_file}
        data: dict = {"model": model}
        if prompt:
            data["prompt"] = prompt

        resp = requests.post(
            f"{_BASE_URL}/audio/translations",
            headers=self._headers(),
            files=files,
            data=data,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        payload = resp.json()
        return TranscriptionResponse(text=payload["text"], provider="openai-whisper", raw=payload)

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API OpenAI invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Whisper atteinte.")
        if resp.status_code >= 400:
            try:
                detail = resp.json()["error"]["message"]
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur Whisper {resp.status_code} : {detail}")


class OpenAITTSProvider:
    """Synthèse vocale (text-to-speech) avec OpenAI TTS."""

    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def speak(
        self,
        text: str,
        *,
        model: TTSModel = "tts-1",
        voice: Voice = "nova",
        response_format: AudioFormat = "mp3",
        speed: float = 1.0,                  # 0.25 à 4.0
    ) -> TTSResponse:
        """
        Convertit du texte en audio.

        Parameters
        ----------
        model : str
            "tts-1" (rapide, streaming) ou "tts-1-hd" (haute qualité).
        voice : str
            Parmi : "alloy", "echo", "fable", "onyx", "nova", "shimmer".
        response_format : str
            "mp3" (défaut), "opus" (streaming), "flac" (sans perte), "wav".
        speed : float
            Vitesse de parole, de 0.25 (lent) à 4.0 (rapide).
        """
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        resp = requests.post(
            f"{_BASE_URL}/audio/speech",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        return TTSResponse(audio=resp.content, format=response_format, provider="openai-tts")

    def stream(
        self,
        text: str,
        *,
        model: TTSModel = "tts-1",
        voice: Voice = "nova",
        response_format: AudioFormat = "opus",
        speed: float = 1.0,
        chunk_size: int = 4096,
    ):
        """Génère et stream l'audio par chunks — idéal pour lecture en temps réel."""
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        with requests.post(
            f"{_BASE_URL}/audio/speech",
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            self._raise_for_status(resp)
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    yield chunk

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API OpenAI invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit TTS atteinte.")
        if resp.status_code >= 400:
            try:
                detail = resp.json()["error"]["message"]
            except Exception:
                detail = resp.text
            raise APIError(f"Erreur TTS {resp.status_code} : {detail}")