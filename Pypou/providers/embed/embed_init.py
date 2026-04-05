"""aipou.embed — embeddings vectoriels unifiés.

Trois providers : OpenAI, Voyage AI, Jina AI.
Interface identique pour les trois via EmbedClient.
"""

from typing import Literal, Optional
import requests

from aipou.models import EmbeddingResponse, TokenUsage
from aipou.exceptions import AuthenticationError, APIError, RateLimitError


# ---------------------------------------------------------------------------
# OpenAI Embeddings
# ---------------------------------------------------------------------------

class OpenAIEmbedProvider:
    """Embeddings via OpenAI (text-embedding-3-large, text-embedding-3-small)."""

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def embed(
        self,
        texts: list[str],
        *,
        model: str = "text-embedding-3-large",
        dimensions: Optional[int] = None,   # Réduction de dimension (ex. 256, 512, 1536)
    ) -> EmbeddingResponse:
        """
        Génère des embeddings pour une liste de textes.

        Parameters
        ----------
        model : str
            "text-embedding-3-large" (3072 dims, meilleure qualité),
            "text-embedding-3-small" (1536 dims, moins cher),
            "text-embedding-ada-002" (legacy, 1536 dims).
        dimensions : int, optional
            Réduit la dimension de sortie (uniquement pour les modèles v3).
        """
        payload: dict = {"model": model, "input": texts}
        if dimensions:
            payload["dimensions"] = dimensions

        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        _raise(resp, "OpenAI Embeddings")
        data = resp.json()

        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        usage = data.get("usage", {})
        return EmbeddingResponse(
            vectors=vectors,
            model=data.get("model", model),
            provider="openai",
            usage=TokenUsage(prompt_tokens=usage.get("prompt_tokens", 0)),
            raw=data,
        )


# ---------------------------------------------------------------------------
# Voyage AI Embeddings
# ---------------------------------------------------------------------------

class VoyageEmbedProvider:
    """Embeddings spécialisés via Voyage AI.

    Clé API : https://dash.voyageai.com/api-keys

    Modèles :
      - voyage-3-large     : meilleure qualité générale (1024 dims)
      - voyage-3           : équilibré (1024 dims)
      - voyage-3-lite      : économique (512 dims)
      - voyage-code-3      : optimisé pour le code
      - voyage-finance-2   : optimisé pour la finance
      - voyage-law-2       : optimisé pour le juridique
      - voyage-multilingual-2 : multilingue
    """

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def embed(
        self,
        texts: list[str],
        *,
        model: str = "voyage-3-large",
        input_type: Optional[Literal["query", "document"]] = None,
        truncation: bool = True,
    ) -> EmbeddingResponse:
        """
        Génère des embeddings via Voyage AI.

        Parameters
        ----------
        input_type : str, optional
            "query" pour les requêtes de recherche, "document" pour les textes à indexer.
            Améliore la pertinence en mode RAG.
        truncation : bool
            Tronque automatiquement les textes dépassant la limite du modèle.
        """
        payload: dict = {"model": model, "input": texts, "truncation": truncation}
        if input_type:
            payload["input_type"] = input_type

        resp = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        _raise(resp, "Voyage AI")
        data = resp.json()

        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        usage = data.get("usage", {})
        return EmbeddingResponse(
            vectors=vectors,
            model=model,
            provider="voyage",
            usage=TokenUsage(prompt_tokens=usage.get("total_tokens", 0)),
            raw=data,
        )


# ---------------------------------------------------------------------------
# Jina AI Embeddings
# ---------------------------------------------------------------------------

class JinaEmbedProvider:
    """Embeddings open-source via Jina AI — longue fenêtre (8192 tokens).

    Clé API : https://jina.ai/?sui=apikey

    Modèles :
      - jina-embeddings-v3          : multilingue, 1024 dims (recommandé)
      - jina-clip-v2                : texte + images (multimodal)
      - jina-colbert-v2             : re-ranking
    """

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def embed(
        self,
        texts: list[str],
        *,
        model: str = "jina-embeddings-v3",
        task: Literal[
            "retrieval.query", "retrieval.passage",
            "separation", "classification", "text-matching"
        ] = "retrieval.passage",
        dimensions: int = 1024,
        late_chunking: bool = False,
    ) -> EmbeddingResponse:
        """
        Génère des embeddings via Jina AI.

        Parameters
        ----------
        task : str
            Type de tâche. "retrieval.query" pour les requêtes,
            "retrieval.passage" pour les documents à indexer.
        dimensions : int
            Dimension des vecteurs (jusqu'à 1024 pour jina-embeddings-v3).
        late_chunking : bool
            Active le découpage tardif — améliore la qualité sur les longs documents.
        """
        payload: dict = {
            "model": model,
            "input": [{"text": t} for t in texts],
            "task": task,
            "dimensions": dimensions,
            "late_chunking": late_chunking,
        }
        resp = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        _raise(resp, "Jina AI")
        data = resp.json()

        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        usage = data.get("usage", {})
        return EmbeddingResponse(
            vectors=vectors,
            model=model,
            provider="jina",
            usage=TokenUsage(prompt_tokens=usage.get("prompt_tokens", 0)),
            raw=data,
        )


# ---------------------------------------------------------------------------
# Client unifié
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai": OpenAIEmbedProvider,
    "voyage": VoyageEmbedProvider,
    "jina": JinaEmbedProvider,
}


class EmbedClient:
    """
    Client d'embeddings unifié — même interface pour OpenAI, Voyage et Jina.

    Exemple
    -------
        from aipou.embed import EmbedClient

        client = EmbedClient("openai", api_key="sk-...")
        response = client.embed(["Bonjour monde", "Hello world"])
        print(response.dimension)   # 3072
        print(response.vectors[0]) # [0.023, -0.104, ...]
    """

    def __init__(self, provider: str, api_key: str, **kwargs):
        key = provider.lower()
        if key not in _PROVIDERS:
            raise ValueError(f"Provider embed inconnu '{provider}'. Disponibles : {list(_PROVIDERS)}")
        self._provider = _PROVIDERS[key](api_key=api_key, **kwargs)
        self.provider_name = key

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResponse:
        return self._provider.embed(texts, **kwargs)

    def embed_one(self, text: str, **kwargs) -> list[float]:
        """Raccourci pour un texte unique — retourne directement le vecteur."""
        return self._provider.embed([text], **kwargs).vectors[0]


# ---------------------------------------------------------------------------
# Helper d'erreurs partagé
# ---------------------------------------------------------------------------

def _raise(resp: requests.Response, provider_name: str) -> None:
    if resp.status_code == 401:
        raise AuthenticationError(f"Clé API {provider_name} invalide.")
    if resp.status_code == 429:
        raise RateLimitError(f"Limite de débit {provider_name} atteinte.")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            detail = resp.text
        raise APIError(f"Erreur {provider_name} {resp.status_code} : {detail}")