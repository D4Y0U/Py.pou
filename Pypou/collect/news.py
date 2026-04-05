"""aipou.collect.news — Flux d'actualités en temps réel.

Providers :
  - NewsAPI : articles de presse de +80 000 sources (clé : https://newsapi.org/account)
  - GDELT   : base mondiale d'événements géopolitiques (gratuit, sans clé)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Literal, Optional
import requests

from aipou.exceptions import AuthenticationError, APIError, RateLimitError


@dataclass
class NewsArticle:
    """Un article de presse."""
    title: str
    url: str
    source: str
    published_at: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[str] = None
    provider: str = ""

    def __str__(self) -> str:
        return f"[{self.source}] {self.title}"


@dataclass
class GDELTEvent:
    """Un événement GDELT."""
    date: str
    actor1: str
    actor2: str
    event_code: str
    event_description: str
    country: Optional[str] = None
    tone: Optional[float] = None          # Négatif = hostile, positif = coopératif
    source_url: Optional[str] = None
    num_mentions: int = 0

    def __str__(self) -> str:
        return f"{self.date} | {self.actor1} → {self.actor2} : {self.event_description}"


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

class NewsAPIProvider:
    """
    Articles de presse via NewsAPI.

    Clé API : https://newsapi.org/account
    Plan gratuit : 100 req/jour, articles des 30 derniers jours, usage dev uniquement.
    """

    _BASE = "https://newsapi.org/v2"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout

    def _params(self, **kwargs) -> dict:
        return {"apiKey": self.api_key, **kwargs}

    def search(
        self,
        query: str,
        *,
        language: str = "fr",
        from_date: Optional[str] = None,      # Format ISO : "2024-01-01"
        to_date: Optional[str] = None,
        sort_by: Literal["relevancy", "popularity", "publishedAt"] = "publishedAt",
        page_size: int = 20,
        page: int = 1,
        sources: Optional[str] = None,        # Ex. "bbc-news,le-monde"
        domains: Optional[str] = None,        # Ex. "lemonde.fr,lefigaro.fr"
    ) -> list[NewsArticle]:
        """
        Recherche des articles par mots-clés.

        Parameters
        ----------
        query : str
            Mots-clés, booléens supportés : "intelligence artificielle AND France"
        language : str
            Code ISO-639-1 : "fr", "en", "de", "es"…
        sort_by : str
            "publishedAt" (récents), "relevancy" (pertinents), "popularity" (populaires).
        """
        params = self._params(
            q=query, language=language, sortBy=sort_by,
            pageSize=min(page_size, 100), page=page,
        )
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if sources:
            params["sources"] = sources
        if domains:
            params["domains"] = domains

        resp = requests.get(f"{self._BASE}/everything", params=params, timeout=self.timeout)
        self._raise(resp)
        return [self._parse(a) for a in resp.json().get("articles", [])]

    def top_headlines(
        self,
        *,
        query: Optional[str] = None,
        country: str = "fr",
        category: Optional[Literal[
            "business", "entertainment", "general",
            "health", "science", "sports", "technology"
        ]] = None,
        page_size: int = 20,
    ) -> list[NewsArticle]:
        """Retourne les titres de presse du moment."""
        params = self._params(country=country, pageSize=min(page_size, 100))
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        resp = requests.get(f"{self._BASE}/top-headlines", params=params, timeout=self.timeout)
        self._raise(resp)
        return [self._parse(a) for a in resp.json().get("articles", [])]

    def sources(self, language: str = "fr", country: str = "fr") -> list[dict]:
        """Liste les sources de presse disponibles."""
        resp = requests.get(
            f"{self._BASE}/top-headlines/sources",
            params=self._params(language=language, country=country),
            timeout=self.timeout,
        )
        self._raise(resp)
        return resp.json().get("sources", [])

    @staticmethod
    def _parse(a: dict) -> NewsArticle:
        return NewsArticle(
            title=a.get("title", ""),
            url=a.get("url", ""),
            source=a.get("source", {}).get("name", ""),
            published_at=a.get("publishedAt"),
            description=a.get("description"),
            content=a.get("content"),
            author=a.get("author"),
            image_url=a.get("urlToImage"),
            provider="newsapi",
        )

    def _raise(self, resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API NewsAPI invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit NewsAPI atteinte.")
        if resp.status_code >= 400:
            msg = resp.json().get("message", resp.text)
            raise APIError(f"Erreur NewsAPI {resp.status_code}: {msg}")


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------

class GDELTProvider:
    """
    Base mondiale d'événements géopolitiques GDELT — gratuit, sans clé API.

    GDELT surveille les médias du monde entier et encode les événements
    selon la taxonomie CAMEO (Conflict and Mediation Event Observations).

    Deux modes :
      - GKG (Global Knowledge Graph) : articles récents + ton émotionnel
      - DOC : recherche plein-texte dans les médias indexés
    """

    _DOC_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
    _GKG_BASE = "https://api.gdeltproject.org/api/v2/gkg/gkg"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def search_articles(
        self,
        query: str,
        *,
        mode: Literal["artlist", "timelinevol", "tonechart"] = "artlist",
        max_records: int = 25,
        start_date: Optional[str] = None,    # Format : "20240101000000"
        end_date: Optional[str] = None,
        source_country: Optional[str] = None,  # Code ISO-2 : "FR", "US"
        source_lang: Optional[str] = None,     # Ex. "french", "english"
        sort: Literal["DateDesc", "DateAsc", "ToneDesc", "ToneAsc"] = "DateDesc",
    ) -> list[NewsArticle]:
        """
        Recherche des articles dans les médias mondiaux indexés par GDELT.

        Parameters
        ----------
        query : str
            Mots-clés en anglais (GDELT indexe surtout la presse anglophone).
        source_country : str
            Filtre par pays source. Ex. "FR" pour la France.
        """
        params: dict = {
            "query": query,
            "mode": mode,
            "maxrecords": max_records,
            "sort": sort,
            "format": "json",
        }
        if start_date:
            params["startdatetime"] = start_date
        if end_date:
            params["enddatetime"] = end_date
        if source_country:
            params["query"] += f" sourcecountry:{source_country}"
        if source_lang:
            params["query"] += f" sourcelang:{source_lang}"

        resp = requests.get(self._DOC_BASE, params=params, timeout=self.timeout)
        if resp.status_code >= 400:
            raise APIError(f"Erreur GDELT {resp.status_code}: {resp.text}")

        articles = []
        for a in resp.json().get("articles", []):
            articles.append(NewsArticle(
                title=a.get("title", ""),
                url=a.get("url", ""),
                source=a.get("domain", ""),
                published_at=a.get("seendate"),
                description=None,
                provider="gdelt",
            ))
        return articles

    def tone_timeline(self, query: str, *, timespan: str = "1month") -> list[dict]:
        """
        Retourne l'évolution du ton émotionnel d'un sujet dans le temps.

        Returns
        -------
        list[dict]
            Chaque entrée : {"date": "...", "tone": float, "volume": int}
            Tone négatif = couverture hostile, positif = coopérative.
        """
        params = {
            "query": query,
            "mode": "timelinetone",
            "timespan": timespan,
            "format": "json",
        }
        resp = requests.get(self._DOC_BASE, params=params, timeout=self.timeout)
        if resp.status_code >= 400:
            raise APIError(f"Erreur GDELT {resp.status_code}: {resp.text}")
        return resp.json().get("timeline", [])