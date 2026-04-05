"""aipou.collect.search — Résultats de recherche web structurés.

Providers :
  - SerpAPI     : résultats Google, Bing, YouTube, Scholar… (clé : https://serpapi.com)
  - Brave Search: résultats indépendants, privacy-first (clé : https://api.search.brave.com)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import requests

from aipou.exceptions import AuthenticationError, APIError, RateLimitError


@dataclass
class SearchResult:
    """Un résultat de recherche."""
    title: str
    url: str
    snippet: str
    position: int = 0
    source: Optional[str] = None
    date: Optional[str] = None
    provider: str = ""

    def __str__(self) -> str:
        return f"{self.position}. {self.title}\n   {self.url}\n   {self.snippet}"


@dataclass
class SearchResponse:
    """Réponse complète d'une recherche."""
    query: str
    results: list[SearchResult]
    total_results: Optional[int] = None
    provider: str = ""

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def to_context(self) -> str:
        """Formate les résultats en contexte texte pour un LLM."""
        lines = [f"Résultats de recherche pour : {self.query}\n"]
        for r in self.results:
            lines.append(f"[{r.position}] {r.title}")
            lines.append(f"    URL : {r.url}")
            lines.append(f"    {r.snippet}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SerpAPI
# ---------------------------------------------------------------------------

class SerpAPIProvider:
    """
    Résultats de recherche via SerpAPI.

    Supporte : Google, Bing, DuckDuckGo, Google Scholar, Google Images,
               Google News, YouTube Search, Google Shopping, Google Maps.
    Clé API : https://serpapi.com/manage-api-key
    """

    _BASE = "https://serpapi.com/search"

    def __init__(self, api_key: str, timeout: int = 20):
        self.api_key = api_key
        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        engine: Literal["google", "bing", "duckduckgo", "google_news", "google_scholar"] = "google",
        num: int = 10,
        language: str = "fr",
        country: str = "fr",
        safe: Literal["active", "off"] = "off",
        time_range: Optional[Literal["d", "w", "m", "y"]] = None,
    ) -> SearchResponse:
        """
        Effectue une recherche web.

        Parameters
        ----------
        engine : str
            Moteur de recherche. "google_scholar" pour la recherche académique.
        num : int
            Nombre de résultats (max 100 pour Google).
        time_range : str
            Filtre temporel : "d" (jour), "w" (semaine), "m" (mois), "y" (an).
        """
        params: dict = {
            "api_key": self.api_key,
            "engine": engine,
            "q": query,
            "num": min(num, 100),
            "hl": language,
            "gl": country,
            "safe": safe,
        }
        if time_range:
            params["tbs"] = f"qdr:{time_range}"

        resp = requests.get(self._BASE, params=params, timeout=self.timeout)
        self._raise(resp)
        data = resp.json()

        results = []
        for i, r in enumerate(data.get("organic_results", []), 1):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                position=i,
                date=r.get("date"),
                provider="serpapi",
            ))

        return SearchResponse(
            query=query,
            results=results,
            total_results=data.get("search_information", {}).get("total_results"),
            provider="serpapi",
        )

    def scholar(self, query: str, *, num: int = 10, year_from: Optional[int] = None) -> SearchResponse:
        """Recherche dans Google Scholar (articles académiques)."""
        params: dict = {"api_key": self.api_key, "engine": "google_scholar", "q": query, "num": num}
        if year_from:
            params["as_ylo"] = year_from
        resp = requests.get(self._BASE, params=params, timeout=self.timeout)
        self._raise(resp)
        data = resp.json()
        results = []
        for i, r in enumerate(data.get("organic_results", []), 1):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                position=i,
                date=str(r.get("publication_info", {}).get("summary", "")),
                provider="serpapi-scholar",
            ))
        return SearchResponse(query=query, results=results, provider="serpapi-scholar")

    def _raise(self, resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API SerpAPI invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit SerpAPI atteinte.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur SerpAPI {resp.status_code}: {resp.text}")


# ---------------------------------------------------------------------------
# Brave Search
# ---------------------------------------------------------------------------

class BraveSearchProvider:
    """
    Résultats de recherche indépendants via Brave Search API.

    Privacy-first, pas de tracking Google.
    Clé API : https://api.search.brave.com/app/keys
    Plan gratuit : 2 000 req/mois.
    """

    _BASE = "https://api.search.brave.com/res/v1"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

    def search(
        self,
        query: str,
        *,
        count: int = 10,
        country: str = "FR",
        language: str = "fr",
        safe_search: Literal["off", "moderate", "strict"] = "moderate",
        freshness: Optional[Literal["pd", "pw", "pm", "py"]] = None,
    ) -> SearchResponse:
        """
        Recherche web via Brave.

        Parameters
        ----------
        freshness : str
            Filtre temporel : "pd" (hier), "pw" (semaine), "pm" (mois), "py" (an).
        """
        params: dict = {
            "q": query,
            "count": min(count, 20),
            "country": country,
            "search_lang": language,
            "safesearch": safe_search,
        }
        if freshness:
            params["freshness"] = freshness

        resp = requests.get(f"{self._BASE}/web/search",
                            headers=self._headers(), params=params, timeout=self.timeout)
        self._raise(resp)
        data = resp.json()

        results = []
        for i, r in enumerate(data.get("web", {}).get("results", []), 1):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                position=i,
                date=r.get("age"),
                provider="brave",
            ))
        return SearchResponse(
            query=query,
            results=results,
            total_results=data.get("web", {}).get("totalCount"),
            provider="brave",
        )

    def news(self, query: str, *, count: int = 10, country: str = "FR") -> SearchResponse:
        """Recherche dans les actualités récentes."""
        params = {"q": query, "count": min(count, 20), "country": country}
        resp = requests.get(f"{self._BASE}/news/search",
                            headers=self._headers(), params=params, timeout=self.timeout)
        self._raise(resp)
        results = []
        for i, r in enumerate(resp.json().get("results", []), 1):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                position=i,
                date=r.get("age"),
                source=r.get("meta_url", {}).get("hostname"),
                provider="brave-news",
            ))
        return SearchResponse(query=query, results=results, provider="brave-news")

    def _raise(self, resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Brave Search invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite de débit Brave Search atteinte.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Brave Search {resp.status_code}: {resp.text}")