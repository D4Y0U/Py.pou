"""aipou.collect.web — Scraping web propre pour LLM.

Deux providers :
  - Firecrawl  : scraping avancé, crawling de sites entiers, sitemap
  - Jina Reader: simple, gratuit, retourne du Markdown clean depuis n'importe quelle URL

Clés API :
  - Firecrawl : https://firecrawl.dev/app/api-keys
  - Jina      : https://jina.ai/?sui=apikey  (ou sans clé, rate-limitée)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import requests

from aipou.exceptions import AuthenticationError, APIError, RateLimitError


@dataclass
class ScrapedPage:
    """Résultat du scraping d'une page web."""
    url: str
    markdown: str                        # Contenu principal en Markdown
    title: Optional[str] = None
    description: Optional[str] = None
    links: list[str] = field(default_factory=list)
    provider: str = ""
    raw: Optional[dict] = field(default=None, repr=False)

    def __str__(self) -> str:
        return self.markdown

    @property
    def word_count(self) -> int:
        return len(self.markdown.split())


# ---------------------------------------------------------------------------
# Firecrawl
# ---------------------------------------------------------------------------

class FirecrawlProvider:
    """
    Scraping avancé via Firecrawl.

    Capacités :
      - Scraping d'une URL unique (retourne Markdown propre)
      - Crawling d'un site entier (suit les liens)
      - Extraction de sitemap
      - Mode LLM-ready : supprime nav, footer, publicités
    """

    _BASE = "https://api.firecrawl.dev/v1"

    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def scrape(
        self,
        url: str,
        *,
        formats: list[str] = ["markdown"],
        only_main_content: bool = True,
        exclude_tags: list[str] = ["nav", "footer", "header", "aside"],
        wait_for: int = 0,                  # ms à attendre (pour les SPAs)
    ) -> ScrapedPage:
        """
        Scrape une URL et retourne le contenu en Markdown.

        Parameters
        ----------
        only_main_content : bool
            Supprime la navigation, les pubs, les footers.
        wait_for : int
            Millisecondes d'attente avant scraping (utile pour les pages JS).
        """
        payload: dict = {
            "url": url,
            "formats": formats,
            "onlyMainContent": only_main_content,
            "excludeTags": exclude_tags,
        }
        if wait_for:
            payload["waitFor"] = wait_for

        resp = requests.post(f"{self._BASE}/scrape", headers=self._headers(),
                             json=payload, timeout=self.timeout)
        _raise(resp, "Firecrawl")
        data = resp.json().get("data", {})

        return ScrapedPage(
            url=url,
            markdown=data.get("markdown", ""),
            title=data.get("metadata", {}).get("title"),
            description=data.get("metadata", {}).get("description"),
            links=data.get("links", []),
            provider="firecrawl",
            raw=data,
        )

    def crawl(
        self,
        url: str,
        *,
        max_depth: int = 2,
        limit: int = 10,
        exclude_patterns: list[str] = [],
        include_patterns: list[str] = [],
        only_main_content: bool = True,
    ) -> list[ScrapedPage]:
        """
        Crawle un site entier depuis une URL de départ.

        Parameters
        ----------
        max_depth : int
            Profondeur maximale de navigation (1 = page + liens directs).
        limit : int
            Nombre maximum de pages à scraper.
        """
        payload: dict = {
            "url": url,
            "maxDepth": max_depth,
            "limit": limit,
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": only_main_content},
        }
        if exclude_patterns:
            payload["excludePaths"] = exclude_patterns
        if include_patterns:
            payload["includePaths"] = include_patterns

        # Lance le crawl (asynchrone)
        resp = requests.post(f"{self._BASE}/crawl", headers=self._headers(),
                             json=payload, timeout=self.timeout)
        _raise(resp, "Firecrawl")
        crawl_id = resp.json().get("id")

        # Poll jusqu'à complétion
        import time
        for _ in range(60):
            time.sleep(2)
            status_resp = requests.get(f"{self._BASE}/crawl/{crawl_id}",
                                       headers=self._headers(), timeout=self.timeout)
            _raise(status_resp, "Firecrawl")
            status_data = status_resp.json()
            if status_data.get("status") == "completed":
                pages = []
                for item in status_data.get("data", []):
                    pages.append(ScrapedPage(
                        url=item.get("metadata", {}).get("sourceURL", ""),
                        markdown=item.get("markdown", ""),
                        title=item.get("metadata", {}).get("title"),
                        provider="firecrawl",
                        raw=item,
                    ))
                return pages
        raise APIError("Firecrawl crawl timeout — le site prend trop de temps.")

    def sitemap(self, url: str) -> list[str]:
        """Retourne toutes les URLs d'un sitemap."""
        resp = requests.get(f"{self._BASE}/map",
                            headers=self._headers(),
                            params={"url": url},
                            timeout=self.timeout)
        _raise(resp, "Firecrawl")
        return resp.json().get("links", [])


# ---------------------------------------------------------------------------
# Jina Reader
# ---------------------------------------------------------------------------

class JinaReaderProvider:
    """
    Scraping simple et gratuit via Jina Reader.

    Aucune configuration requise — préfixe simplement r.jina.ai/ à n'importe quelle URL.
    Une clé API est optionnelle mais lève les limites de débit.
    """

    _BASE = "https://r.jina.ai"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        h = {"Accept": "application/json", "X-Return-Format": "markdown"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def scrape(self, url: str, *, images: bool = False) -> ScrapedPage:
        """
        Scrape une URL via Jina Reader.

        Parameters
        ----------
        images : bool
            Inclut les URLs des images dans le Markdown retourné.
        """
        headers = self._headers()
        if not images:
            headers["X-Remove-Selector"] = "img"

        resp = requests.get(f"{self._BASE}/{url}", headers=headers, timeout=self.timeout)
        if resp.status_code == 429:
            raise RateLimitError("Limite Jina Reader atteinte. Ajoutez une clé API.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Jina Reader {resp.status_code}: {resp.text}")

        data = resp.json()
        content = data.get("data", {})
        return ScrapedPage(
            url=url,
            markdown=content.get("content", ""),
            title=content.get("title"),
            description=content.get("description"),
            links=content.get("links", {}).get("hrefs", []),
            provider="jina-reader",
            raw=data,
        )

    def scrape_many(self, urls: list[str], **kwargs) -> list[ScrapedPage]:
        """Scrape plusieurs URLs séquentiellement."""
        return [self.scrape(url, **kwargs) for url in urls]


def _raise(resp: requests.Response, name: str) -> None:
    if resp.status_code == 401:
        raise AuthenticationError(f"Clé API {name} invalide.")
    if resp.status_code == 429:
        raise RateLimitError(f"Limite de débit {name} atteinte.")
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise APIError(f"Erreur {name} {resp.status_code}: {detail}")