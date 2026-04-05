"""aipou.collect.knowledge — Bases de connaissances structurées.

Providers :
  - Wikipedia      : articles encyclopédiques (sans clé API)
  - OpenAlex       : 250M+ papers scientifiques (sans clé pour usage modéré)
  - Semantic Scholar: papers + citations + résumés IA (sans clé / clé optionnelle)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import requests

from aipou.exceptions import APIError, RateLimitError


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class WikiArticle:
    """Article Wikipédia."""
    title: str
    summary: str
    url: str
    full_text: Optional[str] = None
    categories: list[str] = field(default_factory=list)
    language: str = "fr"
    page_id: Optional[int] = None

    def __str__(self) -> str:
        return self.summary


@dataclass
class AcademicPaper:
    """Article scientifique."""
    title: str
    abstract: Optional[str]
    authors: list[str]
    year: Optional[int]
    doi: Optional[str] = None
    url: Optional[str] = None
    venue: Optional[str] = None        # Journal ou conférence
    citation_count: int = 0
    open_access: bool = False
    provider: str = ""

    def __str__(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{self.title} — {authors_str} ({self.year})"


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

class WikipediaProvider:
    """
    Extraction d'articles Wikipédia — sans clé API, multilingue.

    Utilise l'API REST Wikipedia officielle.
    """

    def __init__(self, language: str = "fr", timeout: int = 10):
        self.language = language
        self.timeout = timeout

    def _base(self) -> str:
        return f"https://{self.language}.wikipedia.org/api/rest_v1"

    def _wiki_base(self) -> str:
        return f"https://{self.language}.wikipedia.org/w/api.php"

    def search(self, query: str, *, limit: int = 5) -> list[str]:
        """Retourne les titres d'articles correspondant à la recherche."""
        resp = requests.get(
            self._wiki_base(),
            params={"action": "opensearch", "search": query, "limit": limit, "format": "json"},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise APIError(f"Erreur Wikipedia: {resp.text}")
        data = resp.json()
        return data[1] if len(data) > 1 else []

    def get(self, title: str, *, full_text: bool = False) -> WikiArticle:
        """
        Récupère un article Wikipédia par son titre.

        Parameters
        ----------
        title : str
            Titre exact ou approximatif de l'article.
        full_text : bool
            Si True, récupère le texte complet en plus du résumé.
        """
        # Résumé via REST API
        resp = requests.get(
            f"{self._base()}/page/summary/{requests.utils.quote(title)}",
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            raise APIError(f"Article Wikipedia '{title}' introuvable.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Wikipedia {resp.status_code}: {resp.text}")

        data = resp.json()
        article = WikiArticle(
            title=data.get("title", title),
            summary=data.get("extract", ""),
            url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            language=self.language,
            page_id=data.get("pageid"),
        )

        if full_text:
            article.full_text = self._get_full_text(data.get("pageid", 0))

        return article

    def search_and_get(self, query: str) -> Optional[WikiArticle]:
        """Cherche et retourne directement le premier résultat."""
        titles = self.search(query, limit=1)
        if not titles:
            return None
        return self.get(titles[0])

    def _get_full_text(self, page_id: int) -> str:
        resp = requests.get(
            self._wiki_base(),
            params={
                "action": "query", "pageids": page_id, "prop": "extracts",
                "explaintext": True, "format": "json",
            },
            timeout=self.timeout,
        )
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        return next(iter(pages.values()), {}).get("extract", "")


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

class OpenAlexProvider:
    """
    250M+ articles scientifiques via OpenAlex — gratuit, sans clé.

    Ajouter un email améliore les limites de débit (polite pool).
    Clé API optionnelle : https://openalex.org/
    """

    _BASE = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 15):
        self.email = email
        self.api_key = api_key
        self.timeout = timeout

    def _params(self, **kwargs) -> dict:
        p = dict(kwargs)
        if self.email:
            p["mailto"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        open_access_only: bool = False,
        sort: str = "cited_by_count:desc",
    ) -> list[AcademicPaper]:
        """
        Recherche des articles scientifiques.

        Parameters
        ----------
        sort : str
            "cited_by_count:desc" (plus cités), "publication_date:desc" (récents).
        open_access_only : bool
            Filtre les articles en accès libre uniquement.
        """
        filters = [f"title_and_abstract.search:{query}"]
        if year_from:
            filters.append(f"publication_year:>{year_from - 1}")
        if year_to:
            filters.append(f"publication_year:<{year_to + 1}")
        if open_access_only:
            filters.append("is_oa:true")

        params = self._params(
            filter=",".join(filters),
            per_page=min(limit, 200),
            sort=sort,
        )
        resp = requests.get(f"{self._BASE}/works", params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RateLimitError("Limite OpenAlex atteinte. Ajoutez votre email dans le constructeur.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur OpenAlex {resp.status_code}: {resp.text}")

        return [self._parse(w) for w in resp.json().get("results", [])]

    def get_by_doi(self, doi: str) -> AcademicPaper:
        """Récupère un article par son DOI."""
        resp = requests.get(
            f"{self._BASE}/works/doi:{doi}", params=self._params(), timeout=self.timeout
        )
        if resp.status_code == 404:
            raise APIError(f"DOI '{doi}' introuvable sur OpenAlex.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur OpenAlex {resp.status_code}: {resp.text}")
        return self._parse(resp.json())

    def citations(self, openalex_id: str, *, limit: int = 20) -> list[AcademicPaper]:
        """Retourne les articles qui citent un travail donné."""
        params = self._params(filter=f"cites:{openalex_id}", per_page=min(limit, 200))
        resp = requests.get(f"{self._BASE}/works", params=params, timeout=self.timeout)
        if resp.status_code >= 400:
            raise APIError(f"Erreur OpenAlex: {resp.text}")
        return [self._parse(w) for w in resp.json().get("results", [])]

    @staticmethod
    def _parse(w: dict) -> AcademicPaper:
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in w.get("authorships", [])
        ]
        best_loc = w.get("best_oa_location") or w.get("primary_location") or {}
        return AcademicPaper(
            title=w.get("title", ""),
            abstract=w.get("abstract"),
            authors=authors,
            year=w.get("publication_year"),
            doi=w.get("doi", "").replace("https://doi.org/", "") if w.get("doi") else None,
            url=best_loc.get("landing_page_url") or w.get("doi"),
            venue=w.get("primary_location", {}).get("source", {}).get("display_name"),
            citation_count=w.get("cited_by_count", 0),
            open_access=w.get("open_access", {}).get("is_oa", False),
            provider="openalex",
        )


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

class SemanticScholarProvider:
    """
    Articles scientifiques + résumés IA via Semantic Scholar.

    Clé API optionnelle (augmente les limites) : https://www.semanticscholar.org/product/api
    Sans clé : 100 req/5min. Avec clé : 1 req/s.
    """

    _BASE = "https://api.semanticscholar.org/graph/v1"

    _DEFAULT_FIELDS = "title,abstract,authors,year,citationCount,openAccessPdf,venue,externalIds"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        fields: Optional[str] = None,
        year_range: Optional[str] = None,    # Ex. "2020-2024" ou "2022-"
        open_access_only: bool = False,
    ) -> list[AcademicPaper]:
        """
        Recherche des articles avec Semantic Scholar.

        Parameters
        ----------
        year_range : str, optional
            Plage d'années : "2020-2024", "2022-", "-2021".
        """
        params: dict = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields or self._DEFAULT_FIELDS,
        }
        if year_range:
            params["year"] = year_range
        if open_access_only:
            params["openAccessPdf"] = ""

        resp = requests.get(f"{self._BASE}/paper/search",
                            headers=self._headers(), params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RateLimitError("Limite Semantic Scholar atteinte.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Semantic Scholar {resp.status_code}: {resp.text}")

        return [self._parse(p) for p in resp.json().get("data", [])]

    def get_paper(self, paper_id: str) -> AcademicPaper:
        """Récupère un paper par son ID (S2, DOI, ArXiv, etc.)."""
        resp = requests.get(
            f"{self._BASE}/paper/{paper_id}",
            headers=self._headers(),
            params={"fields": self._DEFAULT_FIELDS},
            timeout=self.timeout,
        )
        if resp.status_code == 404:
            raise APIError(f"Paper '{paper_id}' introuvable.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Semantic Scholar: {resp.text}")
        return self._parse(resp.json())

    def recommendations(self, paper_id: str, *, limit: int = 10) -> list[AcademicPaper]:
        """Retourne des articles similaires recommandés par l'IA de S2."""
        resp = requests.get(
            f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}",
            headers=self._headers(),
            params={"limit": limit, "fields": self._DEFAULT_FIELDS},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise APIError(f"Erreur S2 recommendations: {resp.text}")
        return [self._parse(p) for p in resp.json().get("recommendedPapers", [])]

    @staticmethod
    def _parse(p: dict) -> AcademicPaper:
        authors = [a.get("name", "") for a in p.get("authors", [])]
        pdf = p.get("openAccessPdf") or {}
        ext = p.get("externalIds") or {}
        return AcademicPaper(
            title=p.get("title", ""),
            abstract=p.get("abstract"),
            authors=authors,
            year=p.get("year"),
            doi=ext.get("DOI"),
            url=pdf.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
            venue=p.get("venue"),
            citation_count=p.get("citationCount", 0),
            open_access=bool(pdf.get("url")),
            provider="semantic-scholar",
        )