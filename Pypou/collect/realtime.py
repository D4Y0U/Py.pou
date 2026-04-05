"""aipou.collect.realtime — Données temps réel : météo et réseaux sociaux."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import requests

from aipou.exceptions import AuthenticationError, APIError, RateLimitError


# ===========================================================================
# MÉTÉO — OpenWeatherMap
# ===========================================================================

@dataclass
class WeatherData:
    """Données météorologiques."""
    location: str
    country: str
    temperature: float          # °C
    feels_like: float           # °C
    humidity: int               # %
    description: str
    wind_speed: float           # m/s
    visibility: Optional[int] = None  # mètres
    pressure: Optional[int] = None    # hPa
    clouds: Optional[int] = None      # % couverture nuageuse
    sunrise: Optional[str] = None
    sunset: Optional[str] = None
    icon: Optional[str] = None
    provider: str = "openweathermap"

    def __str__(self) -> str:
        return (f"{self.location} ({self.country}): {self.temperature:.1f}°C, "
                f"{self.description}, Humidité {self.humidity}%")


@dataclass
class ForecastPoint:
    """Un point de prévision météo (toutes les 3h sur 5 jours)."""
    datetime: str
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    rain_probability: float = 0.0


class OpenWeatherProvider:
    """
    Météo actuelle et prévisions via OpenWeatherMap.

    Clé API : https://home.openweathermap.org/api_keys
    Plan gratuit : 1 000 req/jour, prévisions 5 jours.
    """

    _BASE = "https://api.openweathermap.org/data/2.5"
    _GEO_BASE = "https://api.openweathermap.org/geo/1.0"

    def __init__(self, api_key: str, units: Literal["metric", "imperial", "standard"] = "metric",
                 language: str = "fr", timeout: int = 10):
        self.api_key = api_key
        self.units = units          # metric = °C, imperial = °F
        self.language = language
        self.timeout = timeout

    def _params(self, **kwargs) -> dict:
        return {"appid": self.api_key, "units": self.units, "lang": self.language, **kwargs}

    def current(self, location: str) -> WeatherData:
        """
        Météo actuelle pour une ville.

        Parameters
        ----------
        location : str
            Nom de la ville (ex. "Paris", "Paris,FR"), coordonnées "48.85,2.35",
            ou code postal "75001,FR".
        """
        if "," in location and all(c.isdigit() or c in ".,- " for c in location):
            lat, lon = [x.strip() for x in location.split(",")]
            params = self._params(lat=lat, lon=lon)
        else:
            params = self._params(q=location)

        resp = requests.get(f"{self._BASE}/weather", params=params, timeout=self.timeout)
        self._raise(resp)
        d = resp.json()

        import datetime
        def ts(t): return datetime.datetime.fromtimestamp(t).strftime("%H:%M") if t else None

        return WeatherData(
            location=d.get("name", location),
            country=d.get("sys", {}).get("country", ""),
            temperature=d["main"]["temp"],
            feels_like=d["main"]["feels_like"],
            humidity=d["main"]["humidity"],
            description=d["weather"][0]["description"].capitalize(),
            wind_speed=d["wind"]["speed"],
            visibility=d.get("visibility"),
            pressure=d["main"].get("pressure"),
            clouds=d.get("clouds", {}).get("all"),
            sunrise=ts(d.get("sys", {}).get("sunrise")),
            sunset=ts(d.get("sys", {}).get("sunset")),
            icon=d["weather"][0].get("icon"),
        )

    def forecast(self, location: str, *, days: int = 5) -> list[ForecastPoint]:
        """Prévisions sur 5 jours (points toutes les 3h)."""
        resp = requests.get(f"{self._BASE}/forecast",
                            params=self._params(q=location, cnt=days * 8), timeout=self.timeout)
        self._raise(resp)
        points = []
        for item in resp.json().get("list", []):
            points.append(ForecastPoint(
                datetime=item["dt_txt"],
                temperature=item["main"]["temp"],
                feels_like=item["main"]["feels_like"],
                humidity=item["main"]["humidity"],
                description=item["weather"][0]["description"].capitalize(),
                wind_speed=item["wind"]["speed"],
                rain_probability=item.get("pop", 0.0) * 100,
            ))
        return points

    def _raise(self, resp: requests.Response) -> None:
        if resp.status_code == 401:
            raise AuthenticationError("Clé API OpenWeatherMap invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite OpenWeatherMap atteinte.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur OpenWeatherMap {resp.status_code}: {resp.json().get('message', resp.text)}")


# ===========================================================================
# RÉSEAUX SOCIAUX — Reddit
# ===========================================================================

@dataclass
class RedditPost:
    """Un post Reddit."""
    title: str
    url: str
    subreddit: str
    score: int
    num_comments: int
    author: str
    selftext: str = ""
    created_at: Optional[str] = None
    is_self: bool = True         # True = post texte, False = lien externe
    permalink: str = ""
    provider: str = "reddit"

    def __str__(self) -> str:
        return f"r/{self.subreddit} | {self.title} ({self.score}↑, {self.num_comments} comments)"


@dataclass
class RedditComment:
    """Un commentaire Reddit."""
    body: str
    author: str
    score: int
    created_at: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.author} ({self.score}↑): {self.body[:100]}..."


class RedditProvider:
    """
    Données Reddit via l'API publique JSON — sans clé API pour la lecture.

    Idéal pour : analyse de sentiment, tendances, questions/réponses d'une communauté.
    Limitations sans clé : 10 req/min, pas d'accès aux NSFW.
    Clé OAuth optionnelle : https://www.reddit.com/prefs/apps
    """

    _BASE = "https://www.reddit.com"
    _HEADERS = {"User-Agent": "aipou-collector/0.2.0"}

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        subreddit: Optional[str] = None,
        sort: Literal["relevance", "hot", "top", "new", "comments"] = "relevance",
        time_filter: Literal["hour", "day", "week", "month", "year", "all"] = "month",
        limit: int = 25,
    ) -> list[RedditPost]:
        """
        Recherche des posts Reddit.

        Parameters
        ----------
        subreddit : str, optional
            Restreint la recherche à un subreddit (ex. "MachineLearning", "france").
        """
        base = f"{self._BASE}/r/{subreddit}" if subreddit else self._BASE
        resp = requests.get(
            f"{base}/search.json",
            headers=self._HEADERS,
            params={"q": query, "sort": sort, "t": time_filter, "limit": min(limit, 100)},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise APIError(f"Erreur Reddit {resp.status_code}: {resp.text}")
        return [self._parse_post(p["data"]) for p in resp.json()["data"]["children"]]

    def subreddit_posts(
        self,
        subreddit: str,
        *,
        sort: Literal["hot", "new", "top", "rising"] = "hot",
        time_filter: Literal["hour", "day", "week", "month", "year", "all"] = "day",
        limit: int = 25,
    ) -> list[RedditPost]:
        """Posts d'un subreddit par catégorie."""
        resp = requests.get(
            f"{self._BASE}/r/{subreddit}/{sort}.json",
            headers=self._HEADERS,
            params={"t": time_filter, "limit": min(limit, 100)},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise APIError(f"Erreur Reddit: subreddit '{subreddit}' introuvable ou privé.")
        return [self._parse_post(p["data"]) for p in resp.json()["data"]["children"]]

    def comments(self, post_url: str, *, limit: int = 20) -> list[RedditComment]:
        """Récupère les commentaires d'un post."""
        if not post_url.endswith(".json"):
            post_url = post_url.rstrip("/") + ".json"
        resp = requests.get(
            post_url, headers=self._HEADERS, params={"limit": limit}, timeout=self.timeout
        )
        if resp.status_code >= 400:
            raise APIError(f"Erreur Reddit comments {resp.status_code}")
        comments = []
        for item in resp.json()[1]["data"]["children"]:
            d = item["data"]
            if item["kind"] == "t1" and d.get("body", "") not in ("[deleted]", "[removed]"):
                comments.append(RedditComment(
                    body=d["body"],
                    author=d.get("author", "[deleted]"),
                    score=d.get("score", 0),
                ))
        return comments[:limit]

    @staticmethod
    def _parse_post(d: dict) -> RedditPost:
        import datetime
        created = datetime.datetime.fromtimestamp(d.get("created_utc", 0)).isoformat() \
                  if d.get("created_utc") else None
        return RedditPost(
            title=d.get("title", ""),
            url=d.get("url", ""),
            subreddit=d.get("subreddit", ""),
            score=d.get("score", 0),
            num_comments=d.get("num_comments", 0),
            author=d.get("author", "[deleted]"),
            selftext=d.get("selftext", ""),
            created_at=created,
            is_self=d.get("is_self", True),
            permalink=f"https://reddit.com{d.get('permalink', '')}",
        )