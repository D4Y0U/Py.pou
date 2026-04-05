"""aipou.collect.finance — Données financières et marchés.

Providers :
  - Alpha Vantage : cours actions, forex, crypto, indicateurs techniques (clé gratuite)
  - Polygon.io    : données professionnelles, options, indices (clé gratuite limitée)

Clés API :
  - Alpha Vantage : https://www.alphavantage.co/support/#api-key
  - Polygon.io    : https://polygon.io/dashboard/api-keys
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import requests

from aipou.exceptions import AuthenticationError, APIError, RateLimitError


@dataclass
class Quote:
    """Cours actuel d'un actif financier."""
    symbol: str
    price: float
    change: float             # Variation absolue
    change_percent: float     # Variation en %
    volume: Optional[int] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    previous_close: Optional[float] = None
    market_cap: Optional[float] = None
    timestamp: Optional[str] = None
    provider: str = ""

    def __str__(self) -> str:
        sign = "+" if self.change >= 0 else ""
        return f"{self.symbol}: {self.price:.2f} ({sign}{self.change_percent:.2f}%)"


@dataclass
class OHLCV:
    """Bougie OHLCV (Open/High/Low/Close/Volume) pour une période."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __str__(self) -> str:
        return f"{self.date}: O={self.open} H={self.high} L={self.low} C={self.close} V={self.volume}"


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------

class AlphaVantageProvider:
    """
    Données financières via Alpha Vantage.

    Plan gratuit : 25 req/jour, données end-of-day.
    Clé API : https://www.alphavantage.co/support/#api-key

    Fonctionnalités :
      - Cours actions US et internationaux
      - Cryptomonnaies (BTC, ETH, etc.)
      - Forex (paires de devises)
      - Indicateurs techniques (SMA, EMA, RSI, MACD…)
      - Données économiques (CPI, GDP, chômage)
    """

    _BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout

    def _get(self, **params) -> dict:
        params["apikey"] = self.api_key
        resp = requests.get(self._BASE, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise RateLimitError("Limite Alpha Vantage atteinte (25 req/jour gratuit).")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Alpha Vantage {resp.status_code}: {resp.text}")
        data = resp.json()
        if "Error Message" in data:
            raise APIError(f"Alpha Vantage: {data['Error Message']}")
        if "Information" in data:
            raise RateLimitError(f"Alpha Vantage: {data['Information']}")
        return data

    def quote(self, symbol: str) -> Quote:
        """Cours actuel d'une action (ex. "AAPL", "MSFT", "MC.PAR")."""
        data = self._get(function="GLOBAL_QUOTE", symbol=symbol)
        q = data.get("Global Quote", {})
        return Quote(
            symbol=q.get("01. symbol", symbol),
            price=float(q.get("05. price", 0)),
            change=float(q.get("09. change", 0)),
            change_percent=float(q.get("10. change percent", "0%").replace("%", "")),
            volume=int(q.get("06. volume", 0)),
            open=float(q.get("02. open", 0)),
            high=float(q.get("03. high", 0)),
            low=float(q.get("04. low", 0)),
            previous_close=float(q.get("08. previous close", 0)),
            timestamp=q.get("07. latest trading day"),
            provider="alphavantage",
        )

    def history(
        self,
        symbol: str,
        *,
        interval: Literal["daily", "weekly", "monthly"] = "daily",
        outputsize: Literal["compact", "full"] = "compact",  # compact = 100 jours, full = 20 ans
    ) -> list[OHLCV]:
        """
        Historique OHLCV d'une action.

        Parameters
        ----------
        outputsize : str
            "compact" = 100 dernières données, "full" = toutes les données disponibles.
        """
        func_map = {"daily": "TIME_SERIES_DAILY", "weekly": "TIME_SERIES_WEEKLY",
                    "monthly": "TIME_SERIES_MONTHLY"}
        key_map = {"daily": "Time Series (Daily)", "weekly": "Weekly Time Series",
                   "monthly": "Monthly Time Series"}

        data = self._get(function=func_map[interval], symbol=symbol, outputsize=outputsize)
        series = data.get(key_map[interval], {})

        candles = []
        for date, values in sorted(series.items(), reverse=True):
            candles.append(OHLCV(
                date=date,
                open=float(values["1. open"]),
                high=float(values["2. high"]),
                low=float(values["3. low"]),
                close=float(values["4. close"]),
                volume=int(values["5. volume"]),
            ))
        return candles

    def crypto_quote(self, symbol: str, currency: str = "USD") -> Quote:
        """Cours actuel d'une cryptomonnaie (ex. "BTC", "ETH")."""
        data = self._get(function="CURRENCY_EXCHANGE_RATE",
                         from_currency=symbol, to_currency=currency)
        r = data.get("Realtime Currency Exchange Rate", {})
        price = float(r.get("5. Exchange Rate", 0))
        return Quote(
            symbol=f"{symbol}/{currency}",
            price=price,
            change=0.0,
            change_percent=0.0,
            timestamp=r.get("6. Last Refreshed"),
            provider="alphavantage-crypto",
        )

    def forex(self, from_currency: str, to_currency: str) -> float:
        """Taux de change entre deux devises."""
        data = self._get(function="CURRENCY_EXCHANGE_RATE",
                         from_currency=from_currency, to_currency=to_currency)
        return float(data.get("Realtime Currency Exchange Rate", {}).get("5. Exchange Rate", 0))

    def indicator(
        self,
        symbol: str,
        indicator: Literal["SMA", "EMA", "RSI", "MACD", "BBANDS"],
        *,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ) -> list[dict]:
        """
        Indicateur technique sur une action.

        Parameters
        ----------
        indicator : str
            "SMA" (moyenne mobile), "EMA" (exponentielle), "RSI" (force relative),
            "MACD" (convergence/divergence), "BBANDS" (bandes de Bollinger).
        """
        data = self._get(
            function=indicator, symbol=symbol, interval=interval,
            time_period=time_period, series_type=series_type,
        )
        key = f"Technical Analysis: {indicator}"
        series = data.get(key, {})
        return [{"date": d, **v} for d, v in sorted(series.items(), reverse=True)[:50]]


# ---------------------------------------------------------------------------
# Polygon.io
# ---------------------------------------------------------------------------

class PolygonProvider:
    """
    Données financières professionnelles via Polygon.io.

    Plan gratuit : données end-of-day US, limité à 5 req/min.
    Clé API : https://polygon.io/dashboard/api-keys

    Fonctionnalités :
      - Cours actions US (NYSE, NASDAQ, AMEX)
      - Options (avec plan payant)
      - Crypto et forex
      - Actualités financières
      - Détails des entreprises (secteur, employés, description)
    """

    _BASE = "https://api.polygon.io"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout

    def _get(self, endpoint: str, **params) -> dict:
        params["apiKey"] = self.api_key
        resp = requests.get(f"{self._BASE}{endpoint}", params=params, timeout=self.timeout)
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Polygon invalide.")
        if resp.status_code == 429:
            raise RateLimitError("Limite Polygon atteinte (5 req/min gratuit).")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Polygon {resp.status_code}: {resp.text}")
        return resp.json()

    def quote(self, symbol: str) -> Quote:
        """Cours actuel d'une action US."""
        data = self._get(f"/v2/last/nbbo/{symbol}")
        snapshot = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        day = snapshot.get("ticker", {}).get("day", {})
        prev = snapshot.get("ticker", {}).get("prevDay", {})
        close = day.get("c", 0)
        prev_close = prev.get("c", 1)
        change = close - prev_close
        return Quote(
            symbol=symbol,
            price=close,
            change=change,
            change_percent=(change / prev_close * 100) if prev_close else 0,
            volume=day.get("v"),
            open=day.get("o"),
            high=day.get("h"),
            low=day.get("l"),
            previous_close=prev_close,
            provider="polygon",
        )

    def history(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        *,
        timespan: Literal["minute", "hour", "day", "week", "month"] = "day",
        multiplier: int = 1,
        adjusted: bool = True,
    ) -> list[OHLCV]:
        """
        Historique OHLCV d'une action.

        Parameters
        ----------
        from_date : str
            Date de début au format "YYYY-MM-DD".
        timespan : str
            Granularité : "minute", "hour", "day", "week", "month".
        adjusted : bool
            Ajuste pour les splits et dividendes.
        """
        data = self._get(
            f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            adjusted=str(adjusted).lower(), sort="desc", limit=50000,
        )
        candles = []
        for r in data.get("results", []):
            import datetime
            date_str = datetime.datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d")
            candles.append(OHLCV(
                date=date_str, open=r["o"], high=r["h"], low=r["l"], close=r["c"], volume=int(r["v"])
            ))
        return candles

    def company_details(self, symbol: str) -> dict:
        """Informations sur l'entreprise (secteur, description, site web…)."""
        data = self._get(f"/v3/reference/tickers/{symbol}")
        r = data.get("results", {})
        return {
            "name": r.get("name"),
            "description": r.get("description"),
            "sector": r.get("sic_description"),
            "employees": r.get("total_employees"),
            "website": r.get("homepage_url"),
            "market_cap": r.get("market_cap"),
            "country": r.get("locale"),
            "currency": r.get("currency_name"),
        }

    def news(self, symbol: str, *, limit: int = 10) -> list[dict]:
        """Actualités financières liées à un symbole."""
        data = self._get("/v2/reference/news", ticker=symbol, limit=limit, order="desc")
        return [
            {
                "title": a.get("title"),
                "url": a.get("article_url"),
                "published_at": a.get("published_utc"),
                "source": a.get("publisher", {}).get("name"),
                "summary": a.get("description"),
            }
            for a in data.get("results", [])
        ]