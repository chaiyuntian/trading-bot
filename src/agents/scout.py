"""Scout Agent — sentiment & narrative detection from news feeds.

Inspired by Aura's ScoutAgent but using free RSS feeds instead of LLM APIs.
Scans crypto news for sentiment signals to complement technical analysis.

Source credibility tiers:
  Tier 1 (1.35x): Bloomberg, Reuters, CNBC, WSJ
  Tier 2 (1.15x): CoinDesk, CoinTelegraph, Decrypt, TheBlock
  Tier 3 (0.95x): Reddit, Twitter, social media
"""

import re
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger("agent.scout")

BULLISH_WORDS = {
    "bullish", "surge", "rally", "breakout", "accumulation", "uptick", "soar",
    "gains", "upside", "recovery", "momentum", "buy", "long", "moon", "pump",
    "growth", "adoption", "institutional", "inflow", "etf approval", "upgrade",
    "record high", "all-time high", "ath", "outperform", "beat expectations",
}

BEARISH_WORDS = {
    "bearish", "crash", "plunge", "dump", "sell-off", "selloff", "decline",
    "correction", "downside", "risk", "warning", "collapse", "liquidation",
    "outflow", "ban", "regulation", "hack", "exploit", "rug pull", "scam",
    "downgrade", "miss expectations", "death cross", "capitulation", "fear",
}

TIER1_SOURCES = {"bloomberg", "reuters", "cnbc", "wsj", "financial times", "forbes", "barrons"}
TIER2_SOURCES = {"coindesk", "cointelegraph", "decrypt", "theblock", "benzinga", "seeking alpha"}
TIER3_SOURCES = {"reddit", "twitter", "stocktwits", "discord", "telegram", "youtube"}


def source_credibility(source: str) -> float:
    src = (source or "").lower()
    if any(t in src for t in TIER1_SOURCES):
        return 1.35
    if any(t in src for t in TIER2_SOURCES):
        return 1.15
    if any(t in src for t in TIER3_SOURCES):
        return 0.95
    return 1.0


def text_sentiment(text: str) -> float:
    """Simple keyword-based sentiment: -1 to +1."""
    words = set(re.findall(r'\w+', text.lower()))
    bull = len(words & BULLISH_WORDS)
    bear = len(words & BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


class ScoutAgent:
    """Scans news/social feeds for sentiment signals.

    Returns a sentiment score [-1, +1] and summary.
    Works without API keys (uses RSS feeds).
    Falls back to neutral if feeds unavailable.
    """

    def analyze(self, ticker: str = "BTC") -> dict:
        """Analyze sentiment for a ticker.

        Returns: {score, summary, sources, method}
        """
        symbol = ticker.replace("/USDT", "").replace("-USD", "").replace("USDT", "")

        # Try RSS feeds first
        articles = self._fetch_rss(symbol)
        if articles:
            return self._score_articles(articles, symbol)

        # Fallback: return neutral
        return {
            "score": 0.0,
            "summary": f"No sentiment data available for {symbol}",
            "sources": [],
            "method": "none",
        }

    def _fetch_rss(self, symbol: str) -> list[dict]:
        """Fetch news from free RSS feeds."""
        feeds = [
            f"https://news.google.com/rss/search?q={symbol}+crypto&hl=en",
            f"https://news.google.com/rss/search?q={symbol}+price&hl=en",
        ]
        articles = []
        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            for feed_url in feeds:
                try:
                    req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        xml_data = resp.read()
                    root = ET.fromstring(xml_data)
                    for item in root.findall(".//item")[:15]:
                        title = item.findtext("title", "")
                        source = item.findtext("source", "unknown")
                        articles.append({
                            "title": title,
                            "source": source,
                        })
                except Exception:
                    continue
        except Exception:
            pass
        return articles

    def _score_articles(self, articles: list[dict], symbol: str) -> dict:
        """Score articles by sentiment with source credibility weighting."""
        total_score = 0.0
        total_weight = 0.0

        for article in articles:
            sentiment = text_sentiment(article["title"])
            credibility = source_credibility(article.get("source", ""))
            weight = credibility
            total_score += sentiment * weight
            total_weight += weight

        if total_weight == 0:
            score = 0.0
        else:
            score = total_score / total_weight
            score = max(-1.0, min(1.0, score))

        if score > 0.2:
            summary = f"{symbol}: Positive sentiment ({len(articles)} sources)"
        elif score < -0.2:
            summary = f"{symbol}: Negative sentiment ({len(articles)} sources)"
        else:
            summary = f"{symbol}: Neutral/mixed sentiment ({len(articles)} sources)"

        return {
            "score": round(score, 3),
            "summary": summary,
            "sources": [a.get("source", "unknown") for a in articles[:10]],
            "method": "rss_keyword",
            "article_count": len(articles),
        }
