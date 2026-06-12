"""Crypto-specific sentiment data sources: Reddit and Google News.

Reddit scraper ported from Krypto-v0/src/scraping/reddit/scraper.py.
Google News integration uses the GoogleNews library.

These functions return raw text data (posts, article titles/summaries)
for the LLM analyst to interpret sentiment — no dedicated NLP model needed.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Annotated

import requests

logger = logging.getLogger(__name__)

# ── Reddit Configuration ─────────────────────────────────────────────

_REDDIT_BASE_URL = "https://www.reddit.com"
_REDDIT_USER_AGENT = "TradingAgents/1.0 (crypto research)"
_REDDIT_REQUEST_DELAY = 1.0
_REDDIT_RATE_LIMIT_DELAY = 30.0
_REDDIT_SEARCH_LIMIT = 25

_CRYPTO_SUBREDDITS = [
    "CryptoCurrency",
    "CryptoCurrencyTrading",
    "CoinBase",
    "Bitcoin",
    "ethereum",
    "solana",
    "defi",
]


# ── Reddit Helpers ───────────────────────────────────────────────────


def _reddit_get(url: str, params: dict | None = None) -> dict | list | None:
    """GET with rate-limit handling. Returns parsed JSON or None."""
    headers = {"User-Agent": _REDDIT_USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            logger.warning("Reddit 429 — backing off %ss", _REDDIT_RATE_LIMIT_DELAY)
            time.sleep(_REDDIT_RATE_LIMIT_DELAY)
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
        else:
            resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Reddit request failed (%s): %s", url[:80], e)
        return None
    finally:
        time.sleep(_REDDIT_REQUEST_DELAY)


def _search_subreddit(subreddit: str, query: str, time_filter: str = "week") -> list[dict]:
    """Search a subreddit for query via the public JSON endpoint."""
    url = f"{_REDDIT_BASE_URL}/r/{subreddit}/search.json"
    params = {
        "q": query,
        "restrict_sr": "on",
        "sort": "new",
        "t": time_filter,
        "limit": _REDDIT_SEARCH_LIMIT,
    }
    data = _reddit_get(url, params)
    if not data or not isinstance(data, dict):
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        try:
            d = child["data"]
            posts.append({
                "subreddit": d["subreddit"],
                "title": d["title"],
                "selftext": d.get("selftext", "")[:500],  # Truncate long posts
                "score": int(d.get("score", 0)),
                "num_comments": int(d.get("num_comments", 0)),
                "created_utc": float(d["created_utc"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return posts


def _fetch_hot_posts(subreddit: str) -> list[dict]:
    """Fetch hot posts from a subreddit."""
    url = f"{_REDDIT_BASE_URL}/r/{subreddit}/hot.json"
    params = {"limit": _REDDIT_SEARCH_LIMIT}
    data = _reddit_get(url, params)
    if not data or not isinstance(data, dict):
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        try:
            d = child["data"]
            if d.get("stickied"):
                continue
            posts.append({
                "subreddit": d["subreddit"],
                "title": d["title"],
                "selftext": d.get("selftext", "")[:500],
                "score": int(d.get("score", 0)),
                "num_comments": int(d.get("num_comments", 0)),
                "created_utc": float(d["created_utc"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return posts


# ── Google News Helpers ──────────────────────────────────────────────


def _fetch_google_news(query: str, days_back: int = 7) -> list[dict]:
    """Fetch news articles via the Google News RSS feed.

    The GoogleNews scraper library is unreliable (Google blocks HTML scraping),
    so we use the public RSS endpoint which returns structured XML reliably.
    """
    import re
    import xml.etree.ElementTree as ET
    from urllib.parse import quote_plus

    try:
        encoded_query = quote_plus(query)
        url = (
            f"https://news.google.com/rss/search?"
            f"q={encoded_query}+when:{days_back}d&hl=en-US&gl=US&ceid=US:en"
        )
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TradingAgents/1.0)"},
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        items = root.findall(".//item")

        articles = []
        for item in items[:20]:  # Limit to 20 articles
            title_el = item.find("title")
            desc_el = item.find("description")
            source_el = item.find("source")
            pub_date_el = item.find("pubDate")
            link_el = item.find("link")

            # Strip HTML tags and entities from description (RSS returns HTML snippets)
            raw_desc = desc_el.text if desc_el is not None else ""
            clean_desc = re.sub(r"<[^>]+>", "", raw_desc)
            clean_desc = re.sub(r"&nbsp;", " ", clean_desc)
            clean_desc = re.sub(r"&\w+;", "", clean_desc).strip()

            articles.append({
                "title": title_el.text if title_el is not None else "",
                "description": clean_desc,
                "source": source_el.text if source_el is not None else "Unknown",
                "date": pub_date_el.text if pub_date_el is not None else "",
                "link": link_el.text if link_el is not None else "",
            })
        return articles
    except Exception as e:
        logger.warning("Google News RSS fetch failed: %s", e)
        return []


# ── Public vendor interface functions ────────────────────────────────


def get_reddit_posts(
    coin_name: Annotated[str, "Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Solana')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch Reddit posts about a cryptocurrency from crypto subreddits.

    Returns raw post data (titles, content, scores, comment counts) for
    the LLM analyst to interpret sentiment. No automated sentiment scoring —
    the LLM performs qualitative analysis on the raw text.

    Searches multiple crypto subreddits for posts mentioning the coin.
    Posts are sorted by score (engagement) to surface the most discussed content.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days_diff = (end_dt - start_dt).days

    # Choose time filter based on date range
    if days_diff <= 7:
        time_filter = "week"
    elif days_diff <= 30:
        time_filter = "month"
    elif days_diff <= 365:
        time_filter = "year"
    else:
        time_filter = "all"

    all_posts = []

    # Search across crypto subreddits
    for subreddit in _CRYPTO_SUBREDDITS:
        try:
            posts = _search_subreddit(subreddit, coin_name, time_filter)
            all_posts.extend(posts)
        except Exception as e:
            logger.warning(f"Failed to search r/{subreddit}: {e}")
            continue

    # Also fetch hot posts from general crypto subreddits
    for subreddit in ["CryptoCurrency", "CryptoCurrencyTrading"]:
        try:
            hot = _fetch_hot_posts(subreddit)
            # Filter to posts mentioning the coin
            coin_lower = coin_name.lower()
            relevant = [
                p for p in hot
                if coin_lower in p["title"].lower() or coin_lower in p["selftext"].lower()
            ]
            all_posts.extend(relevant)
        except Exception as e:
            logger.warning(f"Failed to fetch hot posts from r/{subreddit}: {e}")

    if not all_posts:
        return (
            f"No Reddit posts found about {coin_name} in the specified period.\n"
            f"Searched subreddits: {', '.join(_CRYPTO_SUBREDDITS)}"
        )

    # Deduplicate by title
    seen_titles = set()
    unique_posts = []
    for p in all_posts:
        if p["title"] not in seen_titles:
            seen_titles.add(p["title"])
            unique_posts.append(p)

    # Sort by score (highest engagement first)
    unique_posts.sort(key=lambda x: x["score"], reverse=True)

    # Build formatted output
    header = f"# Reddit Posts about {coin_name}\n"
    header += f"# Period: {start_date} to {end_date}\n"
    header += f"# Total unique posts found: {len(unique_posts)}\n"
    header += f"# Subreddits searched: {', '.join(_CRYPTO_SUBREDDITS)}\n\n"

    body = ""
    for i, post in enumerate(unique_posts[:30], 1):  # Top 30 by score
        ts = datetime.fromtimestamp(post["created_utc"]).strftime("%Y-%m-%d %H:%M")
        body += f"### Post {i} (r/{post['subreddit']}) — Score: {post['score']}, Comments: {post['num_comments']}\n"
        body += f"**Date:** {ts}\n"
        body += f"**Title:** {post['title']}\n"
        if post["selftext"]:
            body += f"**Content:** {post['selftext']}\n"
        body += "\n"

    return header + body


def get_crypto_google_news(
    coin_name: Annotated[str, "Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch Google News articles about a cryptocurrency.

    Returns article titles, descriptions, and sources for the LLM analyst
    to interpret sentiment and identify key narratives.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days_back = max((end_dt - start_dt).days, 1)

    # Search with crypto-specific query variations
    queries = [
        f"{coin_name} crypto",
        f"{coin_name} cryptocurrency price",
    ]

    all_articles = []
    for query in queries:
        articles = _fetch_google_news(query, days_back=min(days_back, 30))
        all_articles.extend(articles)

    if not all_articles:
        return f"No Google News articles found about {coin_name} for the specified period."

    # Deduplicate by title
    seen_titles = set()
    unique_articles = []
    for a in all_articles:
        if a["title"] not in seen_titles:
            seen_titles.add(a["title"])
            unique_articles.append(a)

    header = f"# Google News: {coin_name}\n"
    header += f"# Period: {start_date} to {end_date}\n"
    header += f"# Articles found: {len(unique_articles)}\n\n"

    body = ""
    for i, article in enumerate(unique_articles[:20], 1):
        body += f"### Article {i}\n"
        body += f"**Title:** {article['title']}\n"
        body += f"**Source:** {article['source']}\n"
        if article["date"]:
            body += f"**Date:** {article['date']}\n"
        if article["description"]:
            body += f"**Summary:** {article['description']}\n"
        body += "\n"

    return header + body
