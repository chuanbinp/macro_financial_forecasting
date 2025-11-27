from langchain.tools import tool
import feedparser
from datetime import datetime, timedelta, timezone
from typing import List, Dict

@tool
def get_bloomberg_rss_feeds(days: int = 1) -> List[Dict[str, str]]:
    """Fetch Bloomberg RSS news items for the last N days."""

    from config import Config
    config = Config()
    feeds = config.rss_feeds[:3]

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    news = []

    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if hasattr(entry, "published_parsed"):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                if published > cutoff:
                    news.append({
                        "Headline": entry.title,
                        "Link": entry.link,
                        "Article": entry.summary,
                        "Date": published.isoformat(),
                    })
    return news