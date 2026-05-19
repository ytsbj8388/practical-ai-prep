"""Practical AI RSS feed parser.

Returns lightweight `EpisodeMeta` records. Downstream stages (transcript fetch, audio
download, alignment) work off these.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser

FEED_URL = "https://changelog.com/practicalai/feed"


@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: int
    title: str
    audio_url: str
    duration_sec: float
    published_at: str  # ISO-8601 UTC


def _parse_duration(raw: str | None) -> float:
    """Convert an iTunes duration string to seconds.

    Accepts ``HH:MM:SS``, ``MM:SS``, or a bare seconds count — Changelog has used all
    three over the years.
    """
    if not raw:
        return 0.0
    if ":" in raw:
        parts = raw.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return 0.0
        if len(nums) == 3:
            h, m, s = nums
        elif len(nums) == 2:
            h, m, s = 0, nums[0], nums[1]
        else:
            return 0.0
        return float(h * 3600 + m * 60 + s)
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _parse_episode_id(entry) -> int | None:
    raw = entry.get("itunes_episode")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _parse_audio_url(entry) -> str | None:
    for enc in entry.get("enclosures") or []:
        href = enc.get("href") or enc.get("url")
        if href:
            return href
    return None


def _parse_published(entry) -> str:
    pp = entry.get("published_parsed")
    if pp:
        return datetime(*pp[:6], tzinfo=timezone.utc).isoformat()
    return entry.get("published", "")


def entry_to_meta(entry) -> EpisodeMeta | None:
    """Convert one feedparser entry to an `EpisodeMeta`.

    Returns None for entries that lack either an iTunes episode number or an audio
    enclosure — the feed occasionally includes bonus/teaser items without those, and
    they can't be processed by later stages.
    """
    episode_id = _parse_episode_id(entry)
    audio_url = _parse_audio_url(entry)
    if episode_id is None or audio_url is None:
        return None
    return EpisodeMeta(
        episode_id=episode_id,
        title=entry.get("title", ""),
        audio_url=audio_url,
        duration_sec=_parse_duration(entry.get("itunes_duration")),
        published_at=_parse_published(entry),
    )


def fetch_episodes(feed_url: str = FEED_URL) -> list[EpisodeMeta]:
    """Fetch and parse the feed; returns episodes in feed order (newest first)."""
    feed = feedparser.parse(feed_url)
    if feed.bozo and not feed.entries:
        raise RuntimeError(
            f"Failed to fetch/parse RSS feed {feed_url}: {feed.bozo_exception!r}"
        )
    return [meta for entry in feed.entries if (meta := entry_to_meta(entry)) is not None]


def find_episode(episode_id: int, feed_url: str = FEED_URL) -> EpisodeMeta:
    """Return metadata for one episode by iTunes episode number."""
    for meta in fetch_episodes(feed_url):
        if meta.episode_id == episode_id:
            return meta
    raise LookupError(f"Episode {episode_id} not in feed {feed_url}")
