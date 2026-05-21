"""manifest.json read/write — per-episode processing state across runs.

The manifest is the source of truth for "what's been done": RSS discovery populates
new episodes as `pending`, the pipeline transitions them through `processed` →
`tagged`, and any uncaught exception lands them in `failed` with an error string.
The status machine is intentionally small (no `skipped` — a missing GitHub
transcript is just a `failed` with a specific message) so cron retries and human
debugging both have one table to look at.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

MANIFEST_VERSION = 1
DEFAULT_MANIFEST_PATH = Path("data/manifest.json")

EpisodeStatus = Literal["pending", "processed", "tagged", "failed"]


class EpisodeRecord(BaseModel):
    """One episode's state. Forward-compatible: unknown fields ignored on load."""

    model_config = ConfigDict(extra="ignore")

    episode_id: int
    title: str
    published_at: str  # ISO 8601 — keep full timestamp from RSS
    audio_url: str
    status: EpisodeStatus = "pending"
    json_path: str | None = None
    processed_at: str | None = None
    tagged_at: str | None = None
    match_confidence: float | None = None
    annotation_count: int | None = None
    error: str | None = None
    # When set, the orchestrator will pick this `failed` record up again in
    # `daily` / `backfill` mode once `now >= retry_after`. Only used for soft
    # failures (transcript not yet published); permanent failures leave this None.
    retry_after: AwareDatetime | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = MANIFEST_VERSION
    last_updated: str = ""
    episodes: dict[str, EpisodeRecord] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Manifest:
    """Read manifest from disk; return a fresh empty one if the file doesn't exist."""
    if not path.exists():
        return Manifest()
    return Manifest.model_validate_json(path.read_text())


def save_manifest(manifest: Manifest, path: Path = DEFAULT_MANIFEST_PATH) -> None:
    """Atomic write — temp file + rename so a crash mid-write can't leave a partial JSON."""
    manifest.last_updated = _utcnow()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort episodes by integer id for stable diffs in git.
    sorted_episodes = {
        k: manifest.episodes[k]
        for k in sorted(manifest.episodes, key=lambda s: int(s))
    }
    manifest.episodes = sorted_episodes
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2, exclude_none=False))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Episode-level helpers
# ---------------------------------------------------------------------------


def get(manifest: Manifest, episode_id: int) -> EpisodeRecord | None:
    return manifest.episodes.get(str(episode_id))


def get_status(manifest: Manifest, episode_id: int) -> EpisodeStatus | None:
    rec = get(manifest, episode_id)
    return rec.status if rec else None


def upsert(manifest: Manifest, record: EpisodeRecord) -> None:
    manifest.episodes[str(record.episode_id)] = record


def mark_pending(
    manifest: Manifest,
    episode_id: int,
    *,
    title: str,
    published_at: str,
    audio_url: str,
) -> None:
    """Insert if absent (used during RSS discovery). Doesn't overwrite existing records."""
    if str(episode_id) in manifest.episodes:
        return
    manifest.episodes[str(episode_id)] = EpisodeRecord(
        episode_id=episode_id,
        title=title,
        published_at=published_at,
        audio_url=audio_url,
        status="pending",
    )


def mark_processed(
    manifest: Manifest,
    episode_id: int,
    *,
    json_path: str,
    match_confidence: float,
) -> None:
    rec = _require(manifest, episode_id)
    rec.status = "processed"
    rec.json_path = json_path
    rec.match_confidence = match_confidence
    rec.processed_at = _utcnow()
    rec.error = None
    rec.retry_after = None  # successful transition clears any pending retry window


def mark_tagged(
    manifest: Manifest,
    episode_id: int,
    *,
    annotation_count: int,
) -> None:
    rec = _require(manifest, episode_id)
    rec.status = "tagged"
    rec.tagged_at = _utcnow()
    rec.annotation_count = annotation_count
    rec.error = None
    rec.retry_after = None


def mark_failed(
    manifest: Manifest,
    episode_id: int,
    error: str,
    *,
    retry_after: datetime | None = None,
) -> None:
    """Mark `failed`. Pass `retry_after` to schedule a soft retry (daily mode will
    pick the record back up once now ≥ retry_after). Omit for permanent failures."""
    rec = _require(manifest, episode_id)
    rec.status = "failed"
    rec.error = error
    rec.retry_after = retry_after


def list_by_status(manifest: Manifest, status: EpisodeStatus) -> list[int]:
    """Episode IDs at the given status, sorted ascending."""
    return sorted(
        int(k) for k, v in manifest.episodes.items() if v.status == status
    )


def list_pending(manifest: Manifest) -> list[int]:
    return list_by_status(manifest, "pending")


def list_tagged(manifest: Manifest) -> list[int]:
    return list_by_status(manifest, "tagged")


def list_failed_ready_to_retry(manifest: Manifest, now: datetime) -> list[int]:
    """Failed episodes whose `retry_after` has elapsed. `now` must be tz-aware."""
    return sorted(
        int(k)
        for k, v in manifest.episodes.items()
        if v.status == "failed"
        and v.retry_after is not None
        and v.retry_after <= now
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require(manifest: Manifest, episode_id: int) -> EpisodeRecord:
    rec = manifest.episodes.get(str(episode_id))
    if rec is None:
        raise KeyError(f"Episode {episode_id} not in manifest")
    return rec
