"""Tests for `src.storage` — manifest schema, status transitions, retry_after.

The interesting behavior is the `retry_after` field added for soft-retry of
transcript 404s. We verify:
  - Round-trips through JSON (datetime ↔ ISO string with offset)
  - `mark_processed` / `mark_tagged` clear it on a successful transition
  - `list_failed_ready_to_retry` filters by elapsed retry_after
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import storage


def _seed_pending(manifest: storage.Manifest, episode_id: int) -> None:
    storage.mark_pending(
        manifest,
        episode_id,
        title=f"Episode {episode_id}",
        published_at="2026-01-01T00:00:00+00:00",
        audio_url=f"http://example.com/{episode_id}.mp3",
    )


# ---------------------------------------------------------------------------
# retry_after round-trip + clearing
# ---------------------------------------------------------------------------


def test_retry_after_roundtrip_json():
    """An aware datetime in retry_after survives save → load with type preserved."""
    m = storage.Manifest()
    _seed_pending(m, 1)
    retry_at = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    storage.mark_failed(m, 1, error="transcript 404", retry_after=retry_at)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        storage.save_manifest(m, path)
        m2 = storage.load_manifest(path)
    finally:
        path.unlink(missing_ok=True)

    loaded = storage.get(m2, 1)
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.retry_after == retry_at
    assert loaded.retry_after.tzinfo is not None, "retry_after must be tz-aware"


def test_other_failures_have_no_retry_after():
    """Permanent failures (e.g. alignment quality) leave retry_after = None."""
    m = storage.Manifest()
    _seed_pending(m, 1)
    storage.mark_failed(m, 1, error="alignment quality below threshold")
    rec = storage.get(m, 1)
    assert rec.status == "failed"
    assert rec.retry_after is None, "no retry_after kwarg → permanent fail"


def test_mark_processed_clears_retry_after():
    """A previously-soft-failed episode that succeeds on retry must lose retry_after."""
    m = storage.Manifest()
    _seed_pending(m, 1)
    retry_at = datetime.now(timezone.utc) + timedelta(days=3)
    storage.mark_failed(m, 1, error="transcript 404", retry_after=retry_at)
    assert storage.get(m, 1).retry_after is not None

    storage.mark_processed(m, 1, json_path="data/episodes/1/x.json", match_confidence=0.97)
    rec = storage.get(m, 1)
    assert rec.status == "processed"
    assert rec.retry_after is None, "mark_processed must clear retry_after"


def test_mark_tagged_clears_retry_after():
    m = storage.Manifest()
    _seed_pending(m, 1)
    retry_at = datetime.now(timezone.utc) + timedelta(days=3)
    storage.mark_failed(m, 1, error="transcript 404", retry_after=retry_at)
    storage.mark_processed(m, 1, json_path="data/episodes/1/x.json", match_confidence=0.97)
    storage.mark_tagged(m, 1, annotation_count=42)
    assert storage.get(m, 1).retry_after is None


# ---------------------------------------------------------------------------
# Candidate selection: list_failed_ready_to_retry
# ---------------------------------------------------------------------------


def test_daily_picks_up_expired_retry():
    """list_failed_ready_to_retry returns only failed episodes past their retry_after."""
    m = storage.Manifest()
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Episode 1: soft-failed, retry_after in the past → eligible
    _seed_pending(m, 1)
    storage.mark_failed(m, 1, error="transcript 404", retry_after=now - timedelta(days=1))

    # Episode 2: soft-failed, retry_after in the future → not yet eligible
    _seed_pending(m, 2)
    storage.mark_failed(m, 2, error="transcript 404", retry_after=now + timedelta(days=2))

    # Episode 3: permanently failed (retry_after=None) → never eligible via this helper
    _seed_pending(m, 3)
    storage.mark_failed(m, 3, error="alignment failed")

    # Episode 4: pending — different status, ignored by this helper
    _seed_pending(m, 4)

    ready = storage.list_failed_ready_to_retry(m, now)
    assert ready == [1], (
        f"only episode 1's retry_after has elapsed; got {ready}"
    )


def test_list_failed_ready_to_retry_empty_manifest():
    """No failures → empty list, not an error."""
    m = storage.Manifest()
    assert storage.list_failed_ready_to_retry(m, datetime.now(timezone.utc)) == []


# ---------------------------------------------------------------------------
# Backward compat: old manifest files (no retry_after key) still load
# ---------------------------------------------------------------------------


def test_old_manifest_without_retry_after_loads_cleanly():
    """A manifest written before retry_after existed must load without errors."""
    legacy_json = """{
      "version": 1,
      "last_updated": "2026-01-01T00:00:00+00:00",
      "episodes": {
        "1": {
          "episode_id": 1,
          "title": "Old episode",
          "published_at": "2018-07-02T17:00:00+00:00",
          "audio_url": "http://example.com/1.mp3",
          "status": "failed",
          "error": "transcript 404"
        }
      }
    }"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write(legacy_json)
        path = Path(f.name)
    try:
        m = storage.load_manifest(path)
    finally:
        path.unlink(missing_ok=True)

    rec = storage.get(m, 1)
    assert rec is not None
    assert rec.status == "failed"
    assert rec.retry_after is None, "missing field should default to None"
