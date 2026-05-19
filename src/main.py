"""Batch orchestrator for the Practical AI episode pipeline.

Three modes:

    python -m src.main daily              # pending only, small --limit, cron-friendly
    python -m src.main backfill --limit N # explicit fill; --limit required
    python -m src.main retry              # re-attempt `failed` entries

Every mode (1) discovers RSS episodes into the manifest, (2) iterates a candidate list
filtered by mode, (3) runs per-episode {process → tag} with continue-on-error and a
cumulative cost ceiling, and (4) persists the manifest after every episode so a crash
or timeout leaves a recoverable state behind.

The per-episode pipeline below intentionally duplicates `scripts/process_one.py` and
`scripts/tag_one.py` in compact form rather than calling into them — the scripts have
verbose stepped output that's useful for a single run but noisy in batch logs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Allow `python -m src.main` to import sibling modules without an installed package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests
from dotenv import load_dotenv

from src import align, rss, storage, tag, transcript
from src.match import AlignmentQualityError, match as run_match
from src.schema import Episode

DEFAULT_DAILY_LIMIT = 5
DEFAULT_RETRY_LIMIT = 10
DEFAULT_MAX_COST_USD = 5.0

# Skip starting another episode when we're this close to the ceiling. A typical episode
# tag costs $0.03–$0.10, so a $0.10 floor prevents a half-budgeted episode from getting
# cut off mid-batch.
_MIN_BUDGET_TO_START_USD = 0.10


# ---------------------------------------------------------------------------
# Per-episode pipeline (compact: one line per phase)
# ---------------------------------------------------------------------------


def _download_mp3(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    tmp.rename(dest)


def _run_processing_pipeline(episode_id: int, data_dir: Path) -> Episode:
    """RSS → transcript → MP3 → WhisperX (cached) → match → built Episode.

    Raises on any failure; the caller is responsible for marking the manifest entry
    as `failed` with the exception message.
    """
    meta = rss.find_episode(episode_id)
    turns = transcript.fetch_transcript(episode_id)

    episode_dir = data_dir / "episodes" / str(episode_id)
    episode_dir.mkdir(parents=True, exist_ok=True)

    audio_path = episode_dir / "audio.mp3"
    _download_mp3(meta.audio_url, audio_path)

    whisper_cache = episode_dir / "whisper_raw.json"
    if whisper_cache.exists():
        words = align.words_from_json(json.loads(whisper_cache.read_text()))
    else:
        words = align.transcribe_and_align(str(audio_path))
        whisper_cache.write_text(
            json.dumps(align.words_to_json(words), indent=2)
        )

    segments, unaligned, stats = run_match(turns, words)

    return Episode(
        episode_id=meta.episode_id,
        title=meta.title,
        audio_url=meta.audio_url,
        duration_sec=meta.duration_sec,
        published_at=meta.published_at,
        speakers=sorted({t.speaker for t in turns}),
        segments=segments,
        unaligned_regions=unaligned,
        match_stats=stats,
    )


def _episode_json_path(data_dir: Path, episode_id: int) -> Path:
    return (
        data_dir
        / "episodes"
        / str(episode_id)
        / f"practical-ai-{episode_id}.json"
    )


def _save_episode_json(episode: Episode, path: Path) -> None:
    """Atomic save with a .bak of the previous version (one-level undo)."""
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(episode.model_dump_json(indent=2))
    tmp.replace(path)


def _process_one_episode(
    episode_id: int,
    manifest: storage.Manifest,
    *,
    data_dir: Path,
    manifest_path: Path,
    tag_budget_remaining: float,
) -> tuple[float, str]:
    """Run whichever phases this episode still needs. Returns (cost_used, status_word).

    Status words: ``"tagged"`` (newly tagged), ``"already_tagged"`` (no-op),
    ``"failed"``. Manifest is saved after every status transition so partial progress
    survives a crash.
    """
    rec = storage.get(manifest, episode_id)
    if rec is None:
        return 0.0, "failed"  # shouldn't happen — discovery runs first

    json_path = _episode_json_path(data_dir, episode_id)

    # Phase 1: process (download + WhisperX + match) — only if not yet done
    episode: Episode | None = None
    if rec.status in ("pending", "failed"):
        try:
            episode = _run_processing_pipeline(episode_id, data_dir)
            _save_episode_json(episode, json_path)
            storage.mark_processed(
                manifest,
                episode_id,
                json_path=str(json_path.relative_to(data_dir.parent)),
                match_confidence=(
                    episode.match_stats.confidence_score
                    if episode.match_stats
                    else 0.0
                ),
            )
            storage.save_manifest(manifest, manifest_path)
        except (LookupError, AlignmentQualityError) as e:
            # Expected, debuggable failures — record the message, move on.
            storage.mark_failed(manifest, episode_id, error=f"process: {e}")
            storage.save_manifest(manifest, manifest_path)
            return 0.0, "failed"
        except Exception as e:
            # Unexpected — still continue with the run but capture a stack trace
            # in the manifest so it can be debugged later.
            storage.mark_failed(
                manifest,
                episode_id,
                error=f"process (unexpected): {e}",
            )
            storage.save_manifest(manifest, manifest_path)
            traceback.print_exc(file=sys.stderr)
            return 0.0, "failed"

    # Phase 2: tag — skip if already tagged
    rec = storage.get(manifest, episode_id)
    assert rec is not None  # we just upserted above
    if rec.status == "tagged":
        return 0.0, "already_tagged"

    if episode is None:
        # We didn't process in this run (status was already 'processed') — load JSON
        episode = Episode.model_validate_json(json_path.read_text())

    try:
        tagged, stats = tag.tag_episode(
            episode, max_cost_usd=tag_budget_remaining
        )
    except RuntimeError as e:
        # Configuration / setup errors (e.g. missing ANTHROPIC_API_KEY) — these
        # would affect every episode, so we want to stop the whole run, not just
        # mark this episode failed.
        raise
    except Exception as e:
        storage.mark_failed(manifest, episode_id, error=f"tag: {e}")
        storage.save_manifest(manifest, manifest_path)
        traceback.print_exc(file=sys.stderr)
        return 0.0, "failed"

    _save_episode_json(tagged, json_path)
    total_annotations = sum(len(s.annotations) for s in tagged.segments)
    storage.mark_tagged(manifest, episode_id, annotation_count=total_annotations)
    storage.save_manifest(manifest, manifest_path)

    if stats.cost_limit_hit:
        print(
            f"    WARNING: tagging hit cost limit mid-episode; partial coverage "
            f"({stats.tagged_segments}/{len(tagged.segments)} segments tagged)",
            file=sys.stderr,
        )

    return stats.estimated_cost_usd, "tagged"


# ---------------------------------------------------------------------------
# Discovery + candidate selection
# ---------------------------------------------------------------------------


def _discover_episodes(manifest: storage.Manifest) -> int:
    """Pull the RSS feed and insert any new episode IDs as `pending`. Returns count."""
    feed_episodes = rss.fetch_episodes()
    new = 0
    for meta in feed_episodes:
        if str(meta.episode_id) not in manifest.episodes:
            storage.mark_pending(
                manifest,
                meta.episode_id,
                title=meta.title,
                published_at=meta.published_at,
                audio_url=meta.audio_url,
            )
            new += 1
    return new


def _reconcile_with_disk(
    manifest: storage.Manifest, data_dir: Path
) -> tuple[int, int]:
    """Promote `pending` entries whose JSON already exists on disk.

    Lets the orchestrator pick up state from a partial first run (or someone using
    process_one.py / tag_one.py directly) without forcing the user to hand-edit
    manifest.json. Returns ``(promoted_to_processed, promoted_to_tagged)``.
    """
    promoted_processed = 0
    promoted_tagged = 0
    for ep_id in storage.list_pending(manifest):
        json_path = _episode_json_path(data_dir, ep_id)
        if not json_path.exists():
            continue
        try:
            episode = Episode.model_validate_json(json_path.read_text())
        except Exception:
            # Corrupted JSON — leave as pending; the next process run will overwrite it
            continue

        rel_path = str(json_path.relative_to(data_dir.parent))
        confidence = (
            episode.match_stats.confidence_score if episode.match_stats else 0.0
        )
        storage.mark_processed(
            manifest, ep_id, json_path=rel_path, match_confidence=confidence
        )
        promoted_processed += 1

        total_annotations = sum(len(s.annotations) for s in episode.segments)
        if total_annotations > 0:
            storage.mark_tagged(
                manifest, ep_id, annotation_count=total_annotations
            )
            promoted_tagged += 1
    return promoted_processed, promoted_tagged


def _candidates_for(
    manifest: storage.Manifest, mode: str, *, order: str
) -> list[int]:
    if mode == "retry":
        return sorted(storage.list_by_status(manifest, "failed"), reverse=True)
    # daily / backfill: anything that still has work to do
    todo = storage.list_by_status(manifest, "pending") + storage.list_by_status(
        manifest, "processed"
    )
    return sorted(todo, reverse=(order == "newest"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Root directory for episode artifacts + manifest.json (default: ./data)",
    )
    common.add_argument(
        "--max-cost",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        dest="max_cost_usd",
        help=(
            f"Cumulative USD ceiling for Claude tagging across this run "
            f"(default ${DEFAULT_MAX_COST_USD:.2f}). Episodes are skipped once "
            "this is reached."
        ),
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description=(
            "Batch orchestrator for the Practical AI episode pipeline: RSS → "
            "transcript → MP3 → WhisperX → match → Claude tagging."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True, metavar="MODE")

    p_daily = sub.add_parser(
        "daily",
        parents=[common],
        help="Cron-friendly: process pending episodes (newest first, small limit)",
    )
    p_daily.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_DAILY_LIMIT,
        help=f"Max episodes this run (default {DEFAULT_DAILY_LIMIT})",
    )
    p_daily.add_argument(
        "--order",
        choices=["newest", "oldest"],
        default="newest",
    )

    p_back = sub.add_parser(
        "backfill",
        parents=[common],
        help="Manual fill of pending/processed episodes; --limit required",
    )
    p_back.add_argument(
        "--limit",
        type=int,
        required=True,
        help=(
            "Max episodes this run — REQUIRED. GitHub Actions free tier caps a job "
            "at 6h; one episode is ~30 min on CPU, so 10 is a safe upper bound there."
        ),
    )
    p_back.add_argument(
        "--order",
        choices=["newest", "oldest"],
        default="newest",
    )

    p_retry = sub.add_parser(
        "retry",
        parents=[common],
        help="Re-attempt episodes marked `failed` (newest failure first)",
    )
    p_retry.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RETRY_LIMIT,
        help=f"Max episodes this run (default {DEFAULT_RETRY_LIMIT})",
    )

    args = parser.parse_args(argv)
    # retry doesn't carry --order; default it for downstream code
    if not hasattr(args, "order"):
        args.order = "newest"
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()

    data_dir: Path = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = storage.load_manifest(manifest_path)

    overall_start = time.monotonic()
    print(f"\nmain.py — mode={args.mode}, data_dir={data_dir}")
    print(f"  manifest: {manifest_path} ({len(manifest.episodes)} episodes known)")

    # 1. RSS discovery — always run; cheap, and surfaces brand-new episodes
    try:
        new_count = _discover_episodes(manifest)
    except Exception as e:
        print(f"\nERROR fetching RSS feed: {e}", file=sys.stderr)
        return 4
    # 1a. Reconcile pending entries against disk so previously-processed episodes
    # don't get re-run on first manifest creation.
    promoted_proc, promoted_tag = _reconcile_with_disk(manifest, data_dir)
    storage.save_manifest(manifest, manifest_path)
    if new_count:
        print(f"  RSS discovery: +{new_count} new episodes added as pending")
    if promoted_proc:
        print(
            f"  Disk reconcile: {promoted_proc} pending → processed "
            f"({promoted_tag} of which already tagged)"
        )

    # 2. Candidate selection. Always slice — `--limit 0` is a legitimate "nothing
    # to do this run" (useful for dry-runs of discovery + reconcile).
    candidates = _candidates_for(manifest, args.mode, order=args.order)
    candidates = candidates[: args.limit]

    if not candidates:
        print("\nNothing to do.")
        return 0

    print(f"\nProcessing {len(candidates)} episode(s)...\n")

    # 3. Iterate with continue-on-error + cost ceiling
    total_cost = 0.0
    counts: dict[str, int] = {
        "tagged": 0,
        "already_tagged": 0,
        "failed": 0,
        "skipped_budget": 0,
    }

    try:
        for i, ep_id in enumerate(candidates, start=1):
            remaining = args.max_cost_usd - total_cost
            if remaining < _MIN_BUDGET_TO_START_USD:
                counts["skipped_budget"] += len(candidates) - i + 1
                print(
                    f"[{i}/{len(candidates)}] Cost ceiling reached "
                    f"(${total_cost:.4f} / ${args.max_cost_usd:.2f}) — "
                    f"skipping remaining {len(candidates) - i + 1} episode(s)."
                )
                break

            rec = storage.get(manifest, ep_id)
            title = (rec.title[:60] + "...") if rec and len(rec.title) > 60 else (
                rec.title if rec else "?"
            )
            print(f"[{i}/{len(candidates)}] #{ep_id}: {title}")
            t0 = time.monotonic()
            cost, result = _process_one_episode(
                ep_id,
                manifest,
                data_dir=data_dir,
                manifest_path=manifest_path,
                tag_budget_remaining=remaining,
            )
            total_cost += cost
            counts[result] = counts.get(result, 0) + 1
            elapsed = time.monotonic() - t0

            if result == "tagged":
                print(
                    f"    → tagged in {elapsed:.0f}s "
                    f"(${cost:.4f}; cumulative ${total_cost:.4f})"
                )
            elif result == "already_tagged":
                print("    → already tagged, no-op")
            elif result == "failed":
                err = storage.get(manifest, ep_id)
                err_msg = err.error if err else "unknown"
                print(f"    → FAILED: {err_msg}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted — manifest is up to date.", file=sys.stderr)
        return 130
    except RuntimeError as e:
        # Setup-level failure (e.g. missing API key); bail entire run
        print(f"\nFATAL: {e}", file=sys.stderr)
        return 4

    # 4. Summary
    elapsed = time.monotonic() - overall_start
    print("\n=== Summary ===")
    print(f"  Tagged:        {counts['tagged']}")
    print(f"  Already done:  {counts['already_tagged']}")
    print(f"  Failed:        {counts['failed']}")
    if counts["skipped_budget"]:
        print(f"  Skipped (budget): {counts['skipped_budget']}")
    print(f"  Cost total:    ${total_cost:.4f} / ${args.max_cost_usd:.2f}")
    print(f"  Wall time:     {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
