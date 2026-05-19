#!/usr/bin/env python3
"""Process one Practical AI episode through the full Phase-1 pipeline.

Usage:
    python scripts/process_one.py <episode_id>

This runs end-to-end against a single episode and writes an `Episode` JSON to
``data/episodes/<episode_id>/practical-ai-<episode_id>.json``. The slow step
(WhisperX alignment) is cached at ``whisper_raw.json`` in the same directory, so
re-runs after a successful first pass are fast.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `src.*` importable when running as a standalone script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import requests
from dotenv import load_dotenv

from src import align, match, rss, transcript
from src.schema import Episode

_TOTAL_STEPS = 6


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _step(idx: int, desc: str) -> None:
    print(f"[{idx}/{_TOTAL_STEPS}] {desc}", flush=True)


def _done(t0: float) -> None:
    print(f"    done in {_format_duration(time.monotonic() - t0)}\n", flush=True)


def _download_mp3(url: str, dest: Path) -> None:
    if dest.exists():
        mb = dest.stat().st_size / 1_000_000
        print(f"    [cached] {dest.name} ({mb:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_mb = int(r.headers.get("content-length", 0)) / 1_000_000
        print(f"    downloading {total_mb:.1f} MB → {dest}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        tmp.rename(dest)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="process_one.py",
        description=(
            "Run the full Phase-1 pipeline on one Practical AI episode:\n"
            "RSS → GitHub transcript → MP3 → WhisperX → match → JSON."
        ),
        epilog=(
            "Example:\n"
            "    python scripts/process_one.py 1\n\n"
            "The first run on a 30-minute episode takes ~30–40 min on CPU "
            "(WhisperX alignment is the dominant cost). Subsequent runs reuse "
            "the cached whisper_raw.json under data/episodes/<id>/ and finish "
            "in seconds.\n\n"
            "Set ANTHROPIC_API_KEY and HF_TOKEN in .env before running (see "
            ".env.example). HF_TOKEN also requires accepting two pyannote model "
            "licenses on Hugging Face — links are printed if you forget."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "episode_id",
        type=int,
        help="iTunes episode number (e.g. 1 for the very first episode)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Root directory for episode artifacts (default: ./data)",
    )
    parser.add_argument(
        "--force-realign",
        action="store_true",
        help="Re-run WhisperX even if whisper_raw.json exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()

    episode_id: int = args.episode_id
    episode_dir: Path = args.data_dir / "episodes" / str(episode_id)
    episode_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.monotonic()
    print(f"\nProcessing Practical AI episode #{episode_id}")
    print(f"Output: {episode_dir}\n")

    # -----------------------------------------------------------------------
    # 1/6 RSS metadata
    # -----------------------------------------------------------------------
    _step(1, "Fetching RSS metadata...")
    t0 = time.monotonic()
    try:
        meta = rss.find_episode(episode_id)
    except LookupError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2
    print(f"    title:    {meta.title}")
    print(f"    duration: {_format_duration(meta.duration_sec)}")
    print(f"    audio:    {meta.audio_url}")
    _done(t0)

    # -----------------------------------------------------------------------
    # 2/6 GitHub transcript
    # -----------------------------------------------------------------------
    _step(2, "Downloading GitHub transcript...")
    t0 = time.monotonic()
    try:
        github_turns = transcript.fetch_transcript(episode_id)
    except LookupError:
        url = transcript.transcript_url(episode_id)
        print(
            f"\nERROR: Episode {episode_id} transcript not yet published.\n"
            f"       The Changelog usually publishes transcripts within a few days\n"
            f"       of episode release. Check back later:\n"
            f"         {url}",
            file=sys.stderr,
        )
        return 3
    speakers = sorted({t.speaker for t in github_turns})
    print(f"    {len(github_turns)} turns by {len(speakers)} speakers: {speakers}")
    _done(t0)

    # -----------------------------------------------------------------------
    # 3/6 MP3
    # -----------------------------------------------------------------------
    _step(3, "Downloading MP3...")
    t0 = time.monotonic()
    audio_path = episode_dir / "audio.mp3"
    _download_mp3(meta.audio_url, audio_path)
    _done(t0)

    # -----------------------------------------------------------------------
    # 4/6 WhisperX alignment (heavy; cached)
    # -----------------------------------------------------------------------
    whisper_cache = episode_dir / "whisper_raw.json"
    _step(4, "Running WhisperX alignment...")
    t0 = time.monotonic()
    if whisper_cache.exists() and not args.force_realign:
        print(f"    [cached] reusing {whisper_cache.name}")
        whisper_words = align.words_from_json(json.loads(whisper_cache.read_text()))
        print(f"    loaded {len(whisper_words)} words")
    else:
        print(
            f"    expected wall time: ~{_format_duration(meta.duration_sec)} "
            "on CPU (≈real-time)"
        )
        try:
            whisper_words = align.transcribe_and_align(str(audio_path))
        except RuntimeError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 4
        whisper_cache.write_text(
            json.dumps(align.words_to_json(whisper_words), indent=2)
        )
        print(f"    cached {len(whisper_words)} words → {whisper_cache.name}")
    _done(t0)

    # -----------------------------------------------------------------------
    # 5/6 Match GitHub ↔ Whisper
    # -----------------------------------------------------------------------
    _step(5, "Matching transcript with Whisper output...")
    t0 = time.monotonic()
    try:
        segments, unaligned, stats = match.match(github_turns, whisper_words)
    except match.AlignmentQualityError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        print(
            "       Hint: a stale WhisperX cache or an episode/transcript mismatch\n"
            "       are the usual causes. Try `--force-realign` to discard the\n"
            f"       cached {whisper_cache.name} and re-transcribe.",
            file=sys.stderr,
        )
        return 5
    print(f"    {len(segments)} segments, {len(unaligned)} unaligned regions")
    total_tokens = stats.aligned_words + stats.unmatched_github_words
    print(f"    aligned {stats.aligned_words}/{total_tokens} GitHub tokens")
    print(f"    confidence: {stats.confidence_score:.3f}")
    _done(t0)

    # -----------------------------------------------------------------------
    # 6/6 Build Episode and write JSON
    # -----------------------------------------------------------------------
    _step(6, "Building Episode and saving JSON...")
    t0 = time.monotonic()
    episode = Episode(
        episode_id=meta.episode_id,
        title=meta.title,
        audio_url=meta.audio_url,
        duration_sec=meta.duration_sec,
        published_at=meta.published_at,
        speakers=speakers,
        segments=segments,
        unaligned_regions=unaligned,
        match_stats=stats,
    )
    output_path = episode_dir / f"practical-ai-{episode_id}.json"
    output_path.write_text(episode.model_dump_json(indent=2))
    print(f"    wrote {output_path}")
    _done(t0)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.monotonic() - overall_start
    print("=" * 60)
    print(f"Done in {_format_duration(total_elapsed)}.\n")
    print("Summary:")
    print(f"  episode:    #{meta.episode_id} — {meta.title}")
    print(f"  segments:   {len(segments)}")
    print(
        f"  unaligned:  {len(unaligned)} regions "
        f"({stats.unaligned_whisper_seconds:.1f}s total)"
    )
    print(f"  confidence: {stats.confidence_score:.3f}")
    print(f"  output:     {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
