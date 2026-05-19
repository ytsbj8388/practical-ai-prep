#!/usr/bin/env python3
"""Tag one episode JSON with Claude annotations.

Usage:
    python scripts/tag_one.py <episode_id>
    python scripts/tag_one.py <episode_id> --dry-run     # first 3 segments only, no save

Reads ``data/episodes/<id>/practical-ai-<id>.json``, runs `src.tag.tag_episode` to
populate per-segment difficulty + annotations and episode-level summaries, then
overwrites the JSON in place (with a ``.bak`` backup of the previous version).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from src.schema import Episode
from src.tag import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_COST_USD,
    TagStats,
    tag_episode,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tag_one.py",
        description=(
            "Tag one Practical AI episode's segments with vocab / turn-taking / "
            "domain annotations via Claude Haiku 4.5."
        ),
        epilog=(
            "Examples:\n"
            "    python scripts/tag_one.py 1\n"
            "        Tag every segment in episode 1; save back to JSON.\n\n"
            "    python scripts/tag_one.py 1 --dry-run\n"
            "        Tag only the first 3 segments and print the result to stdout.\n"
            "        Use this while tuning the prompt.\n\n"
            "Cost: roughly $0.05–$0.30 per episode at default settings. The script "
            "aborts and saves what it has if --max-cost is exceeded.\n\n"
            "Requires ANTHROPIC_API_KEY in .env (see .env.example)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "episode_id",
        type=int,
        help="iTunes episode number — must already have a processed JSON on disk",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Root directory for episode artifacts (default: ./data)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Segments per Claude call (default {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=DEFAULT_MAX_COST_USD,
        dest="max_cost_usd",
        help=(
            f"USD ceiling per run (default ${DEFAULT_MAX_COST_USD:.2f}). Aborts and "
            "saves partial results if the running cost passes this."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Tag only the first 3 segments, print results, do not save",
    )
    return p.parse_args(argv)


def _print_progress(done: int, total: int, stats: TagStats) -> None:
    print(
        f"\r  Tagged {done}/{total} segments  "
        f"({stats.total_input_tokens:,}+{stats.total_output_tokens:,} tokens, "
        f"${stats.estimated_cost_usd:.4f})",
        end="",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY not set.\n"
            "       Add it to .env (see .env.example) and try again.",
            file=sys.stderr,
        )
        return 2

    json_path = (
        args.data_dir
        / "episodes"
        / str(args.episode_id)
        / f"practical-ai-{args.episode_id}.json"
    )
    if not json_path.exists():
        print(
            f"ERROR: {json_path} not found.\n"
            f"       Run `python scripts/process_one.py {args.episode_id}` first.",
            file=sys.stderr,
        )
        return 3

    episode = Episode.model_validate_json(json_path.read_text())
    original_total = len(episode.segments)

    print(f"\nTagging episode #{args.episode_id} — {episode.title}")
    print(f"Segments: {original_total}")
    if args.dry_run:
        episode.segments = episode.segments[:3]
        print(f"[DRY RUN] Tagging only the first {len(episode.segments)} segments; not saving")
    print()

    t0 = time.monotonic()
    try:
        tagged, stats = tag_episode(
            episode,
            batch_size=args.batch_size,
            max_cost_usd=args.max_cost_usd,
            progress_callback=_print_progress,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        return 130
    except RuntimeError as e:
        # tag_episode raises RuntimeError for unrecoverable setup issues
        # (e.g. missing API key — already checked above, but defense in depth).
        print(f"\n\nERROR: {e}", file=sys.stderr)
        return 4
    elapsed = time.monotonic() - t0

    print()  # newline after the carriage-return progress line
    print()
    print("Summary:")
    print(f"  Tagged:      {stats.tagged_segments}/{len(episode.segments)} segments")
    if stats.failed_segments:
        print(f"  Failed:      {stats.failed_segments} (annotations=[])")
    if stats.dropped_annotations:
        print(
            f"  Dropped:     {stats.dropped_annotations} annotations "
            "(expression didn't match segment.words)"
        )
    print(
        f"  Tokens:      {stats.total_input_tokens:,} in + "
        f"{stats.total_output_tokens:,} out"
    )
    print(f"  Cost:        ${stats.estimated_cost_usd:.4f}")
    print(f"  Annotations: {dict(stats.annotation_counts)}")
    print(f"  Time:        {elapsed:.1f}s")
    if stats.cost_limit_hit:
        print(
            f"  WARNING:     stopped early because cost passed "
            f"${args.max_cost_usd:.2f} ceiling"
        )

    if args.dry_run:
        print("\n[DRY RUN] Tagged segments:")
        for seg in tagged.segments:
            preview = seg.text[:100] + ("..." if len(seg.text) > 100 else "")
            print(f"\n  [{seg.id}] {seg.speaker}: {preview}")
            print(f"    difficulty: {seg.difficulty}")
            if not seg.annotations:
                print("    annotations: (none)")
            for ann in seg.annotations:
                print(
                    f"    • {ann.type:11s} '{ann.expression}' → {ann.ko}  "
                    f"(words[{ann.word_idx[0]}..{ann.word_idx[1]}])"
                )
        return 0

    # Backup current JSON (one level of undo) and overwrite.
    backup = json_path.with_suffix(json_path.suffix + ".bak")
    backup.write_text(json_path.read_text())
    json_path.write_text(tagged.model_dump_json(indent=2))
    print(f"\n  Saved:       {json_path}")
    print(f"  Backup:      {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
