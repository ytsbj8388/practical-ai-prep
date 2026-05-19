"""Tests for `src.match`.

Six hand-built synthetic fixtures cover the matcher's edge cases:

    1. exact_match     — every GitHub token has a matching Whisper token
    2. missing_words   — Whisper dropped a word ("um", "uh") — must interpolate
    3. extra_words     — Whisper hallucinated a word — must become an unaligned region
    4. speaker_change  — diarization splits mid-turn, GitHub says one speaker — GitHub wins
    5. number_format   — "GPT-4" (GitHub) vs "gpt four" (Whisper) — must still match
    6. ad_insertion    — sponsor read with no GitHub counterpart — unaligned region

Plus one integration test on real episode-1 data, which auto-skips until the fixture
files have been recorded (see test docstring for setup).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.align import AlignedWord, words_from_json
from src.match import MAX_UNALIGNED_RATIO, AlignmentQualityError, match
from src.transcript import TranscriptTurn, parse_transcript

_REAL_DIR = Path(__file__).parent / "fixtures" / "real"
_REAL_WHISPER = _REAL_DIR / "practical-ai-1.whisper.json"
_REAL_TRANSCRIPT = _REAL_DIR / "practical-ai-1.transcript.md"


def _words(rows: list[tuple[str, float, float, str | None]]) -> list[AlignedWord]:
    """Sugar so the fixtures below read as ``(text, start, end, speaker)`` rows."""
    return [AlignedWord(word=w, start=s, end=e, speaker=sp) for w, s, e, sp in rows]


# ---------------------------------------------------------------------------
# 1. exact_match
# ---------------------------------------------------------------------------


def test_exact_match():
    """Baseline: every GitHub token has a matching Whisper token; no interpolation."""
    gh = [
        TranscriptTurn(speaker="Daniel Whitenack", text="Welcome to Practical AI."),
        TranscriptTurn(speaker="Chris Benson", text="Thanks for having me."),
    ]
    wh = _words(
        [
            ("welcome", 0.0, 0.3, "SPEAKER_00"),
            ("to", 0.3, 0.5, "SPEAKER_00"),
            ("practical", 0.5, 1.0, "SPEAKER_00"),
            ("AI", 1.0, 1.3, "SPEAKER_00"),
            ("thanks", 1.5, 1.8, "SPEAKER_01"),
            ("for", 1.8, 2.0, "SPEAKER_01"),
            ("having", 2.0, 2.4, "SPEAKER_01"),
            ("me", 2.4, 2.6, "SPEAKER_01"),
        ]
    )

    segments, unaligned, stats = match(gh, wh)

    assert len(segments) == 2, f"expected 2 segments (1 per turn), got {len(segments)}"
    assert segments[0].speaker == "Daniel Whitenack"
    assert segments[1].speaker == "Chris Benson"

    seg0_words = [w.w for w in segments[0].words]
    assert seg0_words == ["Welcome", "to", "Practical", "AI."], (
        f"first segment should preserve raw GitHub tokens, got {seg0_words}"
    )

    # Direct (non-interpolated) timing carries through verbatim from Whisper
    assert segments[0].words[0].s == pytest.approx(0.0)
    assert segments[0].words[3].e == pytest.approx(1.3)
    assert segments[1].words[0].s == pytest.approx(1.5)

    assert stats.aligned_words == 8, f"all 8 GH tokens should align, got {stats}"
    assert stats.unmatched_github_words == 0
    assert stats.confidence_score == pytest.approx(1.0)
    assert stats.unaligned_whisper_seconds == pytest.approx(0.0)
    assert unaligned == [], f"no Whisper-side gaps expected, got {unaligned}"


# ---------------------------------------------------------------------------
# 2. missing_words
# ---------------------------------------------------------------------------


def test_missing_words_interpolates_dropped_word():
    """Whisper dropped 'we'. Its time must be interpolated between the neighbours."""
    gh = [
        TranscriptTurn(
            speaker="Daniel Whitenack",
            text="I think we are starting now.",
        )
    ]
    wh = _words(
        [
            ("I", 0.0, 0.1, "SPEAKER_00"),
            ("think", 0.2, 0.5, "SPEAKER_00"),
            # "we" dropped here
            ("are", 0.9, 1.1, "SPEAKER_00"),
            ("starting", 1.2, 1.6, "SPEAKER_00"),
            ("now", 1.7, 2.0, "SPEAKER_00"),
        ]
    )

    segments, unaligned, stats = match(gh, wh)

    assert len(segments) == 1
    seg = segments[0]
    assert [w.w for w in seg.words] == ["I", "think", "we", "are", "starting", "now."], (
        f"all 6 GitHub tokens should appear in segment.words, got {[w.w for w in seg.words]}"
    )

    we = seg.words[2]
    think_end = seg.words[1].e
    are_start = seg.words[3].s
    assert think_end <= we.s <= we.e <= are_start, (
        f"interpolated 'we' time ({we.s}, {we.e}) must sit between think.e={think_end} "
        f"and are.s={are_start}"
    )

    assert stats.aligned_words == 5
    assert stats.unmatched_github_words == 1
    assert stats.confidence_score == pytest.approx(5 / 6, abs=1e-3)
    assert unaligned == [], (
        f"missing_words should produce no Whisper-side gaps, got {unaligned}"
    )


# ---------------------------------------------------------------------------
# 3. extra_words
# ---------------------------------------------------------------------------


def test_extra_words_becomes_unaligned_region():
    """Whisper hallucinated 'um'. It should not appear in segments — it goes to
    `unaligned_regions` instead.
    """
    gh = [TranscriptTurn(speaker="Daniel Whitenack", text="Hello everyone.")]
    wh = _words(
        [
            ("hello", 0.0, 0.3, "SPEAKER_00"),
            ("um", 0.4, 0.5, "SPEAKER_00"),  # hallucinated
            ("everyone", 0.6, 1.0, "SPEAKER_00"),
        ]
    )

    segments, unaligned, stats = match(gh, wh)

    assert len(segments) == 1
    assert [w.w for w in segments[0].words] == ["Hello", "everyone."], (
        "Whisper hallucinations must not leak into Segment.words"
    )
    assert stats.aligned_words == 2
    assert stats.unmatched_github_words == 0
    assert stats.confidence_score == pytest.approx(1.0)

    assert len(unaligned) == 1, (
        f"the hallucinated 'um' should appear as one unaligned region, got {unaligned}"
    )
    region = unaligned[0]
    assert region.whisper_text.lower() == "um", (
        f"unaligned region text should be 'um', got {region.whisper_text!r}"
    )
    assert region.start == pytest.approx(0.4)
    assert region.end == pytest.approx(0.5)
    assert region.reason == "unknown", (
        f"a hallucination between matched content should classify as 'unknown' "
        f"(not intro/outro), got {region.reason!r}"
    )


# ---------------------------------------------------------------------------
# 4. speaker_change
# ---------------------------------------------------------------------------


def test_speaker_change_keeps_github_attribution():
    """Cross-talk: GitHub attributes the whole sentence to Adam, but Whisper's
    diarization briefly assigns SPEAKER_01 to one word in the middle. Segment.speaker
    must come from GitHub — diarization noise must not change attribution.
    """
    gh = [
        TranscriptTurn(
            speaker="Adam Stacoviak",
            text="I think Jerod that is exactly right.",
        )
    ]
    wh = _words(
        [
            ("I", 0.0, 0.1, "SPEAKER_00"),
            ("think", 0.2, 0.5, "SPEAKER_00"),
            ("Jerod", 0.6, 0.9, "SPEAKER_00"),
            ("that", 1.0, 1.2, "SPEAKER_01"),  # diarization slip
            ("is", 1.3, 1.4, "SPEAKER_00"),
            ("exactly", 1.5, 1.9, "SPEAKER_00"),
            ("right", 2.0, 2.3, "SPEAKER_00"),
        ]
    )

    segments, _unaligned, stats = match(gh, wh)

    assert len(segments) == 1
    assert segments[0].speaker == "Adam Stacoviak", (
        f"speaker must come from GitHub, not Whisper diarization; "
        f"got {segments[0].speaker!r}"
    )
    assert len(segments[0].words) == 7, "all seven GH tokens should align cleanly"
    assert stats.aligned_words == 7
    assert stats.confidence_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. number_format
# ---------------------------------------------------------------------------


def test_number_format_match():
    """``"GPT-4"`` (GitHub, hyphenated form) must align with ``"gpt four"`` (Whisper,
    spoken form). The matcher does this by splitting hyphens and mapping number words
    (``"four"`` → ``"4"``) before alignment.
    """
    gh = [TranscriptTurn(speaker="Chris Benson", text="We use GPT-4 for this.")]
    wh = _words(
        [
            ("we", 0.0, 0.1, "SPEAKER_00"),
            ("use", 0.2, 0.4, "SPEAKER_00"),
            ("gpt", 0.5, 0.8, "SPEAKER_00"),
            ("four", 0.9, 1.2, "SPEAKER_00"),  # spoken form
            ("for", 1.3, 1.4, "SPEAKER_00"),
            ("this", 1.5, 1.7, "SPEAKER_00"),
        ]
    )

    segments, unaligned, stats = match(gh, wh)

    assert len(segments) == 1
    seg = segments[0]
    word_strs = [w.w for w in seg.words]
    assert word_strs == ["We", "use", "GPT", "4", "for", "this."], (
        f'"GPT-4" must split into "GPT" + "4" so it can align to "gpt four", '
        f"got words={word_strs}"
    )
    assert stats.aligned_words == 6, f"all 6 tokens should align, got {stats}"
    assert stats.unmatched_github_words == 0
    assert stats.confidence_score == pytest.approx(1.0)
    assert unaligned == [], f"no Whisper-side gaps expected, got {unaligned}"

    # The split "GPT" and "4" should inherit Whisper's "gpt" and "four" timings
    gpt_idx = word_strs.index("GPT")
    four_idx = word_strs.index("4")
    assert seg.words[gpt_idx].s == pytest.approx(0.5)
    assert seg.words[four_idx].s == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 6. ad_insertion
# ---------------------------------------------------------------------------


def test_ad_insertion_appears_as_unaligned_region():
    """Sponsor read in the audio (no GitHub counterpart). The ad must end up in
    `unaligned_regions`, never in `segments` — segments are study material and ads are
    not. All GitHub tokens should still align with full confidence.
    """
    gh = [
        TranscriptTurn(speaker="Daniel Whitenack", text="Welcome to the show."),
        TranscriptTurn(speaker="Daniel Whitenack", text="Today we have a great guest."),
    ]
    wh = _words(
        [
            ("welcome", 0.0, 0.3, "SPEAKER_00"),
            ("to", 0.3, 0.5, "SPEAKER_00"),
            ("the", 0.5, 0.7, "SPEAKER_00"),
            ("show", 0.7, 1.0, "SPEAKER_00"),
            # --- Ad insert ---
            ("this", 2.0, 2.2, "SPEAKER_03"),
            ("episode", 2.2, 2.6, "SPEAKER_03"),
            ("brought", 2.6, 2.9, "SPEAKER_03"),
            ("to", 2.9, 3.0, "SPEAKER_03"),
            ("you", 3.0, 3.2, "SPEAKER_03"),
            ("by", 3.2, 3.4, "SPEAKER_03"),
            ("Fastly", 3.4, 3.8, "SPEAKER_03"),
            # --- Back to content ---
            ("today", 5.0, 5.3, "SPEAKER_00"),
            ("we", 5.3, 5.4, "SPEAKER_00"),
            ("have", 5.4, 5.6, "SPEAKER_00"),
            ("a", 5.6, 5.7, "SPEAKER_00"),
            ("great", 5.7, 6.0, "SPEAKER_00"),
            ("guest", 6.0, 6.4, "SPEAKER_00"),
        ]
    )

    segments, unaligned, stats = match(gh, wh)

    assert len(segments) == 2
    total_gh_words = sum(len(s.words) for s in segments)
    assert total_gh_words == 10, (
        f"both turns' 10 GitHub tokens must all land in segments, got {total_gh_words}"
    )
    assert stats.aligned_words == 10
    assert stats.unmatched_github_words == 0
    assert stats.confidence_score == pytest.approx(1.0)

    assert len(unaligned) == 1, (
        f"the ad should appear as one contiguous unaligned region, got {len(unaligned)}: "
        f"{unaligned}"
    )
    ad = unaligned[0]
    assert "fastly" in ad.whisper_text.lower(), (
        f"unaligned region should contain 'fastly', got {ad.whisper_text!r}"
    )
    assert ad.start == pytest.approx(2.0)
    assert ad.end == pytest.approx(3.8)
    assert ad.reason == "unknown", (
        f"an ad in the middle of the episode (between matched content) classifies "
        f"as 'unknown', got {ad.reason!r}"
    )
    assert stats.unaligned_whisper_seconds == pytest.approx(1.8, abs=0.01)


# ---------------------------------------------------------------------------
# Bonus: quality gate guardrail
# ---------------------------------------------------------------------------


def test_quality_gate_raises_when_too_few_match():
    """If most GitHub tokens can't find a Whisper counterpart, the matcher refuses to
    return a junk alignment — it raises so the caller can investigate (wrong audio,
    wrong transcript pairing, badly miscued WhisperX, …).

    To force GitHub-side gaps (which are what `unmatched_github_words` counts), Whisper
    has to be *shorter* than GitHub by more than the threshold. Same-length-but-totally-
    different sequences don't trigger this — NW would pair them as mismatches instead,
    which still produce a timing for each GitHub token.
    """
    gh = [
        TranscriptTurn(
            speaker="X",
            text="alpha beta gamma delta epsilon zeta eta theta iota kappa",
        )
    ]
    # Only 2 of 10 GitHub tokens have a Whisper counterpart — 80% gaps, well past 30%.
    wh = _words(
        [
            ("alpha", 0.0, 0.1, "SPEAKER_00"),
            ("beta", 0.2, 0.3, "SPEAKER_00"),
        ]
    )

    with pytest.raises(AlignmentQualityError) as excinfo:
        match(gh, wh)
    assert f"{MAX_UNALIGNED_RATIO:.0%}" in str(excinfo.value), (
        f"error message should mention the threshold for debuggability, got: {excinfo.value}"
    )


# ---------------------------------------------------------------------------
# 7. Real integration test — practical-ai-1 first ~5 minutes
# ---------------------------------------------------------------------------


def _real_fixtures_present() -> bool:
    return _REAL_WHISPER.exists() and _REAL_TRANSCRIPT.exists()


@pytest.mark.skipif(
    not _real_fixtures_present(),
    reason=(
        "Real-episode fixtures missing. To enable:\n"
        "  1. Run `python scripts/process_one.py 1` once (≈30–40 min on CPU)\n"
        "  2. Save the first ~5 minutes of WhisperX word output as "
        f"{_REAL_WHISPER.relative_to(_REAL_DIR.parent.parent)}\n"
        "     (JSON list of {word, start, end, speaker} objects — exactly the format "
        "src.align.words_to_json produces)\n"
        "  3. Save the matching transcript excerpt as "
        f"{_REAL_TRANSCRIPT.relative_to(_REAL_DIR.parent.parent)}\n"
        "     (the raw GitHub Markdown, trimmed to the same time range)"
    ),
)
def test_real_practical_ai_1_smoke():
    """Smoke test against real WhisperX output for the first ~5 minutes of episode 1.

    We don't have a hand-labelled ground truth, so this test only enforces invariants
    that must hold for ANY correct alignment:

    - Match confidence ≥ 0.85 (real audio, real transcript, some drift is OK)
    - All four expected hosts/guests appear at least once in segments
    - Segments are in non-decreasing start-time order
    - Words within each segment are in non-decreasing start-time order

    Per-speaker first-utterance ground truth (±2s) can be layered on top once you've
    listened through the excerpt — add a `practical-ai-1.expected.json` fixture next
    to the other two and extend this test to read it.
    """
    whisper_words = words_from_json(json.loads(_REAL_WHISPER.read_text()))
    github_turns = parse_transcript(_REAL_TRANSCRIPT.read_text())

    segments, unaligned, stats = match(github_turns, whisper_words)

    assert stats.confidence_score >= 0.85, (
        f"real-episode confidence {stats.confidence_score:.2f} is below the 0.85 floor "
        f"— alignment is likely broken (stats={stats})"
    )

    expected = {"Adam Stacoviak", "Jerod Santo", "Chris Benson", "Daniel Whitenack"}
    seen = {s.speaker for s in segments}
    missing = expected - seen
    assert not missing, (
        f"expected hosts/guests missing from segments: {sorted(missing)} "
        f"(saw {sorted(seen)})"
    )

    for prev, curr in zip(segments, segments[1:]):
        assert prev.start <= curr.start, (
            f"segments out of time order: {prev.id} starts at {prev.start:.2f}s, "
            f"but {curr.id} starts at {curr.start:.2f}s"
        )

    for seg in segments:
        for w_prev, w_curr in zip(seg.words, seg.words[1:]):
            assert w_prev.s <= w_curr.s, (
                f"words in {seg.id} out of time order: "
                f"{w_prev.w!r}@{w_prev.s:.2f}s then {w_curr.w!r}@{w_curr.s:.2f}s"
            )
