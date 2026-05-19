"""Match GitHub transcript text against WhisperX word-level output.

Pipeline:

1. Tokenize both sides — GitHub turns flatten to a single token sequence (we remember
   which turn each token came from); Whisper words are already a flat sequence.
2. Banded Needleman-Wunsch on the normalized token sequences. Banding is critical:
   without it the DP grid is O(M·N) which blows up for hour-long episodes (≈10k×10k
   cells). We band the diagonal by ±BAND_RATIO of the longer sequence.
3. For each GitHub token, take the matched Whisper word's timing. Tokens that didn't
   match get linearly interpolated from their nearest matched neighbours.
4. Whisper words that didn't match anything become `UnalignedRegion`s (intros, ads,
   outros, etc). Region boundaries are classified by position relative to the matched
   span.
5. Build one `Segment` per non-empty GitHub turn, with `Segment.text` preserving the
   original cleaned GitHub text and `Segment.words` carrying word-level timings.
6. If too few GitHub tokens aligned, raise `AlignmentQualityError` — beyond that point
   the per-word timings are not trustworthy enough to be worth saving.

Speaker mapping (SPEAKER_XX → real name) is intentionally skipped in Phase 1; the
real name comes directly from the GitHub turn structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.align import AlignedWord
from src.schema import MatchStats, Segment, UnalignedRegion, Word
from src.transcript import TranscriptTurn

# Banded NW parameters
BAND_RATIO = 0.05
MIN_BAND_RADIUS = 100

# Quality gate
MAX_UNALIGNED_RATIO = 0.30

# NW scoring
MATCH_SCORE = 2.0
MISMATCH_SCORE = -1.0
GAP_PENALTY = -1.0

# Extrapolation padding when a GitHub token sits outside the matched-time span
_EXTRAPOLATION_PAD_SEC = 0.05
_NEG_INF = -1e18

# Traceback opcodes
_TRACE_STOP = 0
_TRACE_DIAG = 1
_TRACE_UP = 2  # gap on Whisper side
_TRACE_LEFT = 3  # gap on GitHub side


class AlignmentQualityError(RuntimeError):
    """Raised when fewer than (1 − MAX_UNALIGNED_RATIO) of GitHub tokens aligned."""


# ---------------------------------------------------------------------------
# Tokenization / normalization
# ---------------------------------------------------------------------------

_TOKEN_SPLIT_RE = re.compile(r"[\s\-—–]+")
_STRIP_OUTER_PUNCT_RE = re.compile(r"^\W+|\W+$", re.UNICODE)

# Spoken-form ↔ digit equivalence — keeps the "GPT-4" vs "gpt four" case from costing us
# matches. Intentionally small; the goal is the common case, not exhaustive coverage.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def _split_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT_RE.split(text) if t]


def _normalize(token: str) -> str:
    s = _STRIP_OUTER_PUNCT_RE.sub("", token).lower()
    return _NUMBER_WORDS.get(s, s)


@dataclass(frozen=True)
class _GhToken:
    turn_idx: int
    raw: str  # preserves original casing + inner punctuation (e.g. "don't", "GPT")
    norm: str  # for matching


def _flatten_github(turns: list[TranscriptTurn]) -> list[_GhToken]:
    out: list[_GhToken] = []
    for turn_idx, turn in enumerate(turns):
        for raw in _split_tokens(turn.text):
            norm = _normalize(raw)
            if not norm:
                continue
            out.append(_GhToken(turn_idx=turn_idx, raw=raw, norm=norm))
    return out


@dataclass(frozen=True)
class _WhToken:
    """A Whisper subtoken — sometimes a Whisper "word" expands to multiple subtokens
    (e.g. ``"GPT-4"`` → ``["gpt", "4"]``). All subtokens of one Whisper word inherit
    that word's start/end timing.
    """

    source_idx: int  # index into the original whisper_words list
    norm: str
    start: float
    end: float
    speaker: str | None


def _flatten_whisper(words: list[AlignedWord]) -> list[_WhToken]:
    out: list[_WhToken] = []
    for i, w in enumerate(words):
        for raw in _split_tokens(w.word):
            norm = _normalize(raw)
            if not norm:
                continue
            out.append(
                _WhToken(
                    source_idx=i,
                    norm=norm,
                    start=w.start,
                    end=w.end,
                    speaker=w.speaker,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Banded Needleman-Wunsch
# ---------------------------------------------------------------------------


def _banded_nw(
    seq_a: list[str], seq_b: list[str]
) -> list[tuple[int | None, int | None]]:
    """Return alignment pairs ``(i, j)`` in source order.

    ``i`` is an index into ``seq_a`` (or None for a gap on the a-side), ``j`` into
    ``seq_b`` (or None for a gap on the b-side). At least one of ``i``/``j`` is always
    not-None.
    """
    M, N = len(seq_a), len(seq_b)
    if M == 0:
        return [(None, j) for j in range(N)]
    if N == 0:
        return [(i, None) for i in range(M)]

    band_radius = max(MIN_BAND_RADIUS, int(max(M, N) * BAND_RATIO))
    W = 2 * band_radius + 1

    # Storage shifted along the diagonal: row i's band covers j in
    #     [j_offsets[i], j_offsets[i] + W - 1]
    # When j is out of band, cells stay at _NEG_INF and are never the argmax.
    dp = np.full((M + 1, W), _NEG_INF, dtype=np.float64)
    trace = np.zeros((M + 1, W), dtype=np.int8)
    j_offsets = np.empty(M + 1, dtype=np.int64)
    for i in range(M + 1):
        center = int(round(i * N / M))
        j_offsets[i] = max(0, center - band_radius)

    def _get(i: int, j: int) -> float:
        k = j - int(j_offsets[i])
        if 0 <= k < W:
            return float(dp[i][k])
        return _NEG_INF

    # Base case: dp[0][0] = 0 (both empty), first row/column are gap-only paths
    k0 = -int(j_offsets[0])  # j=0
    if 0 <= k0 < W:
        dp[0][k0] = 0.0
    for j in range(1, N + 1):
        k = j - int(j_offsets[0])
        if 0 <= k < W:
            dp[0][k] = j * GAP_PENALTY
            trace[0][k] = _TRACE_LEFT
    for i in range(1, M + 1):
        k = -int(j_offsets[i])
        if 0 <= k < W:
            dp[i][k] = i * GAP_PENALTY
            trace[i][k] = _TRACE_UP

    for i in range(1, M + 1):
        center = int(round(i * N / M))
        j_lo = max(1, center - band_radius)
        j_hi = min(N, center + band_radius)
        a_tok = seq_a[i - 1]
        for j in range(j_lo, j_hi + 1):
            k = j - int(j_offsets[i])
            if k < 0 or k >= W:
                continue
            sub = MATCH_SCORE if a_tok == seq_b[j - 1] else MISMATCH_SCORE
            diag = _get(i - 1, j - 1) + sub
            up = _get(i - 1, j) + GAP_PENALTY
            left = _get(i, j - 1) + GAP_PENALTY
            if diag >= up and diag >= left:
                dp[i][k] = diag
                trace[i][k] = _TRACE_DIAG
            elif up >= left:
                dp[i][k] = up
                trace[i][k] = _TRACE_UP
            else:
                dp[i][k] = left
                trace[i][k] = _TRACE_LEFT

    # Traceback. (M, N) is virtually always in-band for similar-length sequences; if
    # the bandwidth was too narrow we fall back to the best score on the last row.
    end_i, end_j = M, N
    end_k = N - int(j_offsets[M])
    if not (0 <= end_k < W):
        best = _NEG_INF
        for k in range(W):
            j = k + int(j_offsets[M])
            if 0 <= j <= N and float(dp[M][k]) > best:
                best = float(dp[M][k])
                end_j = j

    pairs: list[tuple[int | None, int | None]] = []
    i, j = end_i, end_j
    while i > 0 or j > 0:
        k = j - int(j_offsets[i])
        if k < 0 or k >= W:
            # Path escaped the band — bail and emit the remainder as gaps so the
            # caller still gets a usable (if degraded) alignment.
            while i > 0:
                pairs.append((i - 1, None))
                i -= 1
            while j > 0:
                pairs.append((None, j - 1))
                j -= 1
            break
        t = int(trace[i][k])
        if t == _TRACE_DIAG:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif t == _TRACE_UP:
            pairs.append((i - 1, None))
            i -= 1
        elif t == _TRACE_LEFT:
            pairs.append((None, j - 1))
            j -= 1
        else:
            break
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Time interpolation for unmatched GitHub tokens
# ---------------------------------------------------------------------------


def _interpolate_times(
    M: int,
    matched_times: dict[int, tuple[float, float]],
) -> list[tuple[float, float]]:
    """Fill timing for every GitHub token, interpolating between matched neighbours.

    Tokens before the first match / after the last match get a tiny pad on either side
    of the nearest known time rather than a wild extrapolation — anchoring to known
    reality is safer than guessing.
    """
    if M == 0:
        return []
    if not matched_times:
        return [(0.0, 0.0)] * M

    times: list[tuple[float, float] | None] = [None] * M
    for i, t in matched_times.items():
        times[i] = t

    # For each unmatched index, find the nearest previous and next matched index in
    # two linear passes.
    prev_match: list[int | None] = [None] * M
    pv: int | None = None
    for i in range(M):
        if times[i] is not None:
            pv = i
        prev_match[i] = pv

    next_match: list[int | None] = [None] * M
    nx: int | None = None
    for i in range(M - 1, -1, -1):
        if times[i] is not None:
            nx = i
        next_match[i] = nx

    out: list[tuple[float, float]] = []
    for i in range(M):
        if times[i] is not None:
            out.append(times[i])  # type: ignore[arg-type]
            continue
        pi, ni = prev_match[i], next_match[i]
        if pi is not None and ni is not None:
            ps, pe = times[pi]  # type: ignore[misc]
            ns, ne = times[ni]  # type: ignore[misc]
            prev_mid = (ps + pe) / 2
            next_mid = (ns + ne) / 2
            frac = (i - pi) / (ni - pi)
            mid = prev_mid + frac * (next_mid - prev_mid)
            # Split the gap evenly across the unmatched tokens between pi and ni so
            # interpolated word widths are consistent.
            slot = (next_mid - prev_mid) / (ni - pi)
            half = max(slot / 2, _EXTRAPOLATION_PAD_SEC / 2)
            out.append((mid - half, mid + half))
        elif pi is not None:
            _, pe = times[pi]  # type: ignore[misc]
            out.append((pe, pe + _EXTRAPOLATION_PAD_SEC))
        elif ni is not None:
            ns, _ = times[ni]  # type: ignore[misc]
            out.append((max(0.0, ns - _EXTRAPOLATION_PAD_SEC), ns))
        else:
            out.append((0.0, 0.0))
    return out


# ---------------------------------------------------------------------------
# Unaligned Whisper regions
# ---------------------------------------------------------------------------


def _extract_unaligned_regions(
    pairs: list[tuple[int | None, int | None]],
    wh_tokens: list[_WhToken],
    whisper_words: list[AlignedWord],
) -> list[UnalignedRegion]:
    """Group consecutive Whisper-side gaps into regions, one per original audio span.

    We classify regions by their position relative to the matched span: a region that
    ends before the first matched word is an `intro`, one that starts after the last
    matched word is an `outro`, everything in between is `unknown` (could be a sponsor
    read, a music break, or just hallucinated transcript).
    """
    first_matched_time = next(
        (
            wh_tokens[w].start
            for g, w in pairs
            if g is not None and w is not None
        ),
        None,
    )
    last_matched_time = next(
        (
            wh_tokens[w].end
            for g, w in reversed(pairs)
            if g is not None and w is not None
        ),
        None,
    )

    # Walk the alignment, grouping consecutive (None, w) pairs. We also collapse
    # subtokens that map to the same original Whisper word so the surfaced text reads
    # naturally (one row per original word, not per subtoken).
    regions: list[UnalignedRegion] = []
    current_source_ids: list[int] = []  # original whisper_words indices
    current_seen: set[int] = set()

    def flush() -> None:
        if not current_source_ids:
            return
        words = [whisper_words[i] for i in current_source_ids]
        start = min(w.start for w in words)
        end = max(w.end for w in words)
        text = " ".join(w.word.strip() for w in words).strip()
        if first_matched_time is not None and end <= first_matched_time:
            reason = "intro"
        elif last_matched_time is not None and start >= last_matched_time:
            reason = "outro"
        else:
            reason = "unknown"
        regions.append(
            UnalignedRegion(start=start, end=end, whisper_text=text, reason=reason)
        )

    for g, w in pairs:
        if g is None and w is not None:
            src = wh_tokens[w].source_idx
            if src not in current_seen:
                current_source_ids.append(src)
                current_seen.add(src)
        else:
            flush()
            current_source_ids = []
            current_seen = set()
    flush()
    return regions


# ---------------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------------


def _build_segments(
    github_turns: list[TranscriptTurn],
    gh_tokens: list[_GhToken],
    token_times: list[tuple[float, float]],
) -> list[Segment]:
    by_turn: dict[int, list[int]] = {}
    for idx, tok in enumerate(gh_tokens):
        by_turn.setdefault(tok.turn_idx, []).append(idx)

    segments: list[Segment] = []
    for turn_idx, turn in enumerate(github_turns):
        indices = by_turn.get(turn_idx, [])
        if not indices:
            continue
        words = [
            Word(w=gh_tokens[i].raw, s=token_times[i][0], e=token_times[i][1])
            for i in indices
        ]
        segments.append(
            Segment(
                id=f"seg-{turn_idx:04d}",
                speaker=turn.speaker,
                start=min(w.s for w in words),
                end=max(w.e for w in words),
                text=turn.text,
                words=words,
            )
        )
    return segments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def match(
    github_turns: list[TranscriptTurn],
    whisper_words: list[AlignedWord],
) -> tuple[list[Segment], list[UnalignedRegion], MatchStats]:
    """Align Whisper word-level output against an authoritative GitHub transcript.

    Returns segments (one per non-empty GitHub turn, carrying real speaker names and
    per-word timing), Whisper-only regions, and matching diagnostics.

    Raises:
        AlignmentQualityError: when more than `MAX_UNALIGNED_RATIO` of the GitHub
            tokens couldn't be matched to a Whisper token — beyond that point the
            per-word timings are mostly interpolation and not worth saving.
    """
    gh_tokens = _flatten_github(github_turns)
    wh_tokens = _flatten_whisper(whisper_words)
    total_gh = len(gh_tokens)
    if total_gh == 0:
        raise ValueError("GitHub transcript yielded no tokens after normalization")

    pairs = _banded_nw([t.norm for t in gh_tokens], [t.norm for t in wh_tokens])

    matched_times: dict[int, tuple[float, float]] = {}
    for g, w in pairs:
        if g is not None and w is not None:
            wh = wh_tokens[w]
            matched_times[g] = (wh.start, wh.end)

    matched_count = len(matched_times)
    unmatched_gh = total_gh - matched_count
    unaligned_ratio = unmatched_gh / total_gh
    if unaligned_ratio > MAX_UNALIGNED_RATIO:
        raise AlignmentQualityError(
            f"Only {matched_count}/{total_gh} GitHub tokens aligned "
            f"({unaligned_ratio:.1%} unmatched, threshold "
            f"{MAX_UNALIGNED_RATIO:.0%}). Whisper transcription likely diverges "
            "from the GitHub transcript — check audio/transcript episode pairing."
        )

    token_times = _interpolate_times(total_gh, matched_times)
    segments = _build_segments(github_turns, gh_tokens, token_times)
    unaligned_regions = _extract_unaligned_regions(pairs, wh_tokens, whisper_words)

    stats = MatchStats(
        aligned_words=matched_count,
        unmatched_github_words=unmatched_gh,
        unaligned_whisper_seconds=sum(r.end - r.start for r in unaligned_regions),
        confidence_score=matched_count / total_gh,
    )
    return segments, unaligned_regions, stats
