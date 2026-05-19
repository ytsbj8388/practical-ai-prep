"""Pydantic models for the processed-episode JSON.

Phase 1 produces everything except the tagging fields (`difficulty`, `annotations`,
`*_summary`); those stay at their defaults until `tag.py` runs in Phase 2, so the same
schema validates Phase 1 and Phase 2 outputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")

    w: str
    s: float
    e: float


AnnotationType = Literal["vocab", "turn_taking", "domain"]


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AnnotationType
    expression: str
    ko: str = Field(max_length=25)
    word_idx: tuple[int, int]


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    speaker: str
    start: float
    end: float
    text: str
    words: list[Word]
    difficulty: int | None = Field(default=None, ge=1, le=5)
    annotations: list[Annotation] = Field(default_factory=list)


UnalignedReason = Literal["intro", "ad", "outro", "unknown"]


class UnalignedRegion(BaseModel):
    """Audio span present in the Whisper transcript but absent from the GitHub one.

    Kept out of `Episode.segments` (not study material) but retained so a downstream
    consumer — e.g. the iOS app's "skip ads" feature — can use them.
    """

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    whisper_text: str
    reason: UnalignedReason = "unknown"


class MatchStats(BaseModel):
    """Diagnostics from `match.py`. Use during alignment-algorithm tuning."""

    model_config = ConfigDict(extra="forbid")

    aligned_words: int
    unmatched_github_words: int
    unaligned_whisper_seconds: float
    confidence_score: float = Field(ge=0.0, le=1.0)


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: int
    title: str
    audio_url: str
    duration_sec: float
    published_at: str
    speakers: list[str]
    segments: list[Segment]
    unaligned_regions: list[UnalignedRegion] = Field(default_factory=list)
    match_stats: MatchStats | None = None
    vocab_summary: list[str] = Field(default_factory=list)
    domain_summary: list[str] = Field(default_factory=list)
    turn_taking_summary: list[str] = Field(default_factory=list)
