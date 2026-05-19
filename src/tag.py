"""Tag an Episode with vocab / turn-taking / domain annotations via Claude Haiku 4.5.

Pipeline:

1. Group segments into batches (default 8).
2. For each batch, send a single `client.messages.parse()` call with a Pydantic
   response schema — Anthropic's structured-output mode enforces the schema, so we
   never have to do raw JSON parsing or repair.
3. Map the LLM's `expression` strings back to `word_idx` ranges by scanning
   `Segment.words` with the same normalization match.py uses (lowercase, outer
   punctuation stripped, hyphens split, simple number-word equivalence). Drop any
   annotation whose expression we can't locate — wrong indices are worse than absent
   annotations.
4. Track input/output tokens per batch and abort if the running cost crosses
   `max_cost_usd` so a runaway prompt never burns silent dollars.

Per-batch retry covers both transient API errors and Pydantic validation failures.
If a batch fails after all retries, every segment in that batch gets
`annotations=[]` (and `difficulty` stays unset) — the rest of the episode keeps
going. The episode-level summaries are deduplicated lowercase expressions per type.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

# Reuse the matcher's tokenization so a "GPT-4" expression aligns with the same
# split-on-hyphen, lowercase, strip-outer-punct convention Segment.words already
# uses. Importing the private names is intentional — keep them in sync with match.py.
from src.match import _normalize as _normalize_token
from src.match import _split_tokens
from src.schema import Annotation, AnnotationType, Episode, Segment

# --- Model + pricing -------------------------------------------------------

MODEL_ID = "claude-haiku-4-5-20251001"

# Haiku 4.5 list price as of skill cache 2026-04-29 — $1/Mtok in, $5/Mtok out.
# Override via the constants if pricing shifts.
HAIKU_INPUT_USD_PER_MTOK = 1.0
HAIKU_OUTPUT_USD_PER_MTOK = 5.0

# --- Defaults --------------------------------------------------------------

DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_COST_USD = 2.0
DEFAULT_MAX_RETRIES_PER_BATCH = 3
DEFAULT_MAX_OUTPUT_TOKENS = 4096


# --- LLM response schema (Pydantic models that Anthropic enforces) ---------


class _LLMAnnotation(BaseModel):
    type: AnnotationType
    expression: str
    ko: str = Field(max_length=25)


class _LLMSegment(BaseModel):
    segment_id: str
    difficulty: int = Field(ge=1, le=5)
    annotations: list[_LLMAnnotation] = Field(default_factory=list)


class _LLMBatch(BaseModel):
    segments: list[_LLMSegment]


# --- System prompt ---------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an English-learning coach for a Korean SENIOR AI engineer working in storage/MLOps \
(MLPerf Training, MLPerf Storage WG). Assume the user already knows all standard ML/DL \
terminology — they live this stuff daily. Their pain is fast-paced English conversation:

1. Figurative / idiomatic expressions and phrasal verbs they can't parse on the fly
2. Short, reusable conversational moves (turn-taking patterns)
3. NICHE technical vocabulary beyond standard ML knowledge

Tag noteworthy expressions of THREE types. Be selective — fewer high-quality tags beat \
many noisy ones. That said, don't be overly conservative — a phrase that's standard \
business English (like "flesh out" or "edge case") is still valuable for a non-native \
speaker even if it's common in English.

**vocab** — idioms, phrasal verbs, figurative/colloquial expressions, business metaphors. \
This is the most common type. Skip everyday GRE-prep vocabulary.
TARGETS include:
  - Classic idioms / metaphors: "wow me", "go back a while", "in the trenches", \
"knock it out of the park", "we're dropping the ball", "throw in the towel", \
"kick the can down the road"
  - Common business idioms: "edge case", "low-hanging fruit", "moving target", \
"commercial viability", "value proposition"
  - Engineering meeting phrasal verbs: "flesh out", "circle back", "double down on", \
"iron out", "drill down into"
  - Conversational intensifiers: "mind-blowing", "game-changing", "no-brainer", \
"spot-on", "off the charts"
These last three categories are exactly the expressions the user hears in meetings and \
needs for production-grade English — DO NOT skip them as "too basic". A senior Korean \
engineer with deep ML knowledge still gets tripped up by everyday business-meeting English.

**turn_taking** — SHORT (≤5 words), REUSABLE patterns that signal a conversational move: \
responding, building, redirecting, pushing back, emphasizing. Must be a phrase the user \
could memorize and reuse in their own meetings.
TARGETS: "to your point", "I'd push back", "the thing is", "having said that", \
"if I may", "let me build on that", "fair point", "that said", "in fairness"
SKIP — these are NOT turn_taking:
  - Full sentences: "There are so many different levels to answer that question on"
  - Segment-specific actions: "I was kind of poking Chris" (one-time, not a pattern)
  - Anything over 5 words
  - Idioms / phrasal verbs even if conversational — those go in **vocab**.
    e.g. "we're dropping the ball" → vocab (idiom), NOT turn_taking.

**domain** — NICHE technical vocabulary the user likely does NOT already know.
SKIP (already familiar to any MLOps engineer):
  - Generic terms: AI, ML, machine learning, deep learning, neural networks, \
data science, big data, GPU, training, inference, model, dataset, MLOps, kubernetes, PyTorch
  - Generic concepts: infrastructure, AI community, AI ethics, AI safety
  - Proper nouns / event names: "GPU Conference", "NeurIPS", product names
  - Non-technical compound nouns: "theoretical physicist", "academic research", \
"Academia and research"
TARGETS (worth flagging — specific subfields, techniques, or cross-domain jargon):
  - "Symbolic AI", "neuro-symbolic", "object detection", "speculative decoding"
  - "model distillation", "RAG pipeline", "MoE", "constitutional AI"
  - "atomic and molecular physics" (cross-domain context shifts)
Rule of thumb: if you'd find it in a generic AI-101 syllabus, SKIP. If it's a \
specialty subfield or technique name, TAG.

Also rate each segment's overall difficulty 1–5 (1 = trivial, 5 = dense slang/jargon).

CRITICAL CONSTRAINTS:

- `expression` MUST appear VERBATIM in the segment (case-insensitive, exact word \
sequence). Downstream code does a literal word-by-word match and DROPS any annotation \
that doesn't align. Example: text says "going back a while" → tag "going back a while", \
NOT "go back a while".

- `ko` field: write what a Korean would ACTUALLY say in the same situation — NOT a \
literal word-for-word translation. Aim for 15 Korean characters or fewer (hard limit 25).
  GOOD: "goes back a while" → "꽤 오래됐다"
  BAD:  "goes back a while" → "오랜 시간 전으로 거슬러 올라감"
  GOOD: "I'd push back" → "반대하고 싶다"
  BAD:  "I'd push back" → "나는 뒤로 밀어버릴 것이다"
  GOOD: "in the trenches" → "현장에서 직접"
  BAD:  "in the trenches" → "참호 안에서"

- If a segment has nothing worth tagging, return empty annotations. Don't force tags.
"""


# --- Public API ------------------------------------------------------------


@dataclass
class TagStats:
    tagged_segments: int = 0
    failed_segments: int = 0
    dropped_annotations: int = 0  # LLM emitted but expression didn't match Segment.words
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    annotation_counts: Counter = field(default_factory=Counter)
    cost_limit_hit: bool = False

    def update_cost(self) -> None:
        self.estimated_cost_usd = (
            self.total_input_tokens / 1_000_000 * HAIKU_INPUT_USD_PER_MTOK
            + self.total_output_tokens / 1_000_000 * HAIKU_OUTPUT_USD_PER_MTOK
        )


ProgressCb = Callable[[int, int, TagStats], None]


def tag_episode(
    episode: Episode,
    *,
    client: anthropic.Anthropic | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    max_retries_per_batch: int = DEFAULT_MAX_RETRIES_PER_BATCH,
    progress_callback: ProgressCb | None = None,
) -> tuple[Episode, TagStats]:
    """Tag the episode's segments in place, populating difficulty + annotations.

    Also deduplicates expressions per type into the episode-level `*_summary` lists.
    Returns the same Episode object (mutated) and a `TagStats` diagnostic record.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file "
            "(see .env.example) before running tag.py."
        )

    if client is None:
        # SDK already retries 429/5xx with exponential backoff. We add a second retry
        # layer above that for content-level failures (Pydantic validation, etc).
        client = anthropic.Anthropic(max_retries=3)

    stats = TagStats()
    segments = episode.segments
    total = len(segments)
    if total == 0:
        return episode, stats

    for start in range(0, total, batch_size):
        if stats.estimated_cost_usd >= max_cost_usd:
            stats.cost_limit_hit = True
            break

        batch = segments[start : start + batch_size]
        batch_result, in_toks, out_toks = _tag_batch_with_retry(
            client, batch, max_retries_per_batch
        )
        stats.total_input_tokens += in_toks
        stats.total_output_tokens += out_toks
        stats.update_cost()

        if batch_result is None:
            stats.failed_segments += len(batch)
            for seg in batch:
                seg.annotations = []
            if progress_callback:
                progress_callback(start + len(batch), total, stats)
            continue

        for seg in batch:
            llm_seg = batch_result.get(seg.id)
            if llm_seg is None:
                # Model omitted this segment from its response; treat as a soft failure
                # (no annotations) but still count it as a tagging attempt.
                stats.failed_segments += 1
                seg.annotations = []
                continue
            applied, dropped = _apply_annotations(seg, llm_seg)
            stats.tagged_segments += 1
            stats.dropped_annotations += dropped
            stats.annotation_counts.update(a.type for a in applied)
            seg.annotations = applied
            seg.difficulty = max(1, min(5, llm_seg.difficulty))

        if progress_callback:
            progress_callback(start + len(batch), total, stats)

    _populate_summaries(episode)
    return episode, stats


# --- Batch execution -------------------------------------------------------


def _tag_batch_with_retry(
    client: anthropic.Anthropic,
    segments: list[Segment],
    max_attempts: int,
) -> tuple[dict[str, _LLMSegment] | None, int, int]:
    """Run one batch with retries. Returns (results_by_seg_id, input_tokens, output_tokens).

    `results_by_seg_id` is None if every attempt failed — the caller defaults the batch
    to empty annotations.
    """
    last_error: Exception | None = None
    input_toks = 0
    output_toks = 0
    for attempt in range(max_attempts):
        try:
            parsed, in_t, out_t = _tag_batch(client, segments)
            input_toks += in_t
            output_toks += out_t
            return {s.segment_id: s for s in parsed.segments}, input_toks, output_toks
        except (anthropic.APIStatusError, ValidationError, ValueError) as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(2**attempt)  # 1s, 2s — SDK already backs off on 429/5xx
            else:
                # Token usage on a failed call is lost — Anthropic doesn't return usage
                # on a non-2xx, and on validation failure the parsed_output is missing.
                # That's OK; we just won't attribute the wasted call to the totals.
                pass
    # Failed all attempts
    print(f"  warning: batch failed after {max_attempts} attempts: {last_error}")
    return None, input_toks, output_toks


def _tag_batch(
    client: anthropic.Anthropic, segments: list[Segment]
) -> tuple[_LLMBatch, int, int]:
    user_message = _build_user_message(segments)
    response = client.messages.parse(
        model=MODEL_ID,
        max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        output_format=_LLMBatch,
    )
    parsed = response.parsed_output
    if parsed is None:
        # parse() returns None on hard schema failures; treat as ValueError for retry.
        raise ValueError("messages.parse() returned no parsed_output")
    return parsed, response.usage.input_tokens, response.usage.output_tokens


def _build_user_message(segments: list[Segment]) -> str:
    lines = [
        f"Tag these {len(segments)} segments from a Practical AI podcast. "
        "Return one entry per segment_id in your response."
    ]
    for seg in segments:
        lines.append("")
        lines.append(f"[{seg.id}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


# --- Annotation -> Segment.words mapping -----------------------------------


def _apply_annotations(
    segment: Segment, llm_segment: _LLMSegment
) -> tuple[list[Annotation], int]:
    """Convert LLM annotations to Annotation objects with word_idx ranges.

    Returns ``(applied, dropped)``. An annotation is dropped — not stored — when its
    `expression` can't be located in `segment.words` by the same normalization scheme
    match.py uses. Wrong indices would point the iOS app at the wrong word; absent
    annotations are safe.
    """
    applied: list[Annotation] = []
    dropped = 0
    word_norms = [_normalize_token(w.w) for w in segment.words]

    for ann in llm_segment.annotations:
        indices = _find_word_indices(ann.expression, word_norms)
        if indices is None:
            dropped += 1
            continue
        applied.append(
            Annotation(
                type=ann.type,
                expression=ann.expression,
                ko=ann.ko,
                word_idx=indices,
            )
        )
    return applied, dropped


def _find_word_indices(
    expression: str, word_norms: list[str]
) -> tuple[int, int] | None:
    """Sliding-window match expression tokens against pre-normalized Segment.words.

    Returns (start, end) inclusive into the words list, or None if not found.
    """
    expr_tokens = [_normalize_token(t) for t in _split_tokens(expression)]
    expr_tokens = [t for t in expr_tokens if t]
    n = len(expr_tokens)
    if n == 0:
        return None
    last_start = len(word_norms) - n
    for i in range(last_start + 1):
        if word_norms[i : i + n] == expr_tokens:
            return (i, i + n - 1)
    return None


# --- Episode summaries -----------------------------------------------------


def _populate_summaries(episode: Episode) -> None:
    """Dedup expressions per type, lowercase-keyed; assign sorted lists to summaries."""
    buckets: dict[str, dict[str, str]] = {
        "vocab": {},
        "turn_taking": {},
        "domain": {},
    }
    for seg in episode.segments:
        for ann in seg.annotations:
            key = ann.expression.strip().lower()
            if not key:
                continue
            buckets[ann.type].setdefault(key, ann.expression)

    episode.vocab_summary = sorted(buckets["vocab"].values(), key=str.lower)
    episode.turn_taking_summary = sorted(buckets["turn_taking"].values(), key=str.lower)
    episode.domain_summary = sorted(buckets["domain"].values(), key=str.lower)
