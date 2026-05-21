"""GitHub transcript fetcher + parser for `thechangelog/transcripts`.

The Changelog stores official transcripts as Markdown, one file per episode at
``practicalai/practical-ai-<N>.md``. Each speaker turn opens with a ``**Name:**`` bold
marker; turns span as many paragraphs as the speaker holds the floor for. Time markers
(``\\[04:05\\]``) and stage cues (``\\[laughter\\]``) appear escape-bracketed and have
no spoken counterpart, so we drop them — same for inline Markdown formatting that
Whisper would never emit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

TRANSCRIPT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/thechangelog/transcripts/master/"
    "practicalai/practical-ai-{episode_id}.md"
)


class TranscriptNotPublishedError(LookupError):
    """Raised when GitHub returns 404 for the episode's transcript.

    Distinct from generic `LookupError` so callers (main.py) can schedule a
    soft retry — the Changelog usually publishes transcripts a few days after the
    audio drops, so a 404 today is often resolvable a few days later. Inherits
    from `LookupError` to stay backward-compatible with callers that catch
    `LookupError` for the older "permanent missing transcript" semantics.
    """

_SPEAKER_RE = re.compile(r"^\*\*([^*:]+?):\*\*\s*")
_ESCAPED_BRACKET_RE = re.compile(r"\\\[[^\]]*\\\]")
_PLAIN_TIME_MARKER_RE = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"\*([^*\n]+)\*")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_WHITESPACE_RE = re.compile(r"\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class TranscriptTurn:
    """One speaker's contiguous span, with paragraphs joined by blank lines."""

    speaker: str  # canonical (longest-form) name
    text: str


def transcript_url(episode_id: int) -> str:
    return TRANSCRIPT_URL_TEMPLATE.format(episode_id=episode_id)


def _clean_paragraph(text: str) -> str:
    text = _ESCAPED_BRACKET_RE.sub("", text)
    text = _PLAIN_TIME_MARKER_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _split_paragraphs(md: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(md.strip()) if p.strip()]


def _canonicalize_speakers(raw_names: set[str]) -> dict[str, str]:
    """Map every observed speaker label to a canonical name.

    Group by first-name prefix, then within each group:

    - Multi-token names (``"Chris Benson"``, ``"Chris Shallue"``) are treated as
      distinct people — different last names mean different humans. Each maps to
      itself.
    - A single-token entry (``"Chris"``) is treated as a short alias and resolved to
      the group's unique multi-token form. With multiple distinct multi-token forms
      in the group the short alias is genuinely ambiguous, so we raise.

    This handles the common case of a host ("Chris Benson") and a guest ("Chris
    Shallue") appearing in the same episode without forcing the parser to fail.
    """
    groups: dict[str, list[str]] = {}
    for name in raw_names:
        first = name.split()[0].lower()
        groups.setdefault(first, []).append(name)

    mapping: dict[str, str] = {}
    for first, names in groups.items():
        full_forms = [n for n in names if len(n.split()) >= 2]
        short_forms = [n for n in names if len(n.split()) == 1]

        # Multi-token forms are each their own identity.
        for n in full_forms:
            mapping[n] = n

        if short_forms:
            if len(full_forms) == 1:
                # Unambiguous short alias for the single full form.
                canonical = full_forms[0]
                for n in short_forms:
                    mapping[n] = canonical
            elif len(full_forms) == 0:
                # No full form anywhere; the short label is the whole identity.
                for n in short_forms:
                    mapping[n] = n
            else:
                raise ValueError(
                    f"Ambiguous short-form speaker label {short_forms!r} for first name "
                    f"{first!r}: cannot pick between multiple full forms {full_forms!r}. "
                    "Manual mapping required."
                )
    return mapping


def parse_transcript(md: str) -> list[TranscriptTurn]:
    """Parse a Markdown transcript into a list of speaker turns."""
    raw_turns: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None

    for para in _split_paragraphs(md):
        m = _SPEAKER_RE.match(para)
        if m:
            if current is not None:
                raw_turns.append(current)
            speaker = m.group(1).strip()
            content = para[m.end() :].strip()
            current = (speaker, [content] if content else [])
        else:
            if current is None:
                # Pre-conversation content (YAML frontmatter, episode intro blurbs, …)
                # — silently drop until the first speaker line.
                continue
            current[1].append(para)
    if current is not None:
        raw_turns.append(current)

    canonical = _canonicalize_speakers({sp for sp, _ in raw_turns})

    turns: list[TranscriptTurn] = []
    for raw_speaker, paras in raw_turns:
        cleaned_paras = [c for c in (_clean_paragraph(p) for p in paras) if c]
        if not cleaned_paras:
            continue
        turns.append(
            TranscriptTurn(
                speaker=canonical[raw_speaker],
                text="\n\n".join(cleaned_paras),
            )
        )
    return turns


def fetch_transcript(episode_id: int) -> list[TranscriptTurn]:
    """Download and parse the transcript for a given episode."""
    url = transcript_url(episode_id)
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        raise TranscriptNotPublishedError(
            f"No GitHub transcript for episode {episode_id} (404 at {url}) — "
            "the Changelog usually publishes transcripts a few days after the episode"
        )
    resp.raise_for_status()
    return parse_transcript(resp.text)
