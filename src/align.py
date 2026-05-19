"""WhisperX wrapper: transcribe → word-level align → speaker diarize.

Returns a flat list of `AlignedWord` records — each is one word with its start/end times
and an anonymous diarization label (`SPEAKER_00`, `SPEAKER_01`, …). Mapping those labels
to real speaker names happens in `match.py`.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AlignedWord:
    word: str
    start: float
    end: float
    speaker: str | None  # SPEAKER_XX from diarization, or None if unassigned


def transcribe_and_align(
    audio_path: str,
    *,
    model_name: str = "base.en",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 16,
    hf_token: str | None = None,
) -> list[AlignedWord]:
    """Run the full Whisper → align → diarize chain on one audio file.

    `whisperx` is imported lazily so that match.py's unit tests can import the package
    without dragging in torch + pyannote.
    """
    import whisperx

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for diarization. Set it in .env, and make sure the "
            "same Hugging Face account has accepted the licenses at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 and "
            "https://huggingface.co/pyannote/segmentation-3.0."
        )

    audio = whisperx.load_audio(audio_path)

    asr_model = whisperx.load_model(model_name, device, compute_type=compute_type)
    transcribed = asr_model.transcribe(audio, batch_size=batch_size)
    language = transcribed.get("language", "en")

    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    aligned = whisperx.align(
        transcribed["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Newer whisperx (≥4.x) moved this class to the submodule and renamed the auth
    # kwarg; the default model also flipped to `community-1`, which carries a separate
    # license. We pin 3.1 explicitly because that's the model the README tells users to
    # accept on Hugging Face.
    from whisperx.diarize import DiarizationPipeline

    diarize = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        token=token,
        device=device,
    )
    diarize_segments = diarize(audio)
    with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned)

    return _flatten_words(with_speakers)


def _flatten_words(result: dict) -> list[AlignedWord]:
    """Flatten WhisperX's ``{segments: [{words: [...]}]}`` into a single word list.

    Drops words for which the aligner couldn't produce a timestamp (rare, usually next
    to silence) since we have no way to place them on the audio timeline.
    """
    out: list[AlignedWord] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if not text or start is None or end is None:
                continue
            out.append(
                AlignedWord(
                    word=text,
                    start=float(start),
                    end=float(end),
                    speaker=w.get("speaker"),
                )
            )
    return out


def words_to_json(words: list[AlignedWord]) -> list[dict]:
    """Serialize for caching to disk — alignment is the pipeline's slowest step."""
    return [dataclasses.asdict(w) for w in words]


def words_from_json(data: list[dict]) -> list[AlignedWord]:
    return [AlignedWord(**d) for d in data]
