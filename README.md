# practical-ai-prep

Preprocessing pipeline that turns each [Practical AI podcast](https://changelog.com/practicalai)
episode into a JSON record optimized for English-study consumption.

End goal is an iOS study app; this repo is the data layer only.

## What it does (per episode)

1. Detect new episodes from the RSS feed.
2. Download the MP3.
3. Pull the official transcript from
   [`thechangelog/transcripts`](https://github.com/thechangelog/transcripts) (accurate text,
   real speaker names, no timing).
4. Run WhisperX for word-level timestamps + speaker diarization (accurate timing,
   approximate text, anonymous `SPEAKER_XX`).
5. Match the two: GitHub text gets Whisper's word-level timing; `SPEAKER_XX` gets mapped
   to real names by majority-vote overlap. Audio regions with no GitHub counterpart
   (intros, sponsor reads, outros) go into `unaligned_regions`.
6. (Phase 2) Tag difficulty + vocab / turn-taking / domain annotations via Claude.
7. Commit the resulting JSON back to the repo.

## Repo layout

```
practical-ai-prep/
├── .github/workflows/process-episodes.yml   # daily cron (Phase 2)
├── src/
│   ├── main.py                              # orchestrator (Phase 2)
│   ├── rss.py                               # feed parser
│   ├── transcript.py                        # GitHub transcript fetch/parse
│   ├── align.py                             # WhisperX wrapper
│   ├── match.py                             # ⭐ banded-NW alignment between GH and Whisper
│   ├── tag.py                               # Claude annotations (Phase 2)
│   ├── schema.py                            # Pydantic models
│   └── storage.py                           # manifest + episode JSON I/O (Phase 2)
├── data/
│   ├── manifest.json                        # processed-episode index (Phase 2)
│   └── episodes/<episode_id>/episode.json
├── tests/
│   ├── fixtures/synthetic/                  # hand-built match cases
│   ├── fixtures/real/                       # one real episode excerpt
│   └── test_match.py
├── scripts/process_one.py                   # CLI for local one-shot runs
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Setup

### System dependencies

- Python 3.11 (3.12+ has flaky torch/whisperx wheels at the moment)
- `ffmpeg` for audio decoding:

  ```sh
  brew install ffmpeg
  ```

### Python environment

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Hugging Face: license acceptance for diarization

Diarization uses `pyannote/speaker-diarization-3.1`, which is license-gated. With the same
account that owns your `HF_TOKEN`, accept both:

- <https://huggingface.co/pyannote/speaker-diarization-3.1>
- <https://huggingface.co/pyannote/segmentation-3.0>

A valid token alone is **not** sufficient — without accepting the licenses you'll get a 401
on first run.

### `.env`

```sh
cp .env.example .env
# fill in ANTHROPIC_API_KEY and HF_TOKEN
```

## Running locally

Phase 1 ships a one-shot CLI that takes an explicit episode ID:

```sh
python scripts/process_one.py <episode_id>
# example (recommended first test):
python scripts/process_one.py 1
```

Output lands at `data/episodes/<episode_id>/episode.json`.

### Performance note

The pipeline is currently CPU-only (`base.en` + `int8`). On Apple Silicon expect
≈ real-time processing (a 40-minute episode takes ~40 minutes). GPU support is out of
scope for Phase 1.

## Testing

```sh
pytest
```

`tests/test_match.py` is the primary suite — match correctness drives study-data quality.
Synthetic fixtures cover word drops, hallucinated words, speaker change inside a turn,
number-format mismatches (e.g. `GPT-4` vs `gpt four`), and ad insertion. A separate
integration test runs the matcher on the first ~5 minutes of episode 1.

## Phase 2: Tagging with Claude

`tag.py` enriches an already-processed episode JSON with study annotations: per-segment
difficulty, plus vocab / turn-taking / domain expressions with Korean glosses. The model
is Claude Haiku 4.5 (`claude-haiku-4-5-20251001`); structured outputs guarantee a parseable
response, and `word_idx` ranges are computed locally by scanning each `Segment.words` for
the expression text (so wrong indices never reach the saved JSON).

### Run

```sh
# Dry-run first to tune the prompt — only the first 3 segments, no save
python scripts/tag_one.py 1 --dry-run

# Once happy, tag the whole episode
python scripts/tag_one.py 1
```

Output overwrites `data/episodes/<id>/practical-ai-<id>.json` with the previous version
preserved at `practical-ai-<id>.json.bak`.

### Cost

Rough expectation per episode at default `--batch-size 8`:

| Episode length | Segments | Tokens (in + out) | Cost   |
| -------------- | -------- | ----------------- | ------ |
| 30 min         | ~60      | ~30K              | ~$0.05 |
| 60 min         | ~150     | ~75K              | ~$0.15 |
| 90 min         | ~250     | ~125K             | ~$0.25 |

`--max-cost` (default `$2.00`) is a per-run ceiling — the script aborts and saves whatever
it has if cumulative spend crosses it.

### Requirements

`ANTHROPIC_API_KEY` in `.env` (see `.env.example`). Haiku 4.5 doesn't require any beta
headers or special license acceptance — unlike Phase 1's pyannote diarization.

## Scope split

Phase 1 (done):

- `schema.py`, `rss.py`, `transcript.py`, `align.py`, `match.py`
- Unit + integration tests for `match.py`
- `scripts/process_one.py` for local end-to-end runs

Phase 2 (in progress):

- `tag.py` + `scripts/tag_one.py` (Claude annotations) — done
- `manifest.json` bookkeeping in `storage.py`
- `main.py` orchestrator
- GitHub Actions daily run
