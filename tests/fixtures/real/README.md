# Real-episode fixtures

`tests/test_match.py::test_real_practical_ai_1_smoke` runs only when both files below
are present. To enable the test:

1. Run `python scripts/process_one.py 1` once. This drops a cached `whisper_raw.json`
   under `data/episodes/1/` (≈30–40 min on CPU).
2. Save the first ~5 minutes of that file as **`practical-ai-1.whisper.json`** here.
   The format is exactly what `src.align.words_to_json` produces — a JSON list of
   `{word, start, end, speaker}` objects. Trim by `end <= 300.0` or similar.
3. Save the matching slice of the GitHub transcript as
   **`practical-ai-1.transcript.md`** here. Use the raw Markdown (with `**Name:**`
   speaker tags); `src.transcript.parse_transcript` parses it back.

Once both files exist the test runs automatically — no flag or env var needed.
