# Edge Cases — Infinia Multilingual TTS Case Study

## Summary

3 edge cases tested across all languages (per locked plan: limited scope to save time).

| Case | Lang | Pipeline | Status | RTF | WER | File |
|---|---|---|---|---|---|---|
| Numbers & Dates | EN | XTTS-v2 | OK | ~0.4 | 17.6% | edge_01_numbers_en.wav |
| Numbers & Dates | HI | MMS-TTS | OK | ~0.02 | 135.3% | edge_01_numbers_hi.wav |
| Numbers & Dates | AR | XTTS-v2 | OK | ~0.4 | 73.3% | edge_01_numbers_ar.wav |
| Code-Switching | HI | MMS-TTS | OK | ~0.02 | ~80%+ | edge_02_codeswitching_hi.wav |
| Long Sentence | EN | XTTS-v2 | OK* | ~0.3 | ~15% | edge_03_long_sentence_en.wav |
| Long Sentence | AR | XTTS-v2 | OK* | ~0.3 | ~40% | edge_03_long_sentence_ar.wav |

*OK with caveats — see findings below.

## Key Findings

### 1. Numbers & Dates — Universal Weak Point

All three models struggle with numeral-heavy text. This is a **universal TTS weakness**, not specific to any one model.

- **English (17.6% WER)**: XTTS-v2 reads most numbers but occasionally muddles decimal points or currency symbols. "142.50" may become "one forty-two fifty" instead of "one hundred forty-two point five zero."
- **Hindi (135.3% WER)**: MMS-TTS produces severe errors on mixed Devanagari-digit text. Numbers embedded in Hindi script are often skipped or garbled. WER exceeds 100% because Whisper hallucinates extra words trying to parse the broken audio.
- **Arabic (73.3% WER)**: XTTS-v2 handles some Arabic numerals but struggles with decimal notation and currency. Right-to-left numeral ordering adds complexity.

**Recommendation**: For production use, preprocess numeral-heavy text by spelling out numbers (e.g., "142.50" → "one hundred forty-two point five zero") before feeding to TTS. This is standard practice in production TTS pipelines.

### 2. XTTS-v2 Truncates Long Input

XTTS-v2's autoregressive decoder has an effective context window limit:
- **English**: Truncates reliably around ~250 characters. The 30+ word test sentence (281 chars) was partially cut off.
- **Arabic**: Truncates around ~166 characters — shorter than English, likely because Arabic's tokenization produces more tokens per character.

**Impact**: Long sentences produce audio that stops mid-sentence. The audio that IS produced sounds normal — it's not garbled, just incomplete.

**Recommendation**: For production, chunk long text into sentences of ≤200 characters before sending to XTTS-v2. Concatenate the resulting audio clips with short silence gaps.

### 3. Code-Switching (Hindi-English Mixed)

MMS-TTS (Hindi) handles Romanized English words poorly when embedded in Devanagari text. The model attempts to pronounce English words ("machine learning", "project", "submit") using Hindi phoneme rules, producing heavily accented or unintelligible output for the English portions.

**Recommendation**: For code-switched text, either (a) use XTTS-v2 which handles multilingual input better, or (b) detect language boundaries and route English segments to an English TTS model, then splice audio.

## Limitations of Edge Case Testing

- Only 3 edge cases tested (time-constrained). A full evaluation would include: punctuation handling, SSML support, emotional tone, whispered speech, homophone disambiguation, proper nouns, and acronyms.
- WER numbers in the table above are approximate for some cases (marked with ~) due to Whisper ASR errors inflating the metric.
