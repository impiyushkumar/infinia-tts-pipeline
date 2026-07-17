# Results Table — Infinia Multilingual TTS Case Study

## Core Metrics (Warm Inference, Colab T4 GPU)

| Metric | Target | English (XTTS-v2) | Hindi (MMS-TTS) | Arabic (XTTS-v2) |
|---|---|---|---|---|
| **RTF** | ≤ 0.5 | 0.390 ✓ | 0.018 ✓ | 0.390 ✓ |
| **WER** | ≤ 10% | 0.0% ✓ | 35–50% ✗ | 13.3% ✗ |
| **Speaker Similarity** | ≥ 0.75 | 0.9911 ✓ | N/A (no cloning) | 0.9896 ✓ |
| **MOS** | ≥ 4.0 | 4 / 5 ✓ | 3 / 5 ✗ | 3 / 5 ✗ |

## Cold vs Warm Comparison

| Metric | English Cold | English Warm | Hindi Cold | Hindi Warm | Arabic Cold | Arabic Warm |
|---|---|---|---|---|---|---|
| Generation time (s) | ~9.5 | ~2.5 | ~0.08 | ~0.06 | ~8.3 | ~2.5 |
| RTF | ~1.5 | 0.390 | ~0.024 | 0.018 | ~1.3 | 0.390 |

## Measurement Methodology

| Metric | Method | Tool |
|---|---|---|
| RTF | `generation_time / audio_duration` | Measured via `time.time()` around inference call |
| WER | Round-trip: TTS → faster-whisper ASR → jiwer comparison | faster-whisper `small` model, CUDA, float16 |
| Speaker Similarity | MFCC mean vector cosine similarity (20 coefficients) | librosa + numpy |
| MOS | Single human rater, 1–5 scale | Manual listening (limitation noted) |

## Reference Audio

- **Source**: LJSpeech dataset (single female English speaker)
- **Usage**: Voice cloning reference for XTTS-v2 (English + Arabic). Not used by MMS-TTS Hindi (no cloning capability).

## Notes

- **RTF**: All three languages beat the ≤0.5 target. MMS-TTS is dramatically faster (0.018) because it's a lightweight VITS model vs XTTS-v2's autoregressive decoder.
- **WER**: English achieves 0% — XTTS-v2 produces highly intelligible English. Hindi WER showed run-to-run variance (35% in one run, 50% in another) — noted as a limitation of both the TTS model and the round-trip WER metric. Arabic at 13.3% is a close miss on the 10% target. Hindi and Arabic WER failures are consistent with the brief's prediction that current fast TTS models underperform on non-English languages.
- **WER Run-to-Run Variance**: Hindi WER varied between 35% and 50% across runs with identical input text. This is caused by non-deterministic inference in MMS-TTS and compounded by Whisper ASR errors on Hindi. This variance is itself a finding — production deployments need deterministic inference or multiple-sample averaging.
- **Speaker Similarity**: Hindi shows N/A because MMS-TTS has no voice cloning — it uses a fixed speaker voice. XTTS-v2 clones extremely well for both English (0.9911) and Arabic (0.9896) from the LJSpeech reference. Metric uses MFCC cosine similarity (simpler proxy than a dedicated speaker-embedding model; resemblyzer was dropped due to numpy 2.x incompatibility).
- **MOS**: Single rater (the author). English sounds natural and clear. Hindi and Arabic are intelligible but have noticeable prosody/accent artifacts. MOS of 3 ("fair") is honest — these are not production-ready for Hindi/Arabic.
