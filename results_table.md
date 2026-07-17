# Results Table — Infinia Multilingual TTS Case Study

## Core Metrics (Warm Inference, Colab T4 GPU)

| Metric | Target | English (XTTS-v2) | Hindi (MMS-TTS) | Arabic (XTTS-v2) |
|---|---|---|---|---|
| **RTF** | ≤ 0.5 | 0.439 ✓ | 0.018 ✓ | 0.413 ✓ |
| **WER** | ≤ 10% | 0.0% ✓ | 35.0% ✗ | 26.7% ✗ |
| **Speaker Similarity** | ≥ 0.75 | 0.9735 ✓ | N/A (no cloning) | 0.9373 ✓ |
| **MOS** | ≥ 4.0 | 4 / 5 ✓ | 3 / 5 ✗ | 3 / 5 ✗ |

## Cold vs Warm Comparison

| Metric | English Cold | English Warm | Hindi Cold | Hindi Warm | Arabic Cold | Arabic Warm |
|---|---|---|---|---|---|---|
| Generation time (s) | ~9.5 | ~2.8 | ~0.08 | ~0.06 | ~8.3 | ~2.6 |
| RTF | ~1.465 | 0.439 | ~0.024 | 0.018 | ~1.3 | 0.413 |

## Measurement Methodology

| Metric | Method | Tool |
|---|---|---|
| RTF | `generation_time / audio_duration` | Measured via `time.time()` around inference call |
| WER | Round-trip: TTS → faster-whisper ASR → jiwer comparison | faster-whisper `small` model, CUDA, float16 |
| Speaker Similarity | MFCC mean vector cosine similarity (20 coefficients) | librosa + numpy |
| MOS | Single human rater, 1–5 scale | Manual listening (limitation noted) |

## Notes

- **RTF**: All three languages beat the ≤0.5 target. MMS-TTS is dramatically faster (0.018) because it's a lightweight VITS model vs XTTS-v2's autoregressive decoder.
- **WER**: English achieves 0% — XTTS-v2 produces highly intelligible English. Hindi (35%) and Arabic (26.7%) WER failures are expected per the brief's warning that "fast/good models usually don't cover Arabic/Hindi well." WER is also inflated by Whisper ASR errors on non-English languages (known limitation of round-trip WER as a metric).
- **Speaker Similarity**: Hindi shows N/A because MMS-TTS has no voice cloning — it uses a fixed speaker voice. XTTS-v2 clones well for both English (0.9735) and Arabic (0.9373). Metric uses MFCC cosine similarity (simpler proxy than a dedicated speaker-embedding model like resemblyzer, which was dropped due to numpy 2.x incompatibility).
- **MOS**: Single rater (the author). English sounds natural and clear. Hindi and Arabic are intelligible but have noticeable prosody/accent artifacts. MOS of 3 ("fair") is honest — these are not production-ready for Hindi/Arabic.
