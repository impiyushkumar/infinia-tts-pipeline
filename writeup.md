# Infinia Multilingual TTS Case Study — Final Write-Up

## Executive Summary

This case study evaluates multilingual text-to-speech across English, Hindi, and Arabic on a Colab T4 GPU, using two models: **XTTS-v2** (English + Arabic, with voice cloning) and **Meta MMS-TTS** (Hindi, fixed speaker). All three languages meet the RTF target (≤0.5), with English achieving near-perfect intelligibility (0% WER) and strong voice cloning fidelity (0.97 cosine similarity). Hindi and Arabic show elevated WER (35% and 26.7% respectively) — consistent with the brief's prediction that current fast TTS models underperform on non-English languages. This is the central finding: **speed is solved, but non-English quality is the real gap in the current open-source TTS landscape.**

## Per-Language Results & Recommendations

### English — XTTS-v2
**Verdict: Production-viable with chunking.**

XTTS-v2 delivers excellent English output: natural prosody, high intelligibility (0% WER), and strong voice cloning (0.97 similarity). RTF of 0.439 comfortably beats the 0.5 target. MOS of 4/5 reflects genuinely good audio quality.

**Limitation**: Truncates input beyond ~250 characters. Production deployments must chunk long text into sentences before synthesis.

**Recommendation**: Use XTTS-v2 for English TTS with voice cloning. Implement a text chunker (sentence-level splitting at ≤200 chars) and a number-to-words preprocessor for numeral-heavy content.

### Hindi — Meta MMS-TTS
**Verdict: Acceptable for prototyping, not production.**

MMS-TTS is extremely fast (RTF 0.018) and produces intelligible Hindi speech, but with noticeable robotic quality (MOS 3/5). The 35% WER is partly inflated by Whisper's Hindi ASR errors, but the underlying audio does have prosody issues — flat intonation, unnatural pauses. No voice cloning available (fixed speaker).

**Limitation**: Cannot handle code-switched Hindi-English text (common in Indian speech). Numbers in Devanagari are severely garbled (135% WER). No voice cloning.

**Recommendation**: For production Hindi TTS with cloning, evaluate fine-tuning XTTS-v2 on Hindi data or waiting for Indic Parler-TTS to become publicly accessible (currently gated). MMS-TTS is a reasonable baseline for low-latency applications where voice identity doesn't matter.

### Arabic — XTTS-v2
**Verdict: Functional but quality gap vs English.**

XTTS-v2 handles Modern Standard Arabic with reasonable intelligibility and good voice cloning (0.94 similarity), but WER of 26.7% reveals pronunciation errors on certain phonemes. MOS of 3/5 reflects perceptible accent/prosody issues. RTF (0.413) meets the speed target.

**Limitation**: Shorter context window for Arabic (~166 chars vs ~250 for English). Numbers and dates cause significant WER spikes (73.3%).

**Recommendation**: XTTS-v2 is the best available open-source option for Arabic TTS with cloning. Quality can be improved by fine-tuning on MSA data. For production, implement aggressive text preprocessing (number spelling, sentence chunking at ≤150 chars).

## Failure Modes Observed

### 1. Dependency Hell (Environment)
The Colab environment required extensive dependency management:
- `transformers` had to be pinned to 4.44.2 (coqui-tts uses internal APIs removed in 4.45+)
- `numpy` had to be pinned to <2.0 (numpy.char API removal broke multiple TTS packages)
- `torch`/`torchaudio` could not be force-reinstalled without breaking CUDA support
- **Chatterbox** failed to build entirely on Python 3.12
- **Indic Parler-TTS** was blocked by HuggingFace gating + version conflicts

**Takeaway**: The open-source TTS ecosystem is fragmented. Production deployments need locked Docker images, not pip-install-and-pray.

### 2. Model Failures
- **Chatterbox**: Build failure on Colab (Python 3.12 incompatibility). Dropped entirely.
- **Indic Parler-TTS**: HuggingFace gating requires manual approval. Dropped for MMS-TTS.
- **XTTS-v2 long text**: Silent truncation beyond ~250 chars (English) / ~166 chars (Arabic). No error raised — audio just stops.

### 3. Metric Limitations
- **Round-trip WER** inflates error rates for non-English languages because Whisper's ASR is less accurate for Hindi/Arabic. The 35% Hindi WER includes both TTS errors AND ASR errors — they cannot be separated without human transcription.
- **MFCC cosine similarity** is a simpler proxy than dedicated speaker-embedding models (resemblyzer was dropped due to numpy 2.x incompatibility). It captures timbral similarity but is less robust to prosody/content variations.
- **MOS**: Single rater (the author). A proper MOS study requires 15+ naive raters. Results should be interpreted as directional, not statistically significant.

## What's Missing (If More Time)

1. **More models**: Chatterbox (needs Python 3.11 env), Indic Parler-TTS (needs HF approval), Piper TTS (lightweight alternative), Bark (multilingual).
2. **Fine-tuning**: XTTS-v2 fine-tuned on Hindi/Arabic data would likely close the quality gap significantly.
3. **Streaming latency**: All measurements are batch (start-to-complete). XTTS-v2 supports streaming inference — measuring time-to-first-audio-chunk would give more realistic latency numbers for real-time applications.
4. **Proper MOS study**: 15+ naive raters, randomized A/B comparisons, statistical significance testing.
5. **More edge cases**: SSML support, emotional tone, whispered speech, proper nouns, acronyms, punctuation effects, multi-speaker conversations.
6. **Speaker similarity**: A proper speaker-embedding model (e.g., ECAPA-TDNN or resemblyzer with numpy 1.x) would give more meaningful similarity scores than MFCC cosine.
7. **Diacritized Arabic**: Testing with fully diacritized Arabic text (tashkeel) would isolate whether WER issues come from pronunciation ambiguity or model quality.

## Reproducibility

All code, generated clips, and reference audio are in the repository. To reproduce:

```bash
# 1. Open in Google Colab with T4 GPU
# 2. Clone the repo and run setup
%run scripts/setup_colab.py

# 3. Upload reference audio to reference_clips/

# 4. Run pipelines
%run scripts/pipeline_english.py
%run scripts/pipeline_hindi.py
%run scripts/pipeline_arabic.py

# 5. Run evals
%run scripts/run_eval_english.py
%run scripts/run_eval_hindi.py
%run scripts/run_eval_arabic.py

# 6. Run edge cases
%run scripts/run_edge_cases.py
```

Hardware: Google Colab T4 GPU (14.6 GB VRAM), Python 3.12, PyTorch 2.11.0+cu128.
