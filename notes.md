# Notes — Clip-to-Model Mapping
# Infinia Multilingual TTS Case Study

## Hardware
- **GPU**: Tesla T4 (14.6 GB VRAM)
- **Platform**: Google Colab
- **PyTorch**: 2.11.0+cu128 (CUDA 12.8)
- **Python**: 3.12

## Models Used

| Model | HuggingFace / Package | Languages | Voice Cloning |
|---|---|---|---|
| XTTS-v2 | `coqui-tts` (community fork) — `tts_models/multilingual/multi-dataset/xtts_v2` | English, Arabic | Yes (reference audio) |
| MMS-TTS Hindi | `facebook/mms-tts-hin` via `transformers` VitsModel | Hindi | No (fixed speaker) |

## Reference Audio

- **Source**: LJSpeech dataset (single female English speaker, public domain)
- **Usage**: Voice cloning reference for XTTS-v2 (English + Arabic)
- **Location**: `reference_clips/`
- **Note**: MMS-TTS Hindi does not use reference audio — confirmed experimentally that output is identical regardless of reference clip presence.

### Model Selection Rationale

- **Chatterbox** (original English primary): Dropped — persistent build failures on Colab Python 3.12.
- **Indic Parler-TTS** (original Hindi primary): Dropped — HuggingFace gating (manual approval required) + transformers version conflicts with coqui-tts.
- **XTTS-v2**: Proven working for English, reused for Arabic (same model, `language="ar"`). Voice cloning via reference audio.
- **MMS-TTS**: Lightweight VITS model, publicly accessible, no gating. No voice cloning.

## Clip Inventory

### English Clips (`clips/english/`)

| Filename | Model | Config | Metrics | Notes |
|---|---|---|---|---|
| `en_test_01_warmup.wav` | XTTS-v2 | language="en", cloned from LJSpeech ref | RTF ~1.5 (cold) | First generation, includes model warmup |
| `en_test_02_warm.wav` | XTTS-v2 | language="en", cloned from LJSpeech ref | RTF 0.390, WER 0%, sim 0.9911, MOS 4 | **Benchmark clip** — warm inference |
| `edge_01_numbers_en.wav` | XTTS-v2 | language="en", cloned from LJSpeech ref | WER 17.6% | Edge case: numbers/dates |
| `edge_03_long_sentence_en.wav` | XTTS-v2 | language="en", cloned from LJSpeech ref | Text truncated ~250 chars | Edge case: 30+ words |

### Hindi Clips (`clips/hindi/`)

| Filename | Model | Config | Metrics | Notes |
|---|---|---|---|---|
| `hi_test_01_warmup.wav` | MMS-TTS | facebook/mms-tts-hin, fixed speaker | RTF ~0.024 (cold) | First generation |
| `hi_test_02_warm.wav` | MMS-TTS | facebook/mms-tts-hin, fixed speaker | RTF 0.018, WER 35–50%, MOS 3 | **Benchmark clip** — warm inference |
| `edge_01_numbers_hi.wav` | MMS-TTS | facebook/mms-tts-hin, fixed speaker | WER 135.3% | Edge case: numbers/dates (severe) |
| `edge_02_codeswitching_hi.wav` | MMS-TTS | facebook/mms-tts-hin, fixed speaker | WER ~80%+ | Edge case: Hindi-English mixed |

### Arabic Clips (`clips/arabic/`)

| Filename | Model | Config | Metrics | Notes |
|---|---|---|---|---|
| `ar_test_01_warmup.wav` | XTTS-v2 | language="ar", cloned from LJSpeech ref | RTF ~1.3 (cold) | First generation |
| `ar_test_02_warm.wav` | XTTS-v2 | language="ar", cloned from LJSpeech ref | RTF 0.390, WER 13.3%, sim 0.9896, MOS 3 | **Benchmark clip** — warm inference |
| `edge_01_numbers_ar.wav` | XTTS-v2 | language="ar", cloned from LJSpeech ref | WER 73.3% | Edge case: numbers/dates |
| `edge_03_long_sentence_ar.wav` | XTTS-v2 | language="ar", cloned from LJSpeech ref | Text truncated ~166 chars | Edge case: 30+ words |

### Reference Clips (`reference_clips/`)

| Filename | Source | Purpose |
|---|---|---|
| LJSpeech clip | LJSpeech dataset (public domain) | Voice reference for XTTS-v2 cloning (English + Arabic) |

## Eval Harness Dependencies

| Package | Version | Purpose |
|---|---|---|
| faster-whisper | (Colab default) | ASR for round-trip WER, `small` model, CUDA float16 |
| jiwer | latest | WER computation |
| librosa | latest | MFCC extraction for speaker similarity |
| soundfile | latest | Audio I/O |
| numpy | <2.0 (pinned) | Many TTS libs require numpy 1.x APIs |
| transformers | 4.44.2 (pinned) | MMS-TTS VitsModel; pinned for coqui-tts compatibility |

## Constraints & Pins

- `torch`: Colab's preinstalled version (never modified — previous force-reinstalls broke CUDA)
- `torchaudio`: Colab's preinstalled version (repaired only if broken by other installs)
- `transformers==4.44.2`: Required for coqui-tts compatibility (is_flax_available removed in 4.45+)
- `numpy<2.0`: Required for coqui-tts and broader TTS ecosystem (numpy.char API removed in 2.0)
