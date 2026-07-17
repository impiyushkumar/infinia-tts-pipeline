# Infinia Multilingual TTS Case Study

**Track B — Audio deliverable**

Benchmarking multilingual TTS (English, Arabic, Hindi) on Colab T4 with real metrics.

## Quick Start (Colab)

1. Open a new Colab notebook, set runtime to **GPU → T4**
2. Clone this repo and upload a reference voice clip (~10-20s WAV) to `reference_clips/`
3. Run scripts in order:

```python
# 1. Setup environment
%run scripts/setup_colab.py

# 2. Generate clips (each produces warmup + benchmark clip)
%run scripts/pipeline_english.py
%run scripts/pipeline_hindi.py
%run scripts/pipeline_arabic.py

# 3. Run evals (WER, speaker similarity, metrics summary)
%run scripts/run_eval_english.py
%run scripts/run_eval_hindi.py
%run scripts/run_eval_arabic.py

# 4. Edge cases (3 total: numbers, code-switching, long text)
%run scripts/run_edge_cases.py
```

## Results Summary

| Metric | English (XTTS-v2) | Hindi (MMS-TTS) | Arabic (XTTS-v2) |
|---|---|---|---|
| RTF (target ≤0.5) | 0.439 ✓ | 0.018 ✓ | 0.413 ✓ |
| WER (target ≤10%) | 0.0% ✓ | 35.0% ✗ | 26.7% ✗ |
| Speaker Sim (target ≥0.75) | 0.9735 ✓ | N/A (no cloning) | 0.9373 ✓ |
| MOS (target ≥4.0) | 4/5 ✓ | 3/5 ✗ | 3/5 ✗ |

## Models Used

| Language | Model | Voice Cloning |
|----------|-------|---|
| English | XTTS-v2 (coqui-tts community fork) | Yes — reference audio |
| Hindi | Meta MMS-TTS (`facebook/mms-tts-hin`) | No — fixed speaker |
| Arabic | XTTS-v2 (coqui-tts community fork) | Yes — reference audio |

> **Dropped models:** Chatterbox (build failure, Python 3.12), Indic Parler-TTS (HuggingFace gating + version conflicts). Documented in writeup.md.

## Hardware

- Google Colab T4 GPU (14.6 GB VRAM)
- PyTorch 2.11.0+cu128, CUDA 12.8, Python 3.12

## Deliverable Structure

```
infinia-tts-case-study/
├── README.md                  # this file
├── writeup.md                 # summary, recommendations, failure modes, what's missing
├── results_table.md           # all metrics vs targets
├── edge_cases.md              # 3 edge case findings
├── notes.md                   # clip → model/config/metrics mapping
├── mos_ratings.csv            # MOS scores (single human rater)
├── clips/                     # generated audio (on Google Drive)
│   ├── english/               # en_test_01_warmup.wav, en_test_02_warm.wav, edge cases
│   ├── hindi/                 # hi_test_01_warmup.wav, hi_test_02_warm.wav, edge cases
│   └── arabic/                # ar_test_01_warmup.wav, ar_test_02_warm.wav, edge cases
├── reference_clips/           # voice reference for XTTS-v2 cloning
└── scripts/
    ├── setup_colab.py         # environment setup (dependency pinning, health checks)
    ├── pipeline_english.py    # English TTS generation (XTTS-v2)
    ├── pipeline_hindi.py      # Hindi TTS generation (MMS-TTS)
    ├── pipeline_arabic.py     # Arabic TTS generation (XTTS-v2)
    ├── eval_harness.py        # reusable eval functions (RTF, WER, speaker sim)
    ├── run_eval_english.py    # run eval against English clips
    ├── run_eval_hindi.py      # run eval against Hindi clips
    ├── run_eval_arabic.py     # run eval against Arabic clips
    └── run_edge_cases.py      # 3 edge case tests across all languages
```

## Status

- [x] Step 1: Colab setup + dependency pinning
- [x] Step 2: English pipeline (XTTS-v2)
- [x] Step 3: Eval harness (RTF, WER, MFCC speaker similarity)
- [x] Step 4: English eval — RTF 0.439, WER 0%, sim 0.9735
- [x] Step 5: Hindi pipeline (MMS-TTS)
- [x] Step 6: Hindi eval — RTF 0.018, WER 35%
- [x] Step 7: Arabic pipeline (XTTS-v2)
- [x] Step 8: Arabic eval — RTF 0.413, WER 26.7%, sim 0.9373
- [x] Step 9: Edge cases (numbers, code-switching, long text)
- [x] Step 10: MOS ratings (single rater)
- [x] Step 11: Results table, notes.md, write-up
