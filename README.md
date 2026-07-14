# Infinia Multilingual TTS Case Study

**Track B — Audio deliverable**

Benchmarking 3 open-source TTS pipelines (English, Arabic, Hindi) on Colab T4.

## Quick Start (Colab)

1. Open a new Colab notebook, set runtime to **GPU → T4**
2. Upload or clone this repo into Colab
3. Run scripts in order:

```python
# Cell 1: Setup environment
%run scripts/setup_colab.py

# Cell 2: English pipeline
%run scripts/pipeline_english.py

# Cell 3: Hindi pipeline
%run scripts/pipeline_hindi.py

# Cell 4: Arabic pipeline
%run scripts/pipeline_arabic.py

# Cell 5: Eval harness (run after generating clips)
%run scripts/eval_harness.py
```

## Models Used

| Language | Primary Model | Fallback Model |
|----------|--------------|----------------|
| English  | Chatterbox   | XTTS-v2 (Coqui) |
| Hindi    | AI4Bharat Indic Parler-TTS | Indic-TTS |
| Arabic   | XTTS-v2 (Coqui) | Meta MMS-TTS |

## Hardware

- Google Colab T4 GPU (16 GB VRAM)
- All benchmarks run on this hardware unless noted otherwise

## Deliverable Structure

```
infinia-tts-case-study/
├── README.md                # this file
├── notes.md                 # clip → model/config/metrics mapping (written from real data)
├── results_table.md         # all 3 languages vs targets
├── edge_cases.md            # real edge case test results
├── clips/
│   ├── english/
│   ├── hindi/
│   └── arabic/
├── reference_clips/         # voice reference clips for cloning
├── scripts/
│   ├── setup_colab.py       # Step 1: environment setup + smoke test
│   ├── pipeline_english.py  # Step 2: English TTS
│   ├── pipeline_hindi.py    # Step 5: Hindi TTS
│   ├── pipeline_arabic.py   # Step 6: Arabic TTS
│   └── eval_harness.py      # Step 3: reusable eval functions
└── mos_ratings.csv           # filled manually by human raters
```

## Status

- [x] Step 1: Colab setup + smoke test
- [ ] Step 2: English pipeline
- [ ] Step 3: Eval harness
- [ ] Step 4: English end-to-end run
- [ ] Step 5: Hindi pipeline
- [ ] Step 6: Arabic pipeline
- [ ] Step 7: MOS collection setup
- [ ] Step 8: Edge case tests
- [ ] Step 9: Results table
- [ ] Step 10: notes.md
- [ ] Step 11: Summary + failure modes
