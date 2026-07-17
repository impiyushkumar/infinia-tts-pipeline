"""
run_edge_cases.py — Edge Case Tests (3 total, per locked plan)
Infinia Multilingual TTS Case Study

3 edge cases across all languages:
  1. Numbers/dates — tests if TTS reads numerals correctly
  2. Code-switching — Hindi-English mixed sentence
  3. Long sentence — 30+ word stress test

Runs each through the working pipelines, logs results + failures to stdout.
Copy output into edge_cases.md.

Usage (Colab):
    %run scripts/run_edge_cases.py
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_harness import roundtrip_wer

# ============================================================
# CONFIG
# ============================================================
OUTPUT_BASE = "/content/drive/MyDrive/infinia-tts-case-study/clips"
REF_DIR = "/content/drive/MyDrive/infinia-tts-case-study/reference_clips"

# Find reference audio for XTTS-v2 cloning
ref_candidates = [
    os.path.join(REF_DIR, f) for f in os.listdir(REF_DIR)
    if f.endswith((".wav", ".mp3", ".flac"))
] if os.path.exists(REF_DIR) else []

REFERENCE = ref_candidates[0] if ref_candidates else None

# ============================================================
# EDGE CASES
# ============================================================
EDGE_CASES = [
    {
        "id": "edge_01_numbers",
        "name": "Numbers & Dates",
        "description": "Tests numeral reading, mixed digits, dates, currency",
        "tests": [
            {"lang": "en", "pipeline": "xtts", "text": "On January 15th, 2025, the stock price rose from $142.50 to $187.93, a gain of 31.8 percent."},
            {"lang": "hi", "pipeline": "mms",  "text": "15 जनवरी 2025 को शेयर की कीमत 142.50 रुपये से बढ़कर 187.93 रुपये हो गई, यानी 31.8 प्रतिशत की वृद्धि।"},
            {"lang": "ar", "pipeline": "xtts", "text": "في 15 يناير 2025، ارتفع سعر السهم من 142.50 دولار إلى 187.93 دولار، بزيادة قدرها 31.8 بالمائة."},
        ],
    },
    {
        "id": "edge_02_codeswitching",
        "name": "Code-Switching (Hindi-English)",
        "description": "Mixed Hindi-English sentence — common in Indian speech. Tests if TTS handles script/language boundaries.",
        "tests": [
            {"lang": "hi", "pipeline": "mms", "text": "मैंने अपना machine learning project submit कर दिया है और अब results का wait कर रहा हूं।"},
        ],
    },
    {
        "id": "edge_03_long_sentence",
        "name": "Long Sentence (30+ words)",
        "description": "Stress test for attention/coherence with very long input.",
        "tests": [
            {"lang": "en", "pipeline": "xtts", "text": "The comprehensive development of artificial intelligence systems across multiple industries including healthcare, finance, transportation, and education has fundamentally transformed the way modern societies approach complex problem solving, resource allocation, and decision making processes in the twenty-first century."},
            {"lang": "ar", "pipeline": "xtts", "text": "إن التطور الشامل لأنظمة الذكاء الاصطناعي عبر صناعات متعددة بما في ذلك الرعاية الصحية والتمويل والنقل والتعليم قد أدى إلى تحول جذري في الطريقة التي تتعامل بها المجتمعات الحديثة مع حل المشكلات المعقدة."},
        ],
    },
]


# ============================================================
# PIPELINE LOADERS (lazy, cached)
# ============================================================
_xtts_model = None
_mms_hi_model = None

def get_xtts():
    global _xtts_model
    if _xtts_model is None:
        from TTS.api import TTS
        print("[...] Loading XTTS-v2 (cached from earlier runs)...", flush=True)
        _xtts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        print("[OK]  XTTS-v2 loaded")
    return _xtts_model

def get_mms_hi():
    global _mms_hi_model
    if _mms_hi_model is None:
        from transformers import VitsModel, VitsTokenizer
        print("[...] Loading MMS-TTS Hindi...", flush=True)
        tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-hin")
        model = VitsModel.from_pretrained("facebook/mms-tts-hin").to("cuda")
        _mms_hi_model = (model, tokenizer)
        print("[OK]  MMS-TTS Hindi loaded")
    return _mms_hi_model


def generate_clip(text, lang, pipeline, output_path):
    """Generate a single clip using the appropriate pipeline."""
    import soundfile as sf

    t0 = time.time()

    if pipeline == "xtts":
        tts = get_xtts()
        tts.tts_to_file(
            text=text,
            speaker_wav=REFERENCE,
            language=lang,
            file_path=output_path,
        )
    elif pipeline == "mms":
        model, tokenizer = get_mms_hi()
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(**inputs)
        audio = output.waveform.squeeze().cpu().numpy()
        sf.write(output_path, audio, model.config.sampling_rate)

    gen_time = time.time() - t0

    # Get duration
    audio_data, sr = sf.read(output_path)
    duration = len(audio_data) / sr

    return gen_time, duration


# ============================================================
# MAIN — run all edge cases
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EDGE CASE TESTS — 3 cases across all languages")
    print("=" * 60)

    if not REFERENCE:
        print("[ERROR] No reference audio found — XTTS-v2 tests will fail.")

    results = []

    for case in EDGE_CASES:
        print(f"\n{'='*60}")
        print(f"EDGE CASE: {case['name']}")
        print(f"  {case['description']}")
        print(f"{'='*60}")

        for test in case["tests"]:
            lang = test["lang"]
            pipeline = test["pipeline"]
            text = test["text"]

            # Output path
            lang_dir = {"en": "english", "hi": "hindi", "ar": "arabic"}[lang]
            out_dir = os.path.join(OUTPUT_BASE, lang_dir)
            os.makedirs(out_dir, exist_ok=True)
            filename = f"{case['id']}_{lang}.wav"
            output_path = os.path.join(out_dir, filename)

            print(f"\n  [{lang.upper()}] ({pipeline}) \"{text[:60]}{'...' if len(text) > 60 else ''}\"")

            try:
                gen_time, duration = generate_clip(text, lang, pipeline, output_path)
                rtf = gen_time / duration if duration > 0 else float('inf')
                print(f"  [OK]  Generated in {gen_time:.2f}s → {duration:.2f}s audio (RTF: {rtf:.3f})")
                print(f"  [OK]  Saved: {filename}")

                # Run WER
                try:
                    wer = roundtrip_wer(output_path, text, language=lang)
                    print(f"  [WER] {wer['wer']*100:.1f}% — Whisper heard: \"{wer['transcript'][:80]}\"")
                    wer_val = wer['wer']
                    transcript = wer['transcript']
                except Exception as e:
                    print(f"  [WER] FAILED: {e}")
                    wer_val = None
                    transcript = "FAILED"

                results.append({
                    "case": case['name'],
                    "lang": lang.upper(),
                    "pipeline": pipeline,
                    "status": "OK",
                    "gen_time": gen_time,
                    "duration": duration,
                    "rtf": rtf,
                    "wer": wer_val,
                    "transcript": transcript,
                    "text": text,
                    "filename": filename,
                })

            except Exception as e:
                print(f"  [FAIL] {e}")
                results.append({
                    "case": case['name'],
                    "lang": lang.upper(),
                    "pipeline": pipeline,
                    "status": f"FAILED: {e}",
                    "text": text,
                    "filename": filename,
                })

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("EDGE CASE SUMMARY")
    print("=" * 60)
    print(f"  {'Case':<25} {'Lang':<6} {'Status':<8} {'RTF':<8} {'WER':<8} {'File'}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    for r in results:
        status = "OK" if r["status"] == "OK" else "FAIL"
        rtf_str = f"{r.get('rtf', 0):.3f}" if r["status"] == "OK" else "—"
        wer_str = f"{r['wer']*100:.1f}%" if r.get("wer") is not None else "—"
        print(f"  {r['case']:<25} {r['lang']:<6} {status:<8} {rtf_str:<8} {wer_str:<8} {r['filename']}")

    print(f"\n  Total: {len(results)} tests, "
          f"{sum(1 for r in results if r['status'] == 'OK')} passed, "
          f"{sum(1 for r in results if r['status'] != 'OK')} failed")
    print("=" * 60)
    print("\nCopy the summary above into edge_cases.md for the final write-up.")

    # Clean up GPU
    del _xtts_model, _mms_hi_model
    torch.cuda.empty_cache()
    print("[OK] GPU memory freed")
