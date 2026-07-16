"""
run_eval_hindi.py — Run eval harness against Hindi pipeline output
Infinia Multilingual TTS Case Study

Mirrors run_eval_english.py structure exactly.

Usage (Colab):
    %run scripts/run_eval_hindi.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_harness import roundtrip_wer, speaker_cosine_similarity, measure_rtf, measure_latency

# ============================================================
# CONFIG
# ============================================================
CLIPS_DIR = "/content/drive/MyDrive/infinia-tts-case-study/clips/hindi"
REF_DIR = "/content/drive/MyDrive/infinia-tts-case-study/reference_clips"

WARM_CLIP = os.path.join(CLIPS_DIR, "hi_test_02_warm.wav")
WARMUP_CLIP = os.path.join(CLIPS_DIR, "hi_test_01_warmup.wav")

# Find reference audio
ref_candidates = [
    os.path.join(REF_DIR, f) for f in os.listdir(REF_DIR)
    if f.endswith((".wav", ".mp3", ".flac"))
] if os.path.exists(REF_DIR) else []

REFERENCE = ref_candidates[0] if ref_candidates else None

# The Hindi text that was synthesized
TEST_TEXT = "भारत में कृत्रिम बुद्धिमत्ता का विकास तेज़ी से हो रहा है और यह हमारे दैनिक जीवन को बदल रहा है।"

print("=" * 60)
print("HINDI EVAL — Real Metrics")
print("=" * 60)
print(f"  Warm clip:     {WARM_CLIP}")
print(f"  Reference:     {REFERENCE or 'NONE (speaker sim will be skipped)'}")
print(f"  Input text:    \"{TEST_TEXT[:50]}...\"")

# Check warm clip exists
if not os.path.exists(WARM_CLIP):
    print(f"\n[ERROR] Warm clip not found: {WARM_CLIP}")
    print("  Run pipeline_hindi.py first to generate clips.")
    sys.exit(1)

# ============================================================
# 1. ROUND-TRIP WER
# ============================================================
print("\n" + "-" * 60)
print("1. Round-trip WER (faster-whisper + jiwer)")
print("-" * 60)
print("  NOTE: Whisper's Hindi ASR may introduce more errors than for English.")
print("  WER inflation from ASR mistakes is a known limitation of this metric.")

try:
    wer_result = roundtrip_wer(WARM_CLIP, TEST_TEXT, language="hi")
    print(f"\n  WER:              {wer_result['wer']:.4f} ({wer_result['wer']*100:.1f}%)")
    print(f"  Target:           ≤10%")
    print(f"  Pass:             {'YES ✓' if wer_result['wer'] <= 0.10 else 'NO ✗'}")
    print(f"  Whisper heard:    \"{wer_result['transcript']}\"")
    print(f"  Original text:    \"{wer_result['reference']}\"")
    print(f"  Whisper model:    {wer_result['whisper_model']}")
    print(f"  Detected lang:    {wer_result['detected_language']} (prob: {wer_result['language_probability']:.2f})")
except Exception as e:
    print(f"  [FAIL] WER measurement failed: {e}")
    wer_result = None
    import traceback
    traceback.print_exc()

# ============================================================
# 2. SPEAKER COSINE SIMILARITY (MFCC-based)
# ============================================================
print("\n" + "-" * 60)
print("2. Speaker Similarity (MFCC cosine)")
print("-" * 60)

sim_result = None
if REFERENCE and os.path.exists(REFERENCE):
    try:
        sim_result = speaker_cosine_similarity(REFERENCE, WARM_CLIP)
        print(f"  Cosine sim:       {sim_result['cosine_similarity']:.4f}")
        print(f"  Target:           ≥0.75")
        print(f"  Pass:             {'YES ✓' if sim_result['cosine_similarity'] >= 0.75 else 'NO ✗'}")
        print(f"  Method:           {sim_result['method']}")
        print(f"  Reference:        {sim_result['reference_file']}")
        print(f"  Generated:        {sim_result['generated_file']}")
        print(f"\n  NOTE: Parler-TTS uses description-based voice (not cloning),")
        print(f"        so low similarity to your reference voice is expected.")
    except Exception as e:
        print(f"  [FAIL] Similarity measurement failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  [SKIP] No reference audio — speaker similarity skipped")

# ============================================================
# 3. ALSO RUN AGAINST WARMUP CLIP
# ============================================================
if os.path.exists(WARMUP_CLIP):
    print("\n" + "-" * 60)
    print("3. Cold-start clip eval (hi_test_01_warmup.wav)")
    print("-" * 60)

    try:
        wer_cold = roundtrip_wer(WARMUP_CLIP, TEST_TEXT, language="hi")
        print(f"  WER (cold):       {wer_cold['wer']:.4f} ({wer_cold['wer']*100:.1f}%)")
        print(f"  Whisper heard:    \"{wer_cold['transcript']}\"")
        if REFERENCE and os.path.exists(REFERENCE):
            sim_cold = speaker_cosine_similarity(REFERENCE, WARMUP_CLIP)
            print(f"  Cosine sim (cold):{sim_cold['cosine_similarity']:.4f}")
    except Exception as e:
        print(f"  [FAIL] Cold clip eval failed: {e}")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("HINDI METRICS SUMMARY")
print("=" * 60)
print(f"  {'Metric':<25} {'Value':<15} {'Target':<15} {'Pass?'}")
print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*5}")

print(f"  {'RTF (warm)':<25} {'TODO: from gen':<15} {'≤0.5':<15} {'check gen'}")

if wer_result:
    print(f"  {'WER':<25} {wer_result['wer']*100:<14.1f}% {'≤10%':<15} {'YES ✓' if wer_result['wer'] <= 0.10 else 'NO ✗'}")
else:
    print(f"  {'WER':<25} {'FAILED':<15} {'≤10%':<15} {'N/A'}")

if sim_result:
    print(f"  {'Speaker similarity':<25} {sim_result['cosine_similarity']:<15.4f} {'≥0.75':<15} {'YES ✓' if sim_result['cosine_similarity'] >= 0.75 else 'NO ✗'}")
else:
    print(f"  {'Speaker similarity':<25} {'SKIPPED':<15} {'≥0.75':<15} {'N/A'}")

print(f"  {'MOS':<25} {'TODO':<15} {'≥4.0/5':<15} {'human rating'}")

print("\n  NOTE: Hindi WER may be higher than English due to Whisper ASR limitations.")
print("  NOTE: Speaker sim expected low — Parler-TTS uses description, not cloning.")
print("=" * 60)
