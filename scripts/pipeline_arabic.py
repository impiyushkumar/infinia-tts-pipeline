"""
pipeline_arabic.py — Arabic TTS Pipeline using XTTS-v2
Infinia Multilingual TTS Case Study

Model: XTTS-v2 via coqui-tts (same model as English pipeline)
Features: Voice cloning from reference audio, multilingual support
Hardware: Colab T4 GPU (16GB VRAM)

Usage:
    %run scripts/pipeline_arabic.py
"""

import os
import sys
import time
import torch

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = "/content/drive/MyDrive/infinia-tts-case-study/clips/arabic"
REFERENCE_DIR = "/content/drive/MyDrive/infinia-tts-case-study/reference_clips"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "ar"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL LOADING
# ============================================================
def load_model():
    """
    Load XTTS-v2 model onto GPU.
    Same model as English pipeline — XTTS-v2 supports Arabic natively.
    """
    from TTS.api import TTS

    print(f"[...] Loading XTTS-v2 model: {MODEL_NAME}")
    print(f"[...] (May use cached download from English pipeline run)", flush=True)

    t0 = time.time()
    tts = TTS(model_name=MODEL_NAME).to("cuda")
    load_time = time.time() - t0

    print(f"[OK]  Model loaded in {load_time:.1f}s")
    print(f"[OK]  Device: {next(tts.synthesizer.tts_model.parameters()).device}")

    gpu_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"[OK]  GPU memory after load: {gpu_mem:.2f} GB")

    return tts


# ============================================================
# GENERATION
# ============================================================
def generate(tts, text, reference_audio_path, output_filename=None):
    """
    Generate Arabic speech from text using XTTS-v2 with voice cloning.

    Args:
        tts: Loaded TTS model instance (from load_model())
        text: Input text in Arabic script
        reference_audio_path: Path to reference audio for voice cloning
        output_filename: Optional output filename (auto-generated if None)

    Returns:
        dict with wav_path, generation_time, audio_duration, text, etc.
    """
    if not os.path.exists(reference_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

    if output_filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(c if c.isalnum() or c == " " else "_" for c in text[:30]).strip().replace(" ", "_")
        output_filename = f"ar_{timestamp}_{safe_text}.wav"

    wav_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n[GEN] Text: \"{text}\"")
    print(f"[GEN] Reference: {os.path.basename(reference_audio_path)}")
    print(f"[GEN] Language: {LANGUAGE}")

    # --- Inference ---
    t0 = time.time()
    tts.tts_to_file(
        text=text,
        speaker_wav=reference_audio_path,
        language=LANGUAGE,
        file_path=wav_path,
    )
    generation_time = time.time() - t0

    # --- Get audio duration ---
    import soundfile as sf
    audio_data, sample_rate = sf.read(wav_path)
    audio_duration = len(audio_data) / sample_rate

    # --- Log results ---
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
    print(f"[OK]  Generated: {wav_path}")
    print(f"[OK]  Generation time: {generation_time:.2f}s")
    print(f"[OK]  Audio duration: {audio_duration:.2f}s")
    print(f"[OK]  RTF: {rtf:.3f} (target: ≤0.5)")

    return {
        "wav_path": wav_path,
        "generation_time": generation_time,
        "audio_duration": audio_duration,
        "text": text,
        "language": LANGUAGE,
        "model": MODEL_NAME,
        "reference_audio": reference_audio_path,
    }


# ============================================================
# MAIN — test with real Arabic sentence
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Arabic TTS Pipeline — XTTS-v2")
    print("=" * 60)

    # Check for reference audio (reuse English reference)
    ref_candidates = [
        os.path.join(REFERENCE_DIR, f)
        for f in os.listdir(REFERENCE_DIR)
        if f.endswith((".wav", ".mp3", ".flac"))
    ] if os.path.exists(REFERENCE_DIR) else []

    if not ref_candidates:
        print(f"\n[ERROR] No reference audio found in {REFERENCE_DIR}")
        print("  Upload a WAV file of your voice (same one used for English).")
        sys.exit(1)

    reference_audio = ref_candidates[0]
    print(f"\n[INFO] Using reference audio: {reference_audio}")

    # Load model (cold start)
    print("\n--- Cold start load ---")
    tts = load_model()

    # Test sentence in Modern Standard Arabic
    test_text = "إن التطور السريع في الذكاء الاصطناعي يعيد تشكيل طريقة تفاعلنا مع التكنولوجيا في حياتنا اليومية."

    # First generation (includes warmup)
    print("\n--- Generation 1 (may include warmup) ---")
    result1 = generate(tts, test_text, reference_audio, "ar_test_01_warmup.wav")

    # Second generation (warm inference — benchmark this)
    print("\n--- Generation 2 (warm inference — benchmark this) ---")
    result2 = generate(tts, test_text, reference_audio, "ar_test_02_warm.wav")

    # Summary
    print("\n" + "=" * 60)
    print("ARABIC PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Reference audio:  {os.path.basename(reference_audio)}")
    print(f"  Test text:        \"{test_text[:50]}...\"")
    print(f"  Gen 1 (warmup):   {result1['generation_time']:.2f}s → {result1['audio_duration']:.2f}s audio")
    print(f"  Gen 2 (warm):     {result2['generation_time']:.2f}s → {result2['audio_duration']:.2f}s audio")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"\n  Clips saved — listen to them before trusting any numbers!")
    print(f"  NOTE: XTTS-v2 Arabic quality may be lower than English — the brief")
    print(f"        warns that fast/good models usually don't cover Arabic well.")
    print("=" * 60)
