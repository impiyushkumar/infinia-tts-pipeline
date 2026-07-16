"""
pipeline_english.py — English TTS Pipeline using XTTS-v2
Infinia Multilingual TTS Case Study

Model: XTTS-v2 via coqui-tts (community fork)
Features: Voice cloning from reference audio, multilingual support
Hardware: Colab T4 GPU (16GB VRAM)

Usage:
    %run scripts/pipeline_english.py
    
    Or import and call:
        from pipeline_english import generate, load_model
        model = load_model()
        wav_path = generate(model, "Hello world", "reference.wav")
"""

import os
import sys
import time
import torch
import numpy as np

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = "/content/drive/MyDrive/infinia-tts-case-study/clips/english"
REFERENCE_DIR = "/content/drive/MyDrive/infinia-tts-case-study/reference_clips"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "en"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL LOADING
# ============================================================
def load_model():
    """
    Load XTTS-v2 model onto GPU.
    Returns the TTS model instance.
    
    Logs cold-start load time for benchmarking.
    """
    from TTS.api import TTS

    print(f"[...] Loading XTTS-v2 model: {MODEL_NAME}")
    print(f"[...] This downloads ~1.8GB on first run...", flush=True)

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
    Generate speech from text using XTTS-v2 with voice cloning.
    
    Args:
        tts: Loaded TTS model instance (from load_model())
        text: Input text to synthesize
        reference_audio_path: Path to reference audio for voice cloning
        output_filename: Optional output filename (auto-generated if None)
    
    Returns:
        dict with:
            - wav_path: Path to generated audio file
            - generation_time: Time taken for inference (seconds)
            - audio_duration: Duration of generated audio (seconds)
            - text: Input text used
    """
    if not os.path.exists(reference_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

    # Generate output filename if not provided
    if output_filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Sanitize text for filename (first 30 chars)
        safe_text = "".join(c if c.isalnum() or c == " " else "_" for c in text[:30]).strip().replace(" ", "_")
        output_filename = f"en_{timestamp}_{safe_text}.wav"

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
# MAIN — test with one real sentence
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("English TTS Pipeline — XTTS-v2")
    print("=" * 60)

    # Check for reference audio
    # NOTE: You need to upload your own voice reference clip first!
    # Record ~10-20 seconds of yourself speaking clearly in English,
    # save as WAV, and upload to the reference_clips directory.
    ref_candidates = [
        os.path.join(REFERENCE_DIR, f)
        for f in os.listdir(REFERENCE_DIR)
        if f.endswith((".wav", ".mp3", ".flac"))
    ] if os.path.exists(REFERENCE_DIR) else []

    if not ref_candidates:
        print("\n[ERROR] No reference audio found!")
        print(f"  Upload a WAV file of your voice to: {REFERENCE_DIR}")
        print("  Record ~10-20 seconds of clear English speech.")
        print("  Then re-run this script.")
        sys.exit(1)

    reference_audio = ref_candidates[0]
    print(f"\n[INFO] Using reference audio: {reference_audio}")

    # Load model (cold start)
    print("\n--- Cold start load ---")
    tts = load_model()

    # Test sentence
    test_text = "The rapid advancement of artificial intelligence is reshaping how we interact with technology in our daily lives."

    # First generation (includes any remaining warmup)
    print("\n--- Generation 1 (may include warmup) ---")
    result1 = generate(tts, test_text, reference_audio, "en_test_01_warmup.wav")

    # Second generation (warm inference — this is the real benchmark)
    print("\n--- Generation 2 (warm inference — benchmark this) ---")
    result2 = generate(tts, test_text, reference_audio, "en_test_02_warm.wav")

    # Summary
    print("\n" + "=" * 60)
    print("ENGLISH PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Reference audio:  {os.path.basename(reference_audio)}")
    print(f"  Test text:        \"{test_text[:60]}...\"")
    print(f"  Gen 1 (warmup):   {result1['generation_time']:.2f}s → {result1['audio_duration']:.2f}s audio")
    print(f"  Gen 2 (warm):     {result2['generation_time']:.2f}s → {result2['audio_duration']:.2f}s audio")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"\n  Clips saved — listen to them before trusting any numbers!")
    print("=" * 60)
