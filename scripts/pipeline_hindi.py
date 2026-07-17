"""
pipeline_hindi.py — Hindi TTS Pipeline using Meta MMS-TTS
Infinia Multilingual TTS Case Study

Model: facebook/mms-tts-hin (Meta Massively Multilingual Speech)
Note: MMS-TTS is a VITS-based model — no voice cloning capability.
      Generates speech with a fixed speaker voice.
      Indic Parler-TTS was dropped (HuggingFace gating + version conflicts).
Hardware: Colab T4 GPU (16GB VRAM)

Usage:
    %run scripts/pipeline_hindi.py
"""

import os
import sys
import time
import torch

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = "/content/drive/MyDrive/infinia-tts-case-study/clips/hindi"
REFERENCE_DIR = "/content/drive/MyDrive/infinia-tts-case-study/reference_clips"
MODEL_NAME = "facebook/mms-tts-hin"
LANGUAGE = "hi"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL LOADING
# ============================================================
def load_model():
    """
    Load MMS-TTS Hindi model onto GPU.
    Returns (model, tokenizer) tuple.
    """
    from transformers import VitsModel, VitsTokenizer

    print(f"[...] Loading MMS-TTS Hindi: {MODEL_NAME}")
    print(f"[...] Downloading model on first run...", flush=True)

    t0 = time.time()
    tokenizer = VitsTokenizer.from_pretrained(MODEL_NAME)
    model = VitsModel.from_pretrained(MODEL_NAME).to("cuda")
    load_time = time.time() - t0

    print(f"[OK]  Model loaded in {load_time:.1f}s")

    gpu_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"[OK]  GPU memory after load: {gpu_mem:.2f} GB")

    return model, tokenizer


# ============================================================
# GENERATION
# ============================================================
def generate(model_tuple, text, reference_audio_path=None, output_filename=None):
    """
    Generate Hindi speech from text using MMS-TTS.

    Args:
        model_tuple: (model, tokenizer) from load_model()
        text: Input text in Hindi (Devanagari script)
        reference_audio_path: NOT USED (MMS-TTS has no voice cloning).
                              Accepted for API consistency with pipeline_english.py.
        output_filename: Optional output filename (auto-generated if None)

    Returns:
        dict with wav_path, generation_time, audio_duration, text, etc.
    """
    model, tokenizer = model_tuple

    if reference_audio_path:
        print(f"[NOTE] reference_audio_path ignored — MMS-TTS has no voice cloning.")

    # Generate output filename
    if output_filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(c if c.isalnum() or c == " " else "_" for c in text[:30]).strip().replace(" ", "_")
        output_filename = f"hi_{timestamp}_{safe_text}.wav"

    wav_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n[GEN] Text: \"{text}\"")
    print(f"[GEN] Language: {LANGUAGE}")

    # --- Tokenize ---
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # --- Inference ---
    t0 = time.time()
    with torch.no_grad():
        output = model(**inputs)
    generation_time = time.time() - t0

    # --- Save audio ---
    import soundfile as sf
    audio_arr = output.waveform.squeeze().cpu().numpy()
    sample_rate = model.config.sampling_rate
    sf.write(wav_path, audio_arr, sample_rate)

    audio_duration = len(audio_arr) / sample_rate

    # --- Log results ---
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
    print(f"[OK]  Generated: {wav_path}")
    print(f"[OK]  Generation time: {generation_time:.2f}s")
    print(f"[OK]  Audio duration: {audio_duration:.2f}s")
    print(f"[OK]  Sample rate: {sample_rate}Hz")
    print(f"[OK]  RTF: {rtf:.3f} (target: ≤0.5)")

    return {
        "wav_path": wav_path,
        "generation_time": generation_time,
        "audio_duration": audio_duration,
        "text": text,
        "language": LANGUAGE,
        "model": MODEL_NAME,
        "reference_audio": None,
    }


# ============================================================
# MAIN — test with real Hindi sentences
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Hindi TTS Pipeline — Meta MMS-TTS")
    print("=" * 60)

    # Reference audio (for eval comparison only, not used in generation)
    ref_candidates = [
        os.path.join(REFERENCE_DIR, f)
        for f in os.listdir(REFERENCE_DIR)
        if f.endswith((".wav", ".mp3", ".flac"))
    ] if os.path.exists(REFERENCE_DIR) else []

    reference_audio = ref_candidates[0] if ref_candidates else None

    # Load model (cold start)
    print("\n--- Cold start load ---")
    model_tuple = load_model()

    # Test sentence in Hindi (Devanagari)
    test_text = "भारत में कृत्रिम बुद्धिमत्ता का विकास तेज़ी से हो रहा है और यह हमारे दैनिक जीवन को बदल रहा है।"

    # First generation (includes warmup)
    print("\n--- Generation 1 (may include warmup) ---")
    result1 = generate(model_tuple, test_text, reference_audio, "hi_test_01_warmup.wav")

    # Second generation (warm inference — benchmark this)
    print("\n--- Generation 2 (warm inference — benchmark this) ---")
    result2 = generate(model_tuple, test_text, reference_audio, "hi_test_02_warm.wav")

    # Summary
    print("\n" + "=" * 60)
    print("HINDI PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Voice:            fixed MMS-TTS speaker (no cloning)")
    print(f"  Test text:        \"{test_text[:50]}...\"")
    print(f"  Gen 1 (warmup):   {result1['generation_time']:.2f}s → {result1['audio_duration']:.2f}s audio")
    print(f"  Gen 2 (warm):     {result2['generation_time']:.2f}s → {result2['audio_duration']:.2f}s audio")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"\n  Clips saved — listen to them before trusting any numbers!")
    print("=" * 60)
