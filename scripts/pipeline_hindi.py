"""
pipeline_hindi.py — Hindi TTS Pipeline using AI4Bharat Indic Parler-TTS
Infinia Multilingual TTS Case Study

Model: AI4Bharat Indic Parler-TTS (purpose-built for Indic languages)
Note: Parler-TTS uses text descriptions for voice control, NOT voice cloning.
      The reference_audio_path parameter is accepted for API consistency but
      is not used for cloning — this is a known limitation vs XTTS-v2.
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
MODEL_NAME = "ai4bharat/indic-parler-tts"
LANGUAGE = "hi"

# Voice description for Parler-TTS (controls speaker characteristics)
# Parler-TTS doesn't clone from reference audio — it uses text descriptions.
VOICE_DESCRIPTION = "A male speaker with a clear, natural Indian Hindi accent, speaking at a moderate pace in a calm and articulate tone."

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL LOADING
# ============================================================
def load_model():
    """
    Load Indic Parler-TTS model onto GPU.
    Returns (model, tokenizer) tuple.
    """
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    print(f"[...] Loading Indic Parler-TTS: {MODEL_NAME}")
    print(f"[...] This downloads the model on first run...", flush=True)

    t0 = time.time()
    
    # Try primary model, fall back to alternatives if not found
    model_candidates = [
        MODEL_NAME,
        "ai4bharat/indic-parler-tts-pretrained",
        "ai4bharat/indic-parler-tts-hindi",
    ]
    
    model = None
    tokenizer = None
    used_model = None
    
    for candidate in model_candidates:
        try:
            print(f"[...] Trying model: {candidate}...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            model = ParlerTTSForConditionalGeneration.from_pretrained(candidate).to("cuda")
            used_model = candidate
            break
        except Exception as e:
            print(f"[WARN] {candidate} failed: {e}")
            continue
    
    if model is None:
        print("[FAIL] No Indic Parler-TTS model could be loaded.")
        print("       Check HuggingFace for the correct model ID:")
        print("       https://huggingface.co/ai4bharat")
        sys.exit(1)
    
    load_time = time.time() - t0
    print(f"[OK]  Model loaded: {used_model} in {load_time:.1f}s")

    gpu_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"[OK]  GPU memory after load: {gpu_mem:.2f} GB")

    return model, tokenizer, used_model


# ============================================================
# GENERATION
# ============================================================
def generate(model_tuple, text, reference_audio_path=None, output_filename=None):
    """
    Generate Hindi speech from text using Indic Parler-TTS.

    Args:
        model_tuple: (model, tokenizer, model_name) from load_model()
        text: Input text in Hindi (Devanagari script)
        reference_audio_path: NOT USED for cloning (Parler-TTS uses descriptions).
                              Accepted for API consistency with pipeline_english.py.
        output_filename: Optional output filename (auto-generated if None)

    Returns:
        dict with wav_path, generation_time, audio_duration, text, etc.
    """
    model, tokenizer, model_name = model_tuple

    if reference_audio_path:
        print(f"[NOTE] reference_audio_path provided but Parler-TTS uses text descriptions,")
        print(f"       not voice cloning. Reference is ignored for generation.")

    # Generate output filename
    if output_filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(c if c.isalnum() or c == " " else "_" for c in text[:30]).strip().replace(" ", "_")
        output_filename = f"hi_{timestamp}_{safe_text}.wav"

    wav_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n[GEN] Text: \"{text}\"")
    print(f"[GEN] Voice: \"{VOICE_DESCRIPTION[:60]}...\"")
    print(f"[GEN] Language: {LANGUAGE}")

    # --- Tokenize ---
    description_tokens = tokenizer(VOICE_DESCRIPTION, return_tensors="pt").to("cuda")
    prompt_tokens = tokenizer(text, return_tensors="pt").to("cuda")

    # --- Inference ---
    t0 = time.time()
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_tokens.input_ids,
            attention_mask=description_tokens.attention_mask,
            prompt_input_ids=prompt_tokens.input_ids,
            prompt_attention_mask=prompt_tokens.attention_mask,
        )
    generation_time = time.time() - t0

    # --- Save audio ---
    import soundfile as sf
    audio_arr = generation.cpu().numpy().squeeze()
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
        "model": model_name,
        "reference_audio": reference_audio_path,
        "voice_description": VOICE_DESCRIPTION,
    }


# ============================================================
# MAIN — test with real Hindi sentences
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Hindi TTS Pipeline — Indic Parler-TTS")
    print("=" * 60)

    # Reference audio (for eval comparison, not for cloning)
    ref_candidates = [
        os.path.join(REFERENCE_DIR, f)
        for f in os.listdir(REFERENCE_DIR)
        if f.endswith((".wav", ".mp3", ".flac"))
    ] if os.path.exists(REFERENCE_DIR) else []

    reference_audio = ref_candidates[0] if ref_candidates else None
    if reference_audio:
        print(f"\n[INFO] Reference audio (for eval only): {reference_audio}")
    else:
        print(f"\n[INFO] No reference audio found — eval will skip speaker similarity")

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
    print(f"  Model:            {result2['model']}")
    print(f"  Voice:            description-based (no cloning)")
    print(f"  Test text:        \"{test_text[:50]}...\"")
    print(f"  Gen 1 (warmup):   {result1['generation_time']:.2f}s → {result1['audio_duration']:.2f}s audio")
    print(f"  Gen 2 (warm):     {result2['generation_time']:.2f}s → {result2['audio_duration']:.2f}s audio")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"\n  Clips saved — listen to them before trusting any numbers!")
    print("=" * 60)
