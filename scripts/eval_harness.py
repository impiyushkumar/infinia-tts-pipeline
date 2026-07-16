"""
eval_harness.py — Reusable Evaluation Harness
Infinia Multilingual TTS Case Study

4 evaluation functions, used across all 3 language pipelines.
All functions take file paths and return real numbers only — never
mock/placeholder values. If a measurement fails, it raises an exception
rather than returning a fake number.

Usage:
    from eval_harness import measure_rtf, measure_latency, roundtrip_wer, speaker_cosine_similarity
"""

import time


# ============================================================
# 1. REAL-TIME FACTOR (RTF)
# ============================================================
def measure_rtf(generation_time, audio_duration):
    """
    Compute Real-Time Factor: how fast the model generates relative to
    real-time playback.

    Args:
        generation_time: Time taken to generate the audio (seconds)
        audio_duration: Duration of the generated audio (seconds)

    Returns:
        float: RTF value. <1.0 means faster than real-time. Target: ≤0.5

    Example:
        If generation takes 1.5s and audio is 5s long → RTF = 0.3 (good)
    """
    if audio_duration <= 0:
        raise ValueError(f"audio_duration must be positive, got {audio_duration}")
    if generation_time < 0:
        raise ValueError(f"generation_time must be non-negative, got {generation_time}")

    rtf = generation_time / audio_duration
    return rtf


# ============================================================
# 2. LATENCY
# ============================================================
def measure_latency(start_time, end_time):
    """
    Measure latency from inference start to first audio output (or full
    clip for batch inference).

    For batch models (like XTTS-v2), this is start-to-complete time.
    For streaming models, pass the time when first chunk was received.

    Args:
        start_time: time.time() captured before inference call
        end_time: time.time() captured when first audio is available

    Returns:
        float: Latency in milliseconds. Target: <500ms streaming / <2s batch
    """
    if end_time < start_time:
        raise ValueError(f"end_time ({end_time}) must be >= start_time ({start_time})")

    latency_ms = (end_time - start_time) * 1000.0
    return latency_ms


# ============================================================
# 3. ROUND-TRIP WORD ERROR RATE (WER)
# ============================================================
def roundtrip_wer(generated_audio_path, input_text, language="en", whisper_model_size="small"):
    """
    Measure round-trip WER: transcribe generated audio with faster-whisper,
    then compare the ASR transcript to the original input text using jiwer.

    Known limitation: WER can be inflated by ASR errors (whisper mishearing
    correct audio). This is a limitation of the metric, not the TTS model.
    We note this honestly in the final report.

    Args:
        generated_audio_path: Path to the generated WAV file
        input_text: The original text that was synthesized
        language: Language code for whisper (e.g., "en", "hi", "ar")
        whisper_model_size: Whisper model size ("small" recommended for speed/accuracy)

    Returns:
        dict with:
            - wer: Word Error Rate as a float (0.0 = perfect, 1.0 = 100% errors)
            - transcript: What whisper heard
            - reference: Normalized input text
            - whisper_model: Model size used
    """
    import os
    from faster_whisper import WhisperModel
    from jiwer import wer as compute_wer

    if not os.path.exists(generated_audio_path):
        raise FileNotFoundError(f"Audio file not found: {generated_audio_path}")

    # Load whisper model (reuse if already loaded via module-level cache)
    global _whisper_cache
    if "_whisper_cache" not in globals():
        _whisper_cache = {}

    cache_key = f"{whisper_model_size}_cuda"
    if cache_key not in _whisper_cache:
        print(f"  [WER] Loading faster-whisper '{whisper_model_size}' on CUDA...", flush=True)
        _whisper_cache[cache_key] = WhisperModel(
            whisper_model_size, device="cuda", compute_type="float16"
        )

    model = _whisper_cache[cache_key]

    # Transcribe
    segments, info = model.transcribe(
        generated_audio_path,
        beam_size=5,
        language=language,
    )
    transcript = " ".join([seg.text.strip() for seg in segments])

    # Normalize both texts for fair comparison (lowercase, strip)
    reference_normalized = input_text.lower().strip()
    transcript_normalized = transcript.lower().strip()

    # Compute WER
    if not reference_normalized:
        raise ValueError("Input text is empty — cannot compute WER")

    wer_score = compute_wer(reference_normalized, transcript_normalized)

    return {
        "wer": wer_score,
        "transcript": transcript,
        "reference": input_text,
        "whisper_model": whisper_model_size,
        "detected_language": info.language,
        "language_probability": info.language_probability,
    }


# ============================================================
# 4. SPEAKER COSINE SIMILARITY (MFCC-based)
# ============================================================
def speaker_cosine_similarity(reference_audio_path, generated_audio_path):
    """
    Compare speaker similarity between reference and generated audio using
    MFCC-based cosine similarity.

    Uses librosa to extract MFCC features, averages across time frames to
    get a single feature vector per clip, then computes cosine similarity.

    Limitation (noted in final write-up): This is a simpler proxy than a
    dedicated speaker-embedding model like resemblyzer. It captures timbral
    similarity but is less robust to variations in content/prosody.
    resemblyzer was dropped due to numpy 2.x incompatibility.

    Args:
        reference_audio_path: Path to reference voice clip
        generated_audio_path: Path to generated voice clip

    Returns:
        dict with:
            - cosine_similarity: float between -1 and 1. Target: ≥0.75
            - method: description of similarity method used
            - reference_file: basename of reference audio
            - generated_file: basename of generated audio
    """
    import os
    import numpy as np
    import librosa

    if not os.path.exists(reference_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
    if not os.path.exists(generated_audio_path):
        raise FileNotFoundError(f"Generated audio not found: {generated_audio_path}")

    # Load audio and extract MFCCs (20 coefficients is standard)
    ref_audio, ref_sr = librosa.load(reference_audio_path, sr=16000)
    gen_audio, gen_sr = librosa.load(generated_audio_path, sr=16000)

    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=16000, n_mfcc=20)
    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=16000, n_mfcc=20)

    # Average across time frames to get a single vector per clip
    ref_vec = np.mean(ref_mfcc, axis=1)
    gen_vec = np.mean(gen_mfcc, axis=1)

    # Cosine similarity
    dot = np.dot(ref_vec, gen_vec)
    norm = np.linalg.norm(ref_vec) * np.linalg.norm(gen_vec)
    cosine_sim = dot / norm if norm > 0 else 0.0

    return {
        "cosine_similarity": float(cosine_sim),
        "method": "MFCC mean vector cosine similarity (librosa, 20 coefficients)",
        "reference_file": os.path.basename(reference_audio_path),
        "generated_file": os.path.basename(generated_audio_path),
    }


# ============================================================
# CLEANUP — free GPU memory from cached models
# ============================================================
def cleanup():
    """Free GPU memory from cached whisper models."""
    import torch
    global _whisper_cache

    if "_whisper_cache" in globals() and _whisper_cache:
        _whisper_cache.clear()
        print("[EVAL] Cleared whisper model cache")

    torch.cuda.empty_cache()
    print("[EVAL] GPU cache cleared")


# ============================================================
# MAIN — self-test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Eval Harness — Self-Test")
    print("=" * 60)

    # Test RTF
    print("\n--- measure_rtf ---")
    rtf = measure_rtf(generation_time=1.5, audio_duration=5.0)
    print(f"  RTF = {rtf:.3f} (expected: 0.300)")
    assert abs(rtf - 0.3) < 0.001, f"RTF test failed: {rtf}"
    print("  [PASS]")

    # Test latency
    print("\n--- measure_latency ---")
    t_start = time.time()
    time.sleep(0.1)
    t_end = time.time()
    latency = measure_latency(t_start, t_end)
    print(f"  Latency = {latency:.1f}ms (expected: ~100ms)")
    assert 50 < latency < 300, f"Latency test failed: {latency}"
    print("  [PASS]")

    print("\n--- roundtrip_wer ---")
    print("  [SKIP] Requires real audio files — tested during pipeline runs")

    print("\n--- speaker_cosine_similarity ---")
    print("  [SKIP] Requires real audio files — tested during pipeline runs")

    print("\n" + "=" * 60)
    print("Eval harness self-test complete.")
    print("=" * 60)

