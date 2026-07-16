"""
setup_colab.py — Infinia Multilingual TTS Case Study
Step 1: Environment setup + smoke test

Run this cell-by-cell in Google Colab (Runtime → T4 GPU).
It checks hardware, installs all dependencies for 3 TTS pipelines
+ eval harness, then runs a smoke test to confirm GPU inference works
before you spend time loading heavy TTS models.
"""

import subprocess
import sys
import os
import time

# ============================================================
# 1. GPU CHECK
# ============================================================
print("=" * 60)
print("STEP 1: GPU & Hardware Check")
print("=" * 60)

try:
    import torch
    if not torch.cuda.is_available():
        print("[FAIL] No CUDA GPU detected.")
        print("       Go to Runtime → Change runtime type → T4 GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[OK]   GPU detected: {gpu_name}")
    print(f"[OK]   VRAM: {vram_gb:.1f} GB")
    print(f"[OK]   PyTorch version: {torch.__version__}")
    print(f"[OK]   CUDA version: {torch.version.cuda}")

    if "T4" not in gpu_name:
        print(f"[WARN] Expected T4, got {gpu_name}. Results may differ from reported benchmarks.")
except ImportError:
    print("[FAIL] PyTorch not found — this should not happen on Colab.")
    print("       Try: !pip install torch")
    sys.exit(1)


# ============================================================
# 2. GOOGLE DRIVE MOUNT
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Google Drive Mount")
print("=" * 60)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("[OK]   Google Drive mounted at /content/drive")
except ImportError:
    print("[SKIP] Not running on Colab — skipping Drive mount.")
    print("       (This is fine for local testing.)")
except Exception as e:
    print(f"[WARN] Drive mount failed: {e}")
    print("       You can mount manually later. Continuing setup...")


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Create Output Directories")
print("=" * 60)

OUTPUT_BASE = "/content/drive/MyDrive/infinia-tts-case-study"
DIRS = [
    f"{OUTPUT_BASE}/clips/english",
    f"{OUTPUT_BASE}/clips/hindi",
    f"{OUTPUT_BASE}/clips/arabic",
    f"{OUTPUT_BASE}/reference_clips",
]

for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"[OK]   {d}")

print(f"[OK]   Output root: {OUTPUT_BASE}")


# ============================================================
# 4. PIP INSTALLS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Installing Dependencies")
print("=" * 60)

# ── Protect Colab's pre-installed PyTorch ──────────────────────
# Problem: `pip install chatterbox-tts` (and coqui-tts) can silently
# upgrade torch/torchaudio to versions built against CUDA 13, which
# doesn't exist on Colab's runtime (CUDA 12.8). This causes:
#   - libcudart.so.13: cannot open shared object file
#   - undefined symbol errors in torchaudio
#
# Fix: Write a pip constraints file that pins torch to whatever Colab
# already has. All subsequent pip installs use -c to respect this.

import torch

COLAB_TORCH = torch.__version__           # e.g. "2.11.0+cu128"
COLAB_CUDA  = torch.version.cuda          # e.g. "12.8"
CUDA_TAG    = "cu" + COLAB_CUDA.replace(".", "")  # e.g. "cu128"
PYTORCH_INDEX = f"https://download.pytorch.org/whl/{CUDA_TAG}"

# Pin transformers globally — coqui-tts uses internal functions like
# is_flax_available and is_torch_xla_available that were removed in
# transformers>=4.45. 4.44.2 is the newest version that works for:
#   - coqui-tts (XTTS-v2): needs is_flax_available  ✓
#   - MMS-TTS: needs transformers>=4.40              ✓
#   - parler-tts: needs transformers>=4.40            ✓
TRANSFORMERS_PIN = "4.44.2"

CONSTRAINTS_PATH = "/tmp/colab_constraints.txt"
with open(CONSTRAINTS_PATH, "w") as f:
    f.write(f"torch=={COLAB_TORCH}\n")
    f.write(f"transformers=={TRANSFORMERS_PIN}\n")
    # Also try to pin torchaudio if already installed
    try:
        import torchaudio
        f.write(f"torchaudio=={torchaudio.__version__}\n")
    except Exception:
        pass

print(f"[INFO] Colab PyTorch: {COLAB_TORCH} (CUDA {COLAB_CUDA})")
print(f"[INFO] Constraints file: {CONSTRAINTS_PATH}")
print(f"[INFO] PyTorch wheel index: {PYTORCH_INDEX}")

# ── Install groups ────────────────────────────────────────────
# Ordered: eval harness (lightest) → TTS models (heaviest).
# All installs use -c constraints to prevent torch version drift.

INSTALL_GROUPS = {
    # --- Eval harness (needed for all 3 languages) ---
    "eval-core": [
        "faster-whisper",   # ASR for round-trip WER
        "jiwer",            # Word error rate calculation
        "resemblyzer",      # Speaker embedding cosine similarity
        "soundfile",        # Audio I/O
        "librosa",          # Audio analysis utilities
    ],

    # --- English TTS: Chatterbox (primary) ---
    "english-chatterbox": [
        "chatterbox-tts",   # Resemble AI's Chatterbox
    ],

    # --- English/Arabic TTS: XTTS-v2 (fallback English, primary Arabic) ---
    # NOTE: Original 'TTS' package (Coqui) is abandoned and requires Python <3.12.
    # 'coqui-tts' is the community-maintained fork that supports Python 3.12+.
    "xtts-v2": [
        "coqui-tts",        # Community fork of Coqui TTS (includes XTTS-v2)
    ],

    # --- Hindi TTS: AI4Bharat Indic Parler-TTS (primary) ---
    # NOTE: This may need to be installed from git. We attempt pip first,
    # and fall back to git clone if it fails. We'll handle this in the
    # Hindi pipeline script if needed.
    "hindi-parler": [
        "parler-tts",       # Parler-TTS base; Indic variant may need git install
    ],

    # --- Arabic fallback: Meta MMS-TTS + transformers ---
    # transformers version is globally pinned in the constraints file
    # to 4.44.2 for coqui-tts compatibility. No per-group pin needed.
    "arabic-mms-fallback": [
        "transformers",       # Meta MMS-TTS; version pinned via constraints file
    ],
}

def pip_install(packages, group_name):
    """Install a list of packages with Colab torch constraints."""
    print(f"\n--- Installing group: {group_name} ---")
    for pkg in packages:
        print(f"  Installing {pkg}...", end=" ", flush=True)
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "-c", CONSTRAINTS_PATH, pkg],
            capture_output=True, text=True
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"[OK] ({elapsed:.1f}s)")
        else:
            # Constraint conflict = the package demands a different torch.
            # Fall back to --no-deps install (we handle deps manually).
            if "constraint" in result.stderr.lower() or "conflict" in result.stderr.lower():
                print(f"[RETRY --no-deps] ({elapsed:.1f}s)")
                print(f"    Constraint conflict detected — retrying without torch deps...")
                result2 = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "--no-deps", pkg],
                    capture_output=True, text=True
                )
                if result2.returncode == 0:
                    print(f"    [OK] Installed {pkg} (no-deps)")
                else:
                    print(f"    [FAIL] {pkg} still failed")
                    err_lines = result2.stderr.strip().split("\n")
                    for line in err_lines[-3:]:
                        print(f"      {line}")
            else:
                print(f"[FAIL] ({elapsed:.1f}s)")
                err_lines = result.stderr.strip().split("\n")
                for line in err_lines[-5:]:
                    print(f"    {line}")
                print(f"  [NOTE] {pkg} install failed — will attempt fix in per-language pipeline.")

for group_name, packages in INSTALL_GROUPS.items():
    pip_install(packages, group_name)

# ── Restore torchaudio from PyTorch's CUDA-matched wheel index ─
# Even with constraints, torchaudio can end up broken if chatterbox-tts
# pulled a source build or if Colab's pre-installed version had issues.
# Reinstall from PyTorch's official index which ships CUDA-matched wheels.
print("\n--- Restoring torchaudio from PyTorch CUDA-matched index ---")
print(f"  Index: {PYTORCH_INDEX}")
print("  Reinstalling torchaudio...", end=" ", flush=True)
t0 = time.time()
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--force-reinstall", "--no-deps",
     "--extra-index-url", PYTORCH_INDEX,
     "torchaudio"],
    capture_output=True, text=True
)
elapsed = time.time() - t0
if result.returncode == 0:
    print(f"[OK] ({elapsed:.1f}s)")
else:
    print(f"[FAIL] ({elapsed:.1f}s)")
    err_lines = result.stderr.strip().split("\n")
    for line in err_lines[-5:]:
        print(f"    {line}")
    print("  [NOTE] torchaudio restore failed — Chatterbox may still work without it.")


# ============================================================
# 5. VERIFY KEY IMPORTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Verify Key Imports")
print("=" * 60)

IMPORTS_TO_CHECK = {
    "torch":           "PyTorch (core)",
    "torchaudio":      "torchaudio (audio processing)",
    "faster_whisper":  "faster-whisper (ASR for WER eval)",
    "jiwer":           "jiwer (WER calculation)",
    "resemblyzer":     "resemblyzer (speaker similarity)",
    "soundfile":       "soundfile (audio I/O)",
    "librosa":         "librosa (audio utilities)",
    "numpy":           "numpy",
    "transformers":    "transformers (HuggingFace, for MMS-TTS)",
}

import_ok = 0
import_fail = 0
for module, label in IMPORTS_TO_CHECK.items():
    try:
        __import__(module)
        print(f"[OK]   {label}")
        import_ok += 1
    except (ImportError, OSError) as e:
        print(f"[FAIL] {label} — {e}")
        import_fail += 1

print(f"\n  {import_ok} OK, {import_fail} FAIL out of {len(IMPORTS_TO_CHECK)} checked")

# TTS model imports checked separately (they're heavier and may need
# special handling in per-language scripts)
print("\n--- TTS model imports (non-blocking) ---")
TTS_IMPORTS = {
    "chatterbox.tts": "Chatterbox TTS (English primary)",
    "TTS.api":        "Coqui TTS / XTTS-v2 (English fallback / Arabic primary)",
}
for module, label in TTS_IMPORTS.items():
    try:
        __import__(module)
        print(f"[OK]   {label}")
    except ImportError as e:
        print(f"[WARN] {label} — {e}")
        print(f"       Will attempt install/fix in per-language pipeline script.")
    except Exception as e:
        print(f"[WARN] {label} — import error: {e}")
        print(f"       Will attempt fix in per-language pipeline script.")


# ============================================================
# 6. SMOKE TEST — faster-whisper on GPU with synthetic audio
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Smoke Test — faster-whisper GPU inference")
print("=" * 60)
print("Generating a 2-second 440Hz sine tone, then transcribing it.")
print("This confirms: GPU works, faster-whisper loads, audio I/O works.\n")

try:
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    # Generate a 2-second 440Hz sine tone (A4 note)
    sample_rate = 16000
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    test_audio_path = "/tmp/smoke_test_tone.wav"
    sf.write(test_audio_path, tone, sample_rate)
    print(f"[OK]   Generated test tone: {test_audio_path} ({duration_sec}s, {sample_rate}Hz)")

    # Load faster-whisper small model on GPU
    print("[...] Loading faster-whisper 'small' model on CUDA...", flush=True)
    t0 = time.time()
    whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
    load_time = time.time() - t0
    print(f"[OK]   Model loaded in {load_time:.1f}s")

    # Transcribe the test tone
    print("[...] Transcribing test tone...", flush=True)
    t0 = time.time()
    segments, info = whisper_model.transcribe(test_audio_path, beam_size=5)
    segments_list = list(segments)  # materialize the generator
    transcribe_time = time.time() - t0

    transcript = " ".join([seg.text.strip() for seg in segments_list])
    print(f"[OK]   Transcription completed in {transcribe_time:.2f}s")
    print(f"[OK]   Detected language: {info.language} (prob: {info.language_probability:.2f})")
    print(f"[OK]   Transcript: '{transcript}'")
    print(f"       (A pure sine tone has no speech — empty or garbage transcript is expected and fine)")

    # Quick GPU memory check after loading a model
    gpu_mem_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
    gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    print(f"\n[OK]   GPU memory — allocated: {gpu_mem_used:.2f} GB, reserved: {gpu_mem_reserved:.2f} GB")

    # Clean up to free VRAM for TTS models
    del whisper_model
    torch.cuda.empty_cache()
    print("[OK]   Cleaned up whisper model from VRAM")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED ✓")
    print("=" * 60)

except Exception as e:
    print(f"\n[FAIL] Smoke test failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 60)
    print("SMOKE TEST FAILED ✗")
    print("Fix the error above before proceeding to TTS pipelines.")
    print("=" * 60)


# ============================================================
# 7. ENVIRONMENT SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ENVIRONMENT SUMMARY")
print("=" * 60)

summary_items = {
    "Python":       sys.version.split()[0],
    "PyTorch":      torch.__version__,
    "CUDA":         torch.version.cuda,
    "GPU":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
    "VRAM":         f"{vram_gb:.1f} GB",
    "Output dir":   OUTPUT_BASE,
}
for k, v in summary_items.items():
    print(f"  {k:15s}: {v}")

# Log exact versions of key packages for reproducibility
print("\n--- Package versions (for reproducibility) ---")
PACKAGES_TO_LOG = [
    "faster_whisper", "jiwer", "resemblyzer", "TTS",
    "transformers", "soundfile", "librosa", "numpy", "torchaudio",
]
for pkg_name in PACKAGES_TO_LOG:
    try:
        mod = __import__(pkg_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {pkg_name:20s}: {ver}")
    except (ImportError, OSError):
        print(f"  {pkg_name:20s}: NOT INSTALLED / BROKEN")

print("\n" + "=" * 60)
print("Setup complete. Proceed to Step 2: English TTS pipeline.")
print("=" * 60)
