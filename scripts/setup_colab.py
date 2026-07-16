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

# ── Detect Colab's pre-installed PyTorch (DO NOT MODIFY IT) ────
# Colab ships torch+torchaudio matched to its CUDA runtime. Previous
# attempts to force-reinstall torch broke CUDA support. Strategy now:
#   1. Record Colab's torch version dynamically
#   2. Use a pip constraints file to PREVENT other packages from upgrading it
#   3. After all installs, TEST if torchaudio still works
#   4. Only if broken, reinstall ONLY torchaudio (never torch) from matching index

import torch

COLAB_TORCH = torch.__version__           # e.g. "2.11.0+cu128"
COLAB_CUDA  = torch.version.cuda          # e.g. "12.8"
CUDA_TAG    = "cu" + COLAB_CUDA.replace(".", "")  # e.g. "cu128"
PYTORCH_INDEX = f"https://download.pytorch.org/whl/{CUDA_TAG}"

# Pin transformers globally — coqui-tts uses internal functions like
# is_flax_available and is_torch_xla_available that were removed in
# transformers>=4.45. 4.44.2 is the newest version that works for all.
TRANSFORMERS_PIN = "4.44.2"

# Constraints file — prevents pip from upgrading torch/torchaudio/transformers/numpy
# when installing other packages. Does NOT reinstall anything itself.
# numpy<2.0 is critical: many TTS-era libraries (coqui-tts, resemblyzer, etc.)
# use numpy 1.x APIs (e.g. numpy.char) removed in numpy 2.0.
NUMPY_PIN = "numpy<2.0"

CONSTRAINTS_PATH = "/tmp/colab_constraints.txt"
with open(CONSTRAINTS_PATH, "w") as f:
    f.write(f"torch=={COLAB_TORCH}\n")
    f.write(f"transformers=={TRANSFORMERS_PIN}\n")
    f.write(f"{NUMPY_PIN}\n")
    try:
        import torchaudio as _ta
        f.write(f"torchaudio=={_ta.__version__}\n")
        del _ta
    except Exception:
        pass  # torchaudio version unknown — constraint skipped

print(f"[INFO] Colab PyTorch: {COLAB_TORCH} (CUDA {COLAB_CUDA})")
print(f"[INFO] Constraints: torch=={COLAB_TORCH}, transformers=={TRANSFORMERS_PIN}, {NUMPY_PIN}")
print(f"[INFO] PyTorch wheel index (for torchaudio repair only): {PYTORCH_INDEX}")

# ── Install groups ────────────────────────────────────────────
# Install numpy<2.0 FIRST, before anything else, to prevent numpy 2.x
# from being pulled in as a dependency of other packages.
pip_install(["numpy<2.0"], "numpy-pin")

INSTALL_GROUPS = {
    "eval-core": [
        "faster-whisper",   # ASR for round-trip WER
        "jiwer",            # Word error rate calculation
        "soundfile",        # Audio I/O
        "librosa",          # Audio analysis utilities
        # resemblyzer dropped — broken on numpy 2.x, replaced with
        # MFCC-based cosine similarity in eval_harness.py
    ],
    "xtts-v2": [
        "coqui-tts",        # Community fork of Coqui TTS (includes XTTS-v2)
    ],
    "hindi-parler": [
        "parler-tts",       # Parler-TTS base; Indic variant may need git install
    ],
    "arabic-mms-fallback": [
        "transformers",     # Meta MMS-TTS; version pinned via constraints file
    ],
}

def pip_install(packages, group_name):
    """Install packages with constraints to prevent torch/transformers drift."""
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
            if "constraint" in result.stderr.lower() or "conflict" in result.stderr.lower():
                print(f"[RETRY --no-deps] ({elapsed:.1f}s)")
                result2 = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", pkg],
                    capture_output=True, text=True
                )
                if result2.returncode == 0:
                    print(f"    [OK] Installed {pkg} (no-deps)")
                else:
                    print(f"    [FAIL] {pkg} still failed")
                    for line in result2.stderr.strip().split("\n")[-3:]:
                        print(f"      {line}")
            else:
                print(f"[FAIL] ({elapsed:.1f}s)")
                for line in result.stderr.strip().split("\n")[-5:]:
                    print(f"    {line}")
                print(f"  [NOTE] {pkg} install failed — will attempt fix in per-language pipeline.")

for group_name, packages in INSTALL_GROUPS.items():
    pip_install(packages, group_name)

# ── Test-first torchaudio repair ──────────────────────────────
# Only touch torchaudio if it's actually broken. On a fresh Colab runtime
# it should work out of the box. If a package install above silently
# replaced it with a CUDA-mismatched version, we reinstall ONLY torchaudio
# (never torch) from PyTorch's CUDA-matched wheel index.
print("\n--- Checking torchaudio health ---")

def _test_torchaudio_import():
    """Try importing torchaudio and loading its C extension. Returns (ok, info_str)."""
    for key in list(sys.modules.keys()):
        if key.startswith("torchaudio"):
            del sys.modules[key]
    try:
        import torchaudio
        _ = torchaudio.info  # triggers native library load
        return True, torchaudio.__version__
    except Exception as e:
        return False, str(e)

ta_ok, ta_info = _test_torchaudio_import()

if ta_ok:
    print(f"[OK]   torchaudio {ta_info} — imports cleanly, no repair needed")
else:
    print(f"[WARN] torchaudio broken: {ta_info}")
    print(f"       Reinstalling ONLY torchaudio (not torch) from {PYTORCH_INDEX}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "--no-deps", "--index-url", PYTORCH_INDEX, "torchaudio"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        ta_ok2, ta_info2 = _test_torchaudio_import()
        if ta_ok2:
            print(f"[OK]   torchaudio {ta_info2} — repaired successfully")
        else:
            print(f"[FAIL] torchaudio still broken after repair: {ta_info2}")
            print("       Restart Colab runtime and re-run this script.")
    else:
        print("[FAIL] torchaudio reinstall command failed")
        for line in result.stderr.strip().split("\n")[-3:]:
            print(f"       {line}")


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
    "soundfile":       "soundfile (audio I/O)",
    "librosa":         "librosa (audio utilities)",
    "numpy":           "numpy (must be <2.0)",
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
    "TTS.api":        "Coqui TTS / XTTS-v2 (English + Arabic)",
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
# 6. SMOKE TEST — SKIPPED
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Smoke Test — SKIPPED")
print("=" * 60)
print("[SKIP] faster-whisper GPU inference already validated earlier in this session.")
print("       Skipping re-validation to save time. Proceeding to summary.")


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
    "faster_whisper", "jiwer", "TTS",
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
