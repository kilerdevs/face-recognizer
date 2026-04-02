"""
bootstrap.py  –  auto-install missing packages and report system/CUDA status.
Called at the very start of main.py before any third-party imports.
"""

import importlib.util
import os
import platform
import subprocess
import sys
import time

# (import_name, pip_install_spec)
_PACKAGES = [
    ("PyQt5",       "PyQt5>=5.15.0"),
    ("rawpy",       "rawpy>=0.19.0"),
    ("imageio",     "imageio>=2.31.0"),
    ("cv2",         "opencv-python>=4.8.0"),
    ("numpy",       "numpy>=1.24.0"),
    ("scipy",       "scipy>=1.10.0"),
    ("sklearn",     "scikit-learn>=1.3.0"),
    ("PIL",         "Pillow>=10.0.0"),
    # insightface: version-pinned entry is a hint only; bootstrap handles
    # the multi-strategy install separately (see _install_insightface below).
    ("insightface", "insightface>=0.7.3"),
]

# Packages that need a more careful multi-strategy install on newer Pythons.
_FALLBACK_INSTALL: dict = {
    "insightface": ["insightface>=0.7.3", "insightface", "insightface --pre"],
}


def _importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pip(spec: str) -> bool:
    """Install a package. Returns True on success."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", spec, "--quiet",
             "--disable-pip-version-check"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0
    except Exception as exc:
        print(f"    pip error: {exc}")
        return False


def _install_onnxruntime() -> str:
    """
    Try onnxruntime-gpu first, fall back to CPU build.
    Returns one of: 'gpu', 'cpu', 'missing'
    """
    if _importable("onnxruntime"):
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                return "gpu"
            return "cpu"
        except Exception:
            pass

    print("    Installing onnxruntime-gpu … ", end="", flush=True)
    if _pip("onnxruntime-gpu>=1.17.0"):
        # Quick validation
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "import onnxruntime as o; print('CUDAExecutionProvider' "
                 "in o.get_available_providers())"],
                capture_output=True, text=True, timeout=30,
            )
            has_cuda = result.stdout.strip() == "True"
        except Exception:
            has_cuda = False

        if has_cuda:
            print("done  ✓  GPU")
            return "gpu"
        print("done  (CUDA not detected – GPU inference unavailable)")
        return "cpu"

    print("failed – trying CPU-only onnxruntime … ", end="", flush=True)
    if _pip("onnxruntime>=1.17.0"):
        print("done")
        return "cpu"

    print("FAILED")
    return "missing"


def _nvidia_smi() -> dict:
    """Query nvidia-smi for GPU info. Returns empty dict on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}

        line  = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return {}

        # CUDA version is printed at the top of normal nvidia-smi output
        cuda_ver = "N/A"
        try:
            full = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            for ln in full.stdout.splitlines():
                if "CUDA Version" in ln:
                    cuda_ver = ln.split("CUDA Version:")[-1].strip().split()[0]
                    break
        except Exception:
            pass

        return {
            "gpu_name":       parts[0],
            "vram_total_mb":  int(parts[1]),
            "vram_free_mb":   int(parts[2]),
            "driver_version": parts[3],
            "cuda_version":   cuda_ver,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def run(auto_install: bool = True) -> dict:
    """
    Check / install all dependencies and collect system info.

    Returns a dict with keys:
        ok          – bool, True if all critical packages are present
        ort_mode    – 'gpu' | 'cpu' | 'missing'
        gpu         – dict from nvidia-smi (may be empty)
        python_ver  – str
        platform    – str
        failures    – list[str]
    """
    SEP = "─" * 62
    print(SEP)
    print("  Face Recognizer  –  Environment Bootstrap")
    print(SEP)
    py_ver = sys.version_info
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Platform : {platform.platform()}")
    if py_ver >= (3, 13):
        print(
            f"\n  ⚠  Python {py_ver.major}.{py_ver.minor} is newer than most ML "
            "packages officially support.\n"
            "     Some packages (e.g. insightface) may lack pre-built wheels\n"
            "     and could fail to install or run.  Python 3.11/3.12 is recommended.\n"
        )
    print()

    failures = []

    # --- core packages ---
    print("  Core packages:")
    for import_name, pip_spec in _PACKAGES:
        if _importable(import_name):
            print(f"    ✓  {import_name}")
        elif auto_install:
            specs = _FALLBACK_INSTALL.get(import_name, [pip_spec])
            installed = False
            for spec in specs:
                label = spec if spec != pip_spec else pip_spec
                print(f"    ↓  Installing {label} … ", end="", flush=True)
                if _pip(spec):
                    print("done")
                    installed = True
                    break
                print("failed – trying next …")
            if not installed:
                print(f"    ✗  {import_name}  all install attempts FAILED")
                failures.append(pip_spec)
        else:
            print(f"    ✗  {import_name}  (not installed)")
            failures.append(import_name)

    # --- onnxruntime (GPU preferred) ---
    print()
    print("  ONNX Runtime / GPU:")
    ort_mode = _install_onnxruntime()

    ort_ver = "N/A"
    try:
        import onnxruntime as ort
        ort_ver = ort.__version__
        print(f"    onnxruntime {ort_ver}")
        print(f"    providers: {', '.join(ort.get_available_providers())}")
    except ImportError:
        if ort_mode == "missing":
            failures.append("onnxruntime")

    # --- GPU info via nvidia-smi ---
    print()
    print("  NVIDIA GPU:")
    gpu = _nvidia_smi()
    if gpu:
        print(f"    GPU    : {gpu['gpu_name']}")
        print(f"    VRAM   : {gpu['vram_total_mb']:,} MB total  /  "
              f"{gpu['vram_free_mb']:,} MB free")
        print(f"    Driver : {gpu['driver_version']}")
        print(f"    CUDA   : {gpu['cuda_version']}")
    else:
        print("    nvidia-smi not found or no NVIDIA GPU present")

    print()
    print(SEP)
    if failures:
        print(f"  ⚠  Failed packages: {', '.join(failures)}")
        print("     Run:  pip install -r requirements.txt")
    else:
        print("  All dependencies satisfied.")
    print(SEP)
    print()

    return {
        "ok":         len(failures) == 0,
        "ort_mode":   ort_mode,
        "gpu":        gpu,
        "ort_ver":    ort_ver,
        "python_ver": sys.version.split()[0],
        "platform":   platform.platform(),
        "failures":   failures,
    }


if __name__ == "__main__":
    info = run()
    sys.exit(0 if info["ok"] else 1)
