#!/usr/bin/env python
"""One-click launcher: open the notebooks (JupyterLab) or the Streamlit app in your browser.

You normally don't run this yourself. Double-click `start_windows.bat` (Windows)
or `start_mac.command` (macOS): they install uv if needed, let uv set up the
exact Python and packages from `uv.lock`, then run this script.

Usage (from the project folder):
    uv run python scripts/launch.py            # menu
    uv run python scripts/launch.py notebooks  # straight to JupyterLab
    uv run python scripts/launch.py app        # straight to the Streamlit app
    uv run python scripts/launch.py check      # only check the setup, then exit
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "Flood Radar Analytics"
APP = PROJECT_ROOT / "app" / "streamlit_app.py"
NOTEBOOKS = PROJECT_ROOT / "notebooks"


# --- Project-specific: data the notebooks need ------------------------------

# Same location as src/flood_radar_analytics/config.py (SEN1FLOODS11_DIR).
DATASET = Path(os.environ.get("FLOOD_RADAR_DATA_DIR", PROJECT_ROOT / "data" / "raw")) / "sen1floods11"
LAYERS = ["S1Hand", "S2Hand", "LabelHand", "JRCWaterHand", "S1OtsuLabelHand"]
CHIPS_PER_LAYER = 446
SPLIT_FILES = ["flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv", "flood_bolivia_data.csv"]
# The ImageNet-pretrained ResNet-34 encoder that notebook 02 starts its U-Net from,
# as segmentation-models-pytorch 0.5.0 fetches it from the Hugging Face hub.
WEIGHTS_REPO, WEIGHTS_REVISION = "smp-hub/resnet34.imagenet", "7a57b34f723329ff020b3f8bc41771163c519d0c"


def missing_data() -> list[str]:
    """What is missing from the Sen1Floods11 hand-labelled set (empty list = complete)."""
    missing = [f"splits/{name}" for name in SPLIT_FILES if not (DATASET / "splits" / name).exists()]
    for layer in LAYERS:
        found = len(list((DATASET / layer).glob("*.tif")))
        if found < CHIPS_PER_LAYER:
            missing.append(f"{layer}/ ({found} of {CHIPS_PER_LAYER} chips)")
    return missing


def _weights_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(WEIGHTS_REPO, name, revision=WEIGHTS_REVISION), str)
               for name in ("config.json", "model.safetensors"))


def ensure_pretrained_weights() -> None:
    """Download notebook 02's pretrained encoder weights once (~85 MB), while online.

    The notebook itself works offline: it uses these weights if they are cached,
    and otherwise trains from random starting weights (slower learning).
    """
    if importlib.util.find_spec("torch") is None or _weights_cached():
        return  # no PyTorch on this computer (Intel Mac), or already downloaded
    print("\nDownloading the pretrained U-Net weights for notebook 02 (about 85 MB, one time only)...", flush=True)
    code = ("import segmentation_models_pytorch as smp; "
            "smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1)")
    env = {**os.environ, "HF_HUB_VERBOSITY": "error"}  # hide the hub's "set a HF_TOKEN" hint: no account is needed
    if subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env).returncode or not _weights_cached():
        print("Could not download them (no internet?). Notebook 02 still runs, from random starting weights.")


def ensure_notebook_data() -> bool:
    """Both notebooks read the Sen1Floods11 chips in data/raw/sen1floods11 (~1.8 GB).

    If they are missing, offer to download them with scripts/download_data.py:
    public data over plain HTTPS, no account needed, roughly 5-30 minutes
    depending on the connection. Files already present are skipped, so an
    interrupted download can simply be started again.
    """
    missing = missing_data()
    if missing:
        print("\nThe notebooks need the Sen1Floods11 satellite images, and these are missing or incomplete")
        print(f"in {DATASET}:")
        for item in missing:
            print(f"  - {item}")
        print("\nThey can be downloaded from the dataset's free public storage (no account needed).")
        print("This downloads about 1.8 GB and takes roughly 5-30 minutes, depending on your internet.")
        print("If it is interrupted, choose this option again: it continues where it stopped.")
        if input("Download them now? [y/N] ").strip().lower() != "y":
            print("Skipped. The app still works; the notebooks need these files.")
            return False
        print("\n>>> Running scripts/download_data.py (it prints a line after each of 5 image layers) ...")
        if subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "download_data.py")],
                          cwd=PROJECT_ROOT).returncode or missing_data():
            print("The download did not finish; see the messages above. Choose this option again to resume.")
            return False
    ensure_pretrained_weights()
    return True


# --- macOS: the OpenMP runtime LightGBM needs --------------------------------
# LightGBM's macOS wheels expect Homebrew's libomp (`brew install libomp`), which
# most Macs don't have. If it can't load, fetch libomp from conda-forge (~300 kB,
# no account, checksum-pinned) into .libomp/ and point the app and notebooks at it.

LIBOMP_DIR = PROJECT_ROOT / ".libomp"
LIBOMP_PACKAGES = {  # platform.machine() -> (conda-forge file, sha256)
    "arm64": ("osx-arm64/llvm-openmp-23.1.2-hdb3d66b_0.conda",
              "f9aea67117079c00b9663ee0bbca7e5409dbe94625c34e68881ce2c97c3d47c9"),
    "x86_64": ("osx-64/llvm-openmp-23.1.2-h8c1e6b9_0.conda",
               "3c17127e2a94d60a8babef57c66d643374c5b08c896ec5f0987a62ecfb81c8ee"),
}


def _openmp_loads() -> bool:
    probe = [sys.executable, "-c", "import lightgbm"]
    return subprocess.run(probe, env=_env(), capture_output=True).returncode == 0


def ensure_openmp(force: bool = False) -> bool:
    """True if LightGBM loads; on a Mac where it can't, download libomp once first."""
    if sys.platform != "darwin" or (not force and _openmp_loads()):
        return True
    import hashlib
    import io
    import platform
    import ssl
    import tarfile
    import zipfile

    import certifi
    from backports import zstd

    print("\nOne-time fix: downloading the OpenMP runtime that LightGBM needs on macOS (~300 kB)...")
    name, sha256 = LIBOMP_PACKAGES[platform.machine()]
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        url = f"https://conda.anaconda.org/conda-forge/{name}"
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            package = response.read()
        if hashlib.sha256(package).hexdigest() != sha256:
            raise OSError("checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(package)) as conda:
            inner = next(n for n in conda.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
            tar_bytes = zstd.decompress(conda.read(inner))
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            dylib = tar.extractfile("lib/libomp.dylib").read()
        LIBOMP_DIR.mkdir(exist_ok=True)
        (LIBOMP_DIR / "libomp.dylib").write_bytes(dylib)
    except (OSError, KeyError, StopIteration, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"That didn't work ({error}). The model needs it.")
        print("Fix: install Homebrew (https://brew.sh), run `brew install libomp`, then start again.")
        return False
    ok = _openmp_loads()
    print("Done." if ok else "LightGBM still can't load; see START_HERE.md, 'If something goes wrong'.")
    return ok


# --- Shared ------------------------------------------------------------------


def _free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found.")


def _open_when_ready(url: str, health: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(1)
    print(f"\nThe app is taking a while to start. Open {url} in your browser when it's ready.")


def _env() -> dict:
    env = os.environ.copy()
    # Keep Jupyter's kernels and settings inside the project, so an older
    # Jupyter install elsewhere on the computer can't take over the kernel.
    env["JUPYTER_DATA_DIR"] = str(PROJECT_ROOT / ".jupyter_data")
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    if sys.platform == "darwin" and (LIBOMP_DIR / "libomp.dylib").exists():
        # Used only if Homebrew's libomp isn't there (see ensure_openmp).
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{LIBOMP_DIR}:/usr/local/lib:/usr/lib"
    return env


def run_notebooks() -> None:
    ensure_openmp()
    if not ensure_notebook_data():
        return
    print("\nOpening JupyterLab in your browser...")
    print("Open a notebook from the list on the left, then use Run > Run All Cells.")
    print("To stop: close the browser tab, then press Ctrl+C in this window.\n")
    port = _free_port(8888)
    subprocess.run([sys.executable, "-m", "jupyter", "lab", f"--notebook-dir={NOTEBOOKS}", f"--port={port}"],
                   cwd=PROJECT_ROOT, env=_env())


def run_app() -> None:
    ensure_openmp()
    port = _free_port(8501)
    url = f"http://localhost:{port}"
    print(f"\nStarting the app at {url} (your browser opens by itself)...")
    print("To stop: press Ctrl+C in this window.\n")
    threading.Thread(target=_open_when_ready, args=(url, f"{url}/_stcore/health"), daemon=True).start()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless=true",
                    f"--server.port={port}"], cwd=PROJECT_ROOT, env=_env())


def check() -> None:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    for name in ("streamlit", "jupyterlab", "ipykernel"):
        __import__(name)
        print(f"  {name}: ok")
    if sys.platform == "darwin":
        print("OpenMP for LightGBM:", "ok" if ensure_openmp() else "missing")
    missing = missing_data()
    print("Notebook data:", "ok" if not missing else f"missing or incomplete: {', '.join(missing)} "
          "(the launcher can download it)")
    if importlib.util.find_spec("torch") is None:
        print("PyTorch: not available on this computer (Intel Mac), so notebook 02 can't run here")
    else:
        print("Pretrained U-Net weights:", "ok" if _weights_cached() else
              "not downloaded yet (the launcher downloads them when you open the notebooks)")
    print("Setup looks good.")


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    actions = {"notebooks": run_notebooks, "app": run_app, "check": check}
    if choice:
        actions[choice]()
        return
    while True:
        print(f"\n=== {PROJECT_NAME} ===")
        print("  1  Open the notebooks (JupyterLab)")
        print("  2  Open the app (Streamlit)")
        print("  3  Quit")
        picked = input("Type 1, 2 or 3 and press Enter: ").strip()
        try:
            if picked == "1":
                run_notebooks()
            elif picked == "2":
                run_app()
            elif picked == "3":
                return
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
