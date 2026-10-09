"""Asset-aware deployment smoke check for the flood-radar-analytics app."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flood_radar_analytics import analysis, config  # noqa: E402

METHOD_SLUGS = ["vv_threshold", "authors_otsu", "lightgbm_radar_plus_texture", "u_net_radar", "u_net_radar_plus_optical"]


def check_bundled_results() -> None:
    for name in ("benchmark.json", "cross_event.json", "event_separability.csv"):
        if not (config.RESULTS_DIR / name).exists():
            raise RuntimeError(f"results/{name} is missing.")
    cross = json.loads((config.RESULTS_DIR / "cross_event.json").read_text())
    if sorted(cross["events"]) != sorted(config.EVENTS):
        raise RuntimeError("cross_event.json does not cover all 11 flood events.")
    for slug in METHOD_SLUGS:
        if len(analysis.load_predictions(slug)) != 446:
            raise RuntimeError(f"results/predictions/{slug}.npz does not hold all 446 chips.")


def check_streamlit() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless=true", "--server.port=8501"],
        cwd=PROJECT_ROOT,
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise RuntimeError("Streamlit health endpoint did not become ready within 90 seconds.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    check_bundled_results()
    check_streamlit()
    print("flood-radar-analytics deployment smoke check passed")
