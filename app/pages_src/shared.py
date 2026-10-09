"""Cached loaders shared across every page.

The app reads only precomputed results (results/*.json, results/*.csv), the
saved held-out predictions (results/predictions/*.npz) and the raw chips —
fetched on demand from the public dataset bucket when not on disk. It never
imports PyTorch or LightGBM, so it starts fast, deploys light, and avoids
their OpenMP clash (see the package docstring).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from flood_radar_analytics import analysis, config, data  # noqa: E402

METHODS = ["VV threshold", "Authors' Otsu", "LightGBM (radar + texture)", "U-Net (radar)", "U-Net (radar + optical)"]
COLORS = {
    "VV threshold": "#8fb4f2", "Authors' Otsu": "#81b29a", "LightGBM (radar + texture)": "#f2cc8f",
    "U-Net (radar)": "#2c5cc5", "U-Net (radar + optical)": "#e07a5f",
}


def require(path: Path, how: str) -> None:
    if not path.exists():
        st.warning(f"`{path.relative_to(config.PROJECT_ROOT)}` not found — {how}")
        st.stop()


def iou(counts: dict) -> float:
    return counts["tp"] / max(counts["tp"] + counts["fp"] + counts["fn"], 1)


@st.cache_data
def benchmark() -> dict:
    path = config.RESULTS_DIR / "benchmark.json"
    require(path, "run `python scripts/train.py`.")
    return json.loads(path.read_text())


@st.cache_data
def cross_event() -> dict:
    """Leave-one-event-out results, merged from whichever stages have finished."""
    merged = config.RESULTS_DIR / "cross_event.json"
    if merged.exists():
        return json.loads(merged.read_text())
    classic = config.RESULTS_DIR / "cross_event_classic.json"
    require(classic, "run `python scripts/cross_event.py`.")
    result = json.loads(classic.read_text())
    unet = config.RESULTS_DIR / "cross_event_unet.json"
    if unet.exists():
        for event, methods in json.loads(unet.read_text())["events"].items():
            result["events"][event].update(methods)
    return result


@st.cache_data
def cross_event_table() -> pd.DataFrame:
    """Held-out IoU: one row per event, one column per method (methods present only)."""
    events = cross_event()["events"]
    table = pd.DataFrame({event: {m: iou(c) for m, c in methods.items()} for event, methods in events.items()}).T
    return table[[m for m in METHODS if m in table.columns]].loc[config.EVENTS]


@st.cache_data
def pooled_cross_event() -> pd.Series:
    events = cross_event()["events"]
    methods = [m for m in METHODS if all(m in d for d in events.values())]
    return pd.Series({m: iou({k: sum(d[m][k] for d in events.values()) for k in ("tp", "fp", "fn")})
                      for m in methods})


@st.cache_data
def separability() -> pd.DataFrame:
    path = config.RESULTS_DIR / "event_separability.csv"
    require(path, "run `python scripts/analyze.py`.")
    return pd.read_csv(path, index_col=0)


@st.cache_data
def error_anatomy() -> dict[str, dict]:
    out = {}
    for path in sorted(config.RESULTS_DIR.glob("error_anatomy_*.json")):
        out[path.stem.removeprefix("error_anatomy_")] = json.loads(path.read_text())
    return out


@st.cache_data
def predictions(slug: str) -> dict:
    return analysis.load_predictions(slug)


@st.cache_data
def chip(name: str) -> data.Chip:
    return data.load_chip(name, with_s2=True)


@st.cache_data
def chip_names() -> list[str]:
    return data.all_names()


def method_slug(method: str) -> str:
    keep = method.lower().replace("+", "plus").replace("'", "")
    return "_".join("".join(c if c.isalnum() else " " for c in keep).split())
