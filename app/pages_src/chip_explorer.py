"""Chip Explorer: any chip — radar, optical, hand label, and each method's held-out map."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from . import shared
from flood_radar_analytics import config, data
from flood_radar_analytics.evaluation import Confusion


def _rgb(s2: np.ndarray) -> np.ndarray:
    idx = config.S2_BAND_INDEX
    rgb = np.stack([s2[idx["B4"]], s2[idx["B3"]], s2[idx["B2"]]], axis=-1)
    return np.clip(rgb / 0.3, 0, 1)  # simple stretch: 30% reflectance -> white


def render():
    st.title("🔍 Chip Explorer")
    st.caption(
        "Each chip is a 512×512-pixel (≈5×5 km) scene. Predictions shown are *held-out*: made by "
        "the model trained without this chip's flood event."
    )
    names = shared.chip_names()
    c1, c2 = st.columns([1, 3])
    event = c1.selectbox("Flood event", config.EVENTS, index=config.EVENTS.index("Sri-Lanka"))
    in_event = [n for n in names if data.event_of(n) == event]
    name = c2.selectbox("Chip", in_event, format_func=lambda n: n.replace("_S1Hand.tif", ""))

    chip = shared.chip(name)
    available = {}
    for method in shared.METHODS:
        path = config.PREDICTIONS_DIR / f"{shared.method_slug(method)}.npz"
        if path.exists():
            available[method] = shared.predictions(shared.method_slug(method))[name]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(np.nan_to_num(chip.s1[0], nan=-25), -25, 0), cmap="gray")
    axes[0].set_title("Radar VV (dB) — water is dark", fontsize=10)
    axes[1].imshow(_rgb(chip.s2))
    axes[1].set_title("Optical (true colour) — often cloudy", fontsize=10)
    axes[2].imshow(np.where(chip.label == config.NODATA_LABEL, np.nan, chip.label), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title(f"Hand label — {np.mean(chip.label == 1):.0%} water", fontsize=10)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

    if not available:
        st.info("Run `python scripts/cross_event.py` to see each method's held-out prediction.")
        return
    st.subheader("Held-out predictions")
    cols = st.columns(len(available))
    for col, (method, predicted) in zip(cols, available.items()):
        score = Confusion().add(predicted, chip.label)
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(np.where(chip.label == config.NODATA_LABEL, np.nan, predicted), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{method}\nIoU {score.iou:.2f}", fontsize=8)
        ax.axis("off")
        col.pyplot(fig)
