"""Headline page: how well does each method work on a flood it has never seen?"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from . import shared


def render():
    st.title("🌊 Does a flood map transfer to a new flood?")
    st.caption(
        "Sentinel-1 radar sees through the clouds that come with floods; calm water looks dark. "
        "Every method below was trained on 10 of the 11 flood events in the Sen1Floods11 benchmark "
        "and scored on the 11th — repeated for each event. That is the situation that matters in "
        "practice: the next flood is always a new one."
    )

    table = shared.cross_event_table()
    pooled = shared.pooled_cross_event()
    missing = [m for m in shared.METHODS if m not in table.columns]
    if missing:
        st.info(f"Still running for: {', '.join(missing)}. This page updates when "
                "`scripts/cross_event.py` finishes.")

    st.subheader("Pooled over all 11 held-out floods")
    cols = st.columns(len(pooled))
    for col, (method, value) in zip(cols, pooled.sort_values(ascending=False).items()):
        col.metric(method, f"{value:.2f}", help="Water IoU over every pixel of every held-out event")

    st.subheader("Per flood (water IoU on the held-out event)")
    order = shared.separability()["vv_auc"].sort_values().index
    shown = table.loc[order]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(shown.to_numpy().T, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(shown)), shown.index, rotation=30, ha="right")
    ax.set_yticks(range(len(shown.columns)), shown.columns)
    for i in range(len(shown)):
        for j in range(len(shown.columns)):
            value = shown.iloc[i, j]
            if np.isfinite(value):
                ax.text(i, j, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="IoU")
    ax.set_title("Floods ordered from hardest (left) to easiest radar separability")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Why fixed rules break: the 'right' threshold moves")
    result = shared.cross_event()
    oracle = result["oracle_threshold_db"]
    learned = result["fold_threshold_db"]
    fig2, ax2 = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(order))
    ax2.scatter(x, [oracle[e] for e in order], color=shared.COLORS["U-Net (radar + optical)"], s=60,
                label="best threshold for that flood (oracle)", zorder=3)
    ax2.scatter(x, [learned[e] for e in order], color=shared.COLORS["U-Net (radar)"], marker="x", s=60,
                label="threshold learned from the other 10 floods", zorder=3)
    for i, e in enumerate(order):
        ax2.plot([i, i], [oracle[e], learned[e]], color="#bbb", zorder=1)
    ax2.set_xticks(x, order, rotation=30, ha="right")
    ax2.set_ylabel("VV threshold (dB)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    spread = max(oracle.values()) - min(oracle.values())
    st.caption(
        f"Each flood's own best threshold spans {spread:.1f} dB, but any threshold learned from other "
        "floods lands within about 1 dB of the same value — so floods whose water is unusually bright "
        "(Sri Lanka, Bolivia) are badly under-mapped by a fixed rule, while a per-scene method "
        "(Otsu) or a model that sees more than one band copes far better."
    )
