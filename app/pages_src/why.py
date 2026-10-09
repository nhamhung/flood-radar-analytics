"""Why floods differ: radar separability per event, and where methods fail."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import shared


def render():
    st.title("🔬 Why some floods are harder")

    sep = shared.separability()
    table = shared.cross_event_table()

    st.subheader("How visible is the water to radar?")
    st.caption(
        "For each flood: how much darker water is than land in VV (median difference, dB), and "
        "how well VV alone ranks water above land (ROC AUC — 1.0 means one threshold separates "
        "them perfectly, 0.5 means it's a coin toss)."
    )
    st.dataframe(
        sep.rename(columns={"water_share": "water share", "vv_water_median_db": "water VV (dB)",
                            "vv_land_median_db": "land VV (dB)", "vv_auc": "VV AUC",
                            "contrast_db": "contrast (dB)"}),
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.2f")
                       for c in ("water share", "water VV (dB)", "land VV (dB)", "VV AUC", "contrast (dB)")},
    )

    st.subheader("The flood matters more than the method")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in table.columns:
        ax.scatter(sep.loc[table.index, "vv_auc"], table[method], label=method,
                   color=shared.COLORS[method], s=40, alpha=0.85)
    for event in table.index:
        ax.annotate(event, (sep.loc[event, "vv_auc"], table.loc[event].max()), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("How separable water is in VV (AUC)")
    ax.set_ylabel("Held-out IoU")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    corr = {m: table[m].corr(sep.loc[table.index, "vv_auc"]) for m in table.columns}
    st.caption("Correlation between a flood's separability and each method's score on it: "
               + ", ".join(f"{m} {v:.2f}" for m, v in corr.items()) + ".")

    anatomy = shared.error_anatomy()
    if not anatomy:
        st.info("Run `python scripts/analyze.py` (after `scripts/cross_event.py`) for the error anatomy.")
        return

    st.subheader("Where do methods miss water?")
    st.caption(
        "Held-out predictions split two ways. Permanent water = rivers and lakes in the JRC Global "
        "Surface Water map; everything else labelled water is new floodwater. Vegetation (NDVI) uses "
        "clear-sky Sentinel-2 pixels only."
    )
    names = {shared.method_slug(m): m for m in shared.METHODS}
    water_type = pd.DataFrame({
        names.get(slug, slug): {r["water type"]: r["recall"] for r in a["by_water_type"]} for slug, a in anatomy.items()
    }).T
    st.markdown("**Share of water found (recall), by water type**")
    st.dataframe(water_type, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in water_type.columns})

    method = st.selectbox("Vegetation breakdown for", list(anatomy), format_func=lambda s: names.get(s, s))
    veg = pd.DataFrame(anatomy[method]["by_vegetation"]).set_index("ndvi")
    st.dataframe(veg, width="stretch",
                 column_config={"clear-sky pixels": st.column_config.NumberColumn(format="%d"),
                                **{c: st.column_config.NumberColumn(format="%.3f")
                                   for c in ("water share", "recall", "false-alarm rate")}})
