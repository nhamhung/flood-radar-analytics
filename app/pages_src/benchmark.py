"""Benchmark page: the official Sen1Floods11 train / validation / test split."""

import pandas as pd
import streamlit as st

from . import shared

SPLITS = {"valid": "Validation", "test": "Test", "bolivia": "Bolivia (held out)"}


def render():
    st.title("📋 The official benchmark split")
    st.caption(
        "Sen1Floods11's own split: train on 252 chips, choose settings on 89 validation chips, "
        "report on 90 test chips — all three drawn from the same 10 flood events — plus 15 chips "
        "of Bolivia, an event kept out entirely. Scores here are higher than in the cross-flood "
        "test because test chips come from floods the model has already seen."
    )
    result = shared.benchmark()
    rows = []
    for split, label in SPLITS.items():
        for method, scores in result["splits"][split].items():
            rows.append({"method": method, "split": label, **{k: scores["pooled"][k]
                                                              for k in ("IoU", "precision", "recall")}})
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="method", columns="split", values="IoU").loc[
        [m for m in shared.METHODS if m in set(table["method"])], list(SPLITS.values())]
    st.dataframe(pivot, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in pivot.columns})

    st.subheader("Precision and recall on the test split")
    test = table[table["split"] == "Test"].set_index("method").loc[pivot.index, ["precision", "recall"]]
    st.dataframe(test, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in test.columns})
    st.caption(f"VV threshold learned on the training split: {result['vv_threshold_db']:.2f} dB.")

    if result.get("history"):
        st.subheader("U-Net training curves (validation IoU)")
        curves = {variant: [h["valid_iou"] for h in hist] for variant, hist in result["history"].items() if hist}
        if curves:
            st.line_chart(pd.DataFrame(curves).rename(columns={"s1": "radar", "s1s2": "radar + optical"}))
