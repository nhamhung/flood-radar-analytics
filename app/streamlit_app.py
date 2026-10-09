"""Multi-page Streamlit app entry point.

Run (after scripts/download_data.py, train.py, cross_event.py, analyze.py):
    streamlit run app/streamlit_app.py
"""

import streamlit as st

from pages_src import benchmark, chip_explorer, transfer, why

st.set_page_config(page_title="Flood Radar Analytics", page_icon="🌊", layout="wide")

pages = [
    # Explicit url_path: every page module exports a same-named `render`.
    st.Page(transfer.render, title="Transfer Across Floods", icon="🌊", url_path="transfer", default=True),
    st.Page(why.render, title="Why Floods Differ", icon="🔬", url_path="why"),
    st.Page(benchmark.render, title="Official Benchmark", icon="📋", url_path="benchmark"),
    st.Page(chip_explorer.render, title="Chip Explorer", icon="🔍", url_path="chips"),
]

st.navigation(pages).run()
