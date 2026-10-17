# Flood Radar Analytics — Does a Flood Map Transfer to a New Flood?

> **New to Python, or just want to try it?** See [START_HERE.md](START_HERE.md): download the ZIP,
> double-click `start_windows.bat` (Windows) or `start_mac.command` (macOS), and it sets everything up.

Floods arrive with storm clouds that blind ordinary satellites. **Sentinel-1
radar** sees through them, and calm water shows up dark — which makes radar
the workhorse of flood mapping. But every flood is different, and a method
tuned on past floods is always applied to a new one.

This project compares five water-mapping methods, from a single threshold to
a U-Net that combines radar with optical imagery, on the
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) benchmark —
446 hand-labelled chips from **11 floods on six continents** — and asks how
much each one degrades on a flood it has never seen (leave-one-event-out).

## Findings

| Method | Familiar floods (official test split) | New floods (pooled, leave-one-event-out) |
|---|---|---|
| VV threshold | 0.464 | 0.434 |
| Authors' Otsu (per-scene) | 0.555 | 0.495 |
| LightGBM (radar + texture) | 0.637 | 0.566 |
| U-Net (radar) | 0.672 | 0.587 |
| **U-Net (radar + optical)** | **0.830** | **0.775** |

*Water IoU. "New floods": each of the 11 floods scored by models trained on the other 10.*

- **The radar + optical U-Net is the best method on all 11 unseen floods**, and
  loses little going from familiar to new floods (0.83 → 0.78). Its biggest
  gains are where radar struggles most — Somalia, where water is only 2.6 dB
  darker than land: radar-only best 0.27, radar + optical 0.70.
- **The "right" radar threshold moves by 5.8 dB between floods** (−18.25 to
  −12.50 dB), but a threshold learned from other floods is always ≈ −16.5 dB.
  Where a flood's water is unusually bright, fixed rules collapse: in Sri
  Lanka a learned threshold scores IoU 0.16 while per-scene Otsu scores 0.65.
- **For radar alone, the flood matters more than the method.** A radar-only
  method's IoU varies by 0.61 on average across floods, but methods differ by
  only 0.19 on the same flood.
- **Misses are new floodwater, not rivers.** Every method finds 75–99% of
  permanent water (rivers, lakes) but only 45–81% of new floodwater; the
  radar + optical U-Net finds 81%, the radar U-Net 62%.

## What's here

| Deliverable | Where |
|---|---|
| Streamlit app: transfer across floods, why floods differ, the official benchmark, a chip explorer | `app/streamlit_app.py` + `app/pages_src/` |
| Research-style writeup (numbers computed from `results/` at render time) | `report/report.qmd` |
| Step-by-step notebook guide for beginners: data, inputs, scoring, VV threshold, Otsu, LightGBM, leave-one-event-out, the analyses (never imports PyTorch) | `notebooks/01_flood_radar_analytics.ipynb` |
| U-Net walkthrough: architecture, inputs, loss, augmentation, a small demo training on the GPU, tiled inference, full-run results (never imports LightGBM) | `notebooks/02_unet_walkthrough.ipynb` |
| Scripts: download, train, leave-one-event-out, analysis | `scripts/` |
| Result tables (small, committed) | `results/` |

All share `src/flood_radar_analytics/`:

| Module | Contents |
|---|---|
| `config.py` | paths, the 11 events, normalisation constants (computed from the training split) |
| `data.py` | loading chips (radar, optical, labels, permanent water), model inputs |
| `evaluation.py` | IoU metrics, per-event evaluation, threshold and Otsu baselines |
| `pixel_model.py` | the LightGBM pixel classifier and its texture features |
| `unet.py` | the U-Net: training, tiled inference, persistence |
| `analysis.py` | per-event separability, error anatomy (water type, vegetation) |

> **LightGBM and PyTorch never share a Python process** — on macOS each ships
> its own OpenMP runtime, and together they segfault or deadlock depending on
> import order. `data`, `evaluation` and `analysis` import neither; the scripts
> run LightGBM and U-Net stages as separate processes; the app imports neither.

## Setup and run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py     # ~2 GB over plain HTTPS from the public bucket (no account)
python scripts/train.py             # official split, all five methods (~30 min, Apple-silicon GPU)
python scripts/cross_event.py       # leave-one-event-out, 11 floods (~2.5 h)
python scripts/analyze.py           # separability + error anatomy tables
streamlit run app/streamlit_app.py
pytest tests/                       # 21 tests, synthetic data, no download needed
```

Render the report:

```bash
python -m ipykernel install --user --name flood-radar-analytics --display-name "Flood Radar Analytics"
QUARTO_PYTHON=.venv/bin/python quarto render report/report.qmd
```

## Deploy

The Streamlit app needs no secrets, model weights or local data: it reads the bundled results in `results/`, and the Chip Explorer downloads only the chips a viewer opens from the public Sen1Floods11 bucket. It installs from the minimal, pinned `app/requirements.txt` (streamlit, matplotlib, rasterio).

- Repository: <https://github.com/nhamhung/flood-radar-analytics>
- Report: <https://nhamhung.github.io/flood-radar-analytics/>
- Beginner's guide: <https://nhamhung.github.io/flood-radar-analytics/beginners-guide.html>
- Streamlit: <https://flood-radar-analytics-bg6uugifczndcbtsrps6cd.streamlit.app/>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)

## Data and credits

**Sen1Floods11** — Bonafilia, Tellman, Anderson & Issenberg, *Sen1Floods11: A
Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for
Sentinel-1*, CVPR Workshops 2020. Public at `gs://sen1floods11/v1.1/`
(Sentinel-1/2 data © ESA Copernicus; permanent water from the JRC Global
Surface Water dataset).
