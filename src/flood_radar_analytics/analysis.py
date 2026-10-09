"""The analyses behind the "why": how separable water is, and where methods fail.

- `event_separability`: per flood event — how much water there is, how dark
  water and land are in VV, and how well VV alone ranks water above land
  (ROC AUC; 1.0 = a perfect threshold exists, 0.5 = no better than chance).
- `error_anatomy`: for a method's held-out predictions — recall on permanent
  water (rivers, lakes: JRC mask) vs. new floodwater, and how errors depend
  on vegetation (NDVI from clear-sky Sentinel-2 pixels).

Uses no PyTorch or LightGBM — only saved predictions (see scripts/cross_event.py).
"""

import numpy as np
import pandas as pd

from . import config
from .data import all_names, event_of, load_chip, ndvi

PREDICTIONS_DIR = config.PREDICTIONS_DIR

# A pixel counts as clear-sky optical when its blue reflectance is below this:
# clouds are bright in blue, land and water are dark. A simple, conservative
# screen — it discards some bright surfaces too, which only costs sample size.
CLEAR_SKY_MAX_BLUE = 0.15
NDVI_BINS = [-1.0, 0.2, 0.4, 0.6, 1.0]
NDVI_LABELS = ["bare / water (< 0.2)", "sparse (0.2-0.4)", "moderate (0.4-0.6)", "dense (> 0.6)"]


def event_separability(sample_per_chip: int = 20_000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score  # imported here so the app doesn't need scikit-learn

    rng = np.random.default_rng(seed)
    rows = {}
    for name in all_names():
        chip = load_chip(name)
        valid = np.flatnonzero(chip.label.ravel() != config.NODATA_LABEL)
        if len(valid) == 0:
            continue
        take = rng.choice(valid, size=min(sample_per_chip, len(valid)), replace=False)
        entry = rows.setdefault(chip.event, {"vv": [], "water": [], "chips": 0, "labelled": 0, "water_px": 0})
        entry["vv"].append(chip.s1[0].ravel()[take])
        entry["water"].append(chip.label.ravel()[take] == 1)
        entry["chips"] += 1
        entry["labelled"] += len(valid)
        entry["water_px"] += int((chip.label == 1).sum())
    out = []
    for event, e in rows.items():
        vv, water = np.concatenate(e["vv"]), np.concatenate(e["water"])
        out.append({
            "event": event,
            "chips": e["chips"],
            "water_share": e["water_px"] / e["labelled"],
            "vv_water_median_db": float(np.median(vv[water])) if water.any() else np.nan,
            "vv_land_median_db": float(np.median(vv[~water])),
            "vv_auc": float(roc_auc_score(water, -vv)) if 0 < water.mean() < 1 else np.nan,
        })
    table = pd.DataFrame(out).set_index("event").loc[config.EVENTS]
    table["contrast_db"] = table["vv_land_median_db"] - table["vv_water_median_db"]
    return table


def load_predictions(method_slug: str) -> dict[str, np.ndarray]:
    path = PREDICTIONS_DIR / f"{method_slug}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/cross_event.py` first.")
    with np.load(path) as saved:
        return {name: np.unpackbits(saved[name])[: config.CHIP_SIZE**2].reshape(config.CHIP_SIZE, config.CHIP_SIZE)
                .astype(bool) for name in saved.files}


def error_anatomy(method_slug: str) -> dict[str, pd.DataFrame]:
    """Held-out (leave-one-event-out) errors of one method, broken down two ways."""
    predictions = load_predictions(method_slug)
    by_water_type: dict[str, dict[str, int]] = {}
    by_ndvi = {label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in NDVI_LABELS}
    for name, predicted in predictions.items():
        chip = load_chip(name, with_s2=True)
        valid = chip.label != config.NODATA_LABEL
        water = chip.label == 1
        event = event_of(name)
        for kind, mask in (("permanent water", chip.permanent_water), ("new floodwater", ~chip.permanent_water)):
            counts = by_water_type.setdefault(f"{event}|{kind}", {"water_px": 0, "found_px": 0})
            counts["water_px"] += int((valid & water & mask).sum())
            counts["found_px"] += int((valid & water & mask & predicted).sum())
        clear = valid & (chip.s2[config.S2_BAND_INDEX["B2"]] < CLEAR_SKY_MAX_BLUE)
        bins = np.digitize(ndvi(chip.s2), NDVI_BINS[1:-1])
        for i, label in enumerate(NDVI_LABELS):
            m = clear & (bins == i)
            by_ndvi[label]["tp"] += int((m & water & predicted).sum())
            by_ndvi[label]["fp"] += int((m & ~water & predicted).sum())
            by_ndvi[label]["fn"] += int((m & water & ~predicted).sum())
            by_ndvi[label]["tn"] += int((m & ~water & ~predicted).sum())

    water_rows = []
    for key, c in by_water_type.items():
        event, kind = key.split("|")
        water_rows.append({"event": event, "water type": kind, "water pixels": c["water_px"],
                           "recall": c["found_px"] / c["water_px"] if c["water_px"] else np.nan})
    water_table = pd.DataFrame(water_rows)
    totals = (water_table.assign(found=water_table["recall"].fillna(0) * water_table["water pixels"])
              .groupby("water type")[["water pixels", "found"]].sum())
    totals["recall"] = totals["found"] / totals["water pixels"]

    ndvi_table = pd.DataFrame(by_ndvi).T
    ndvi_table["clear-sky pixels"] = ndvi_table.sum(axis=1)
    ndvi_table["water share"] = (ndvi_table["tp"] + ndvi_table["fn"]) / ndvi_table["clear-sky pixels"]
    ndvi_table["recall"] = ndvi_table["tp"] / (ndvi_table["tp"] + ndvi_table["fn"]).replace(0, np.nan)
    ndvi_table["false-alarm rate"] = ndvi_table["fp"] / (ndvi_table["fp"] + ndvi_table["tn"]).replace(0, np.nan)
    return {
        "by_event_and_water_type": water_table,
        "by_water_type": totals[["water pixels", "recall"]],
        "by_vegetation": ndvi_table[["clear-sky pixels", "water share", "recall", "false-alarm rate"]],
    }
