"""Paths, constants, and data conventions.

Data: Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020), hand-labelled
part — 446 chips of 512x512 pixels (~10 m) from 11 flood events on six
continents, public at gs://sen1floods11/v1.1/. Per chip (verified directly):

- S1Hand: Sentinel-1 radar, 2 float32 bands (VV, VH), sigma0 in dB.
- S2Hand: Sentinel-2 optical, 13 int16 bands (B1..B12), reflectance x 10,000,
  same date window — often partly cloudy, since floods come with storms.
- LabelHand: hand label, -1 no data, 0 not water, 1 water.
- JRCWaterHand: JRC Global Surface Water permanent-water mask (1 = water
  in normal conditions) — separates rivers/lakes from new floodwater.
- S1OtsuLabelHand: the dataset authors' own Otsu-threshold water map.
"""

import os
from pathlib import Path

import numpy as np

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Raw data location can be overridden (e.g. a scratch disk, or an empty
# directory to test the on-demand download path the deployed app relies on).
DATA_RAW_DIR = Path(os.environ.get("FLOOD_RADAR_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"  # small committed result tables (JSON/CSV) the app and report read

SEN1FLOODS11_DIR = DATA_RAW_DIR / "sen1floods11"
BUCKET_URL = "https://storage.googleapis.com/sen1floods11/"  # public, anonymous HTTPS
# Held-out (leave-one-event-out) predictions, bit-packed — small (~6 MB) and
# committed with the results, since the app's Chip Explorer shows them.
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
LAYERS = ["S1Hand", "S2Hand", "LabelHand", "JRCWaterHand", "S1OtsuLabelHand"]
SPLITS_DIR = SEN1FLOODS11_DIR / "splits"
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",  # a whole event the official split holds out
}

# The 11 flood events (chip filenames start with the event name).
EVENTS = ["Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay",
          "Somalia", "Spain", "Sri-Lanka", "USA"]

# --- Data conventions ------------------------------------------------------

RANDOM_SEED = 42
CHIP_SIZE = 512
NODATA_LABEL = -1

# Radar normalisation (VV, VH, VH-VV in dB), computed once from all 57.3M
# labelled pixels of the official training split after clipping —
# accumulated in float64 (float32 summation over that many pixels drifts).
DB_CLIP = {"vv": (-30.0, 5.0), "vh": (-40.0, -5.0)}
S1_MEAN = np.array([-10.45, -17.30, -6.85], dtype=np.float32)
S1_STD = np.array([4.04, 4.69, 3.39], dtype=np.float32)

# Optical bands used by the radar+optical model: blue, green, red, near-IR,
# and the two short-wave IR bands (water absorbs NIR/SWIR strongly), plus
# NDWI = (green - NIR) / (green + NIR), the standard optical water index.
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S2_BAND_INDEX = {f"B{b}": i for i, b in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "8A", "9", "10", "11", "12"])}
S2_SCALE = 10_000.0  # stored reflectance x 10,000
# Normalisation for (B2, B3, B4, B8, B11, B12 clipped to 0..1, NDWI), from
# the same 57.3M training pixels (float64). Only 0.02% of labelled pixels
# have no optical data at all — but many are cloud, which the model must learn.
S2_MEAN = np.array([0.1253, 0.1228, 0.1062, 0.2558, 0.1935, 0.1070, -0.3225], dtype=np.float32)
S2_STD = np.array([0.0352, 0.0374, 0.0532, 0.0851, 0.0808, 0.0603, 0.2080], dtype=np.float32)

# Model input variants.
INPUT_CHANNELS = {
    "s1": ["vv", "vh", "ratio"],
    "s1s2": ["vv", "vh", "ratio"] + S2_BANDS + ["ndwi"],
}

# Water is 9.5% of labelled training pixels — see BceDiceLoss in model.py.
TRAIN_WATER_SHARE = 0.095


def model_path(name: str) -> Path:
    return MODELS_DIR / name
