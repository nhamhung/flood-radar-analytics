"""Tests for the LightGBM pixel features (feature engineering only — no
LightGBM training here, since this test process also loads PyTorch)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_radar_analytics import pixel_model  # noqa: E402
from flood_radar_analytics.data import Chip  # noqa: E402


def _chip(vv):
    vv = np.asarray(vv, np.float32)
    return Chip(name="x_1_S1Hand.tif", s1=np.stack([vv, vv - 7]), label=np.zeros(vv.shape, np.int8),
                permanent_water=np.zeros(vv.shape, bool))


def test_one_row_per_pixel_with_all_features():
    features = pixel_model.pixel_features(_chip(np.full((20, 30), -12.0)))
    assert features.shape == (600, 3 + 2 * 2 * len(pixel_model.WINDOWS))


def test_smooth_area_has_lower_local_std_than_speckle():
    rng = np.random.default_rng(0)
    vv = np.full((40, 40), -18.0)
    vv[:, 20:] = rng.normal(-9, 3, size=(40, 20))  # speckled land on the right
    f = pixel_model.pixel_features(_chip(vv))
    std = f["vv_std_5"].to_numpy().reshape(40, 40)
    assert std[:, :10].mean() < std[:, 30:].mean()


def test_missing_values_do_not_poison_neighbourhood_stats():
    vv = np.full((10, 10), -15.0)
    vv[5, 5] = np.nan
    f = pixel_model.pixel_features(_chip(vv))
    assert np.isfinite(f.drop(columns=["vv", "vh", "ratio"]).to_numpy()).all()
