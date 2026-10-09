"""Tests for chip handling and model inputs (synthetic data — no download needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_radar_analytics import config, data  # noqa: E402


def _chip(vv=-10.0, vh=-17.0, with_s2=True, size=4):
    s1 = np.stack([np.full((size, size), vv), np.full((size, size), vh)]).astype(np.float32)
    s2 = np.full((13, size, size), 0.1, dtype=np.float32) if with_s2 else None
    return data.Chip(name="Spain_1_S1Hand.tif", s1=s1, label=np.zeros((size, size), np.int8),
                     permanent_water=np.zeros((size, size), bool), s2=s2)


def test_event_is_parsed_from_filename():
    assert data.event_of("Sri-Lanka_120804_S1Hand.tif") == "Sri-Lanka"
    assert data.event_of("USA_430764_S1Hand.tif") == "USA"


@pytest.mark.parametrize("variant", sorted(config.INPUT_CHANNELS))
def test_model_input_shape_per_variant(variant):
    x = data.model_input(_chip(), variant)
    assert x.shape == (len(config.INPUT_CHANNELS[variant]), 4, 4)
    assert x.dtype == np.float32 and np.isfinite(x).all()


def test_training_mean_maps_to_zero():
    chip = _chip(vv=float(config.S1_MEAN[0]), vh=float(config.S1_MEAN[1]))
    np.testing.assert_allclose(data.model_input(chip, "s1")[:2], 0.0, atol=1e-6)


def test_ratio_channel_is_vh_minus_vv():
    raw = data.raw_channels(_chip(vv=-10.0, vh=-18.0), "s1")
    assert raw[2, 0, 0] == pytest.approx(-8.0)


def test_missing_radar_becomes_zero_not_nan():
    chip = _chip()
    chip.s1[0, 0, 0] = np.nan
    assert np.isfinite(data.model_input(chip, "s1")).all()


def test_s1s2_needs_optical():
    with pytest.raises(ValueError, match="with_s2"):
        data.model_input(_chip(with_s2=False), "s1s2")


def test_water_and_vegetation_indices():
    s2 = np.zeros((13, 1, 2), dtype=np.float32)
    g, r, n = config.S2_BAND_INDEX["B3"], config.S2_BAND_INDEX["B4"], config.S2_BAND_INDEX["B8"]
    s2[g] = [[0.10, 0.05]]
    s2[r] = [[0.05, 0.05]]
    s2[n] = [[0.02, 0.40]]
    assert data.ndwi(s2)[0, 0] > 0 > data.ndwi(s2)[0, 1]   # water-like, then vegetation-like
    assert data.ndvi(s2)[0, 1] > 0.7
