"""Tests for metrics and baselines (synthetic data)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_radar_analytics import config, evaluation  # noqa: E402
from flood_radar_analytics.data import Chip  # noqa: E402


class TestConfusion:
    def test_perfect(self):
        label = np.array([[1, 0], [0, 1]])
        c = evaluation.Confusion().add(label == 1, label)
        assert c.iou == c.f1 == c.precision == c.recall == 1.0

    def test_nodata_and_mask_are_ignored(self):
        label = np.array([1, config.NODATA_LABEL, 0, 1])
        predicted = np.array([True, True, True, False])
        mask = np.array([True, True, True, False])
        c = evaluation.Confusion().add(predicted, label, mask)
        assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 0, 0)

    def test_iou(self):
        c = evaluation.Confusion().add(np.array([True, False, True, False]), np.array([1, 1, 0, 0]))
        assert c.iou == pytest.approx(1 / 3)


def test_evaluation_tracks_events_separately():
    e = evaluation.Evaluation()
    for event, hit in (("Spain", True), ("USA", False)):
        chip = Chip(name=f"{event}_1_S1Hand.tif", s1=np.zeros((2, 1, 1), np.float32),
                    label=np.array([[1]], np.int8), permanent_water=np.zeros((1, 1), bool))
        e.add(chip, np.array([[hit]]))
    assert e.by_event["Spain"].iou == 1.0 and e.by_event["USA"].iou == 0.0
    assert e.pooled.iou == 0.5


def test_threshold_predictor():
    chip = Chip(name="x_1_S1Hand.tif", s1=np.array([[[-20.0, -10.0, np.nan]], [[0, 0, 0]]], np.float32),
                label=np.zeros((1, 3), np.int8), permanent_water=np.zeros((1, 3), bool))
    np.testing.assert_array_equal(evaluation.threshold_predictor(-15.0)(chip), [[True, False, False]])


def test_otsu_finds_dark_cluster_and_ignores_flat_scene():
    rng = np.random.default_rng(0)
    vv = np.concatenate([rng.normal(-20, 1, 300), rng.normal(-8, 1, 700)])
    water = evaluation.otsu_water(vv)
    assert water[:300].mean() > 0.95 and water[300:].mean() < 0.05
    assert not evaluation.otsu_water(np.full(10, -12.0)).any()
