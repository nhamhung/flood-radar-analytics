"""Tests for the U-Net helpers (synthetic data; PyTorch only — never LightGBM)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_radar_analytics import config, unet  # noqa: E402


def test_loss_ignores_nodata_and_rewards_correct_logits():
    loss = unet.BceDiceLoss()
    target = torch.tensor([1.0, 0.0, config.NODATA_LABEL])
    assert loss(torch.tensor([5.0, -5.0, 99.0]), target) < loss(torch.tensor([-5.0, 5.0, -99.0]), target)
    assert loss(torch.zeros(2), torch.full((2,), float(config.NODATA_LABEL))).item() == 0.0


class _ConstantNet(torch.nn.Module):
    def forward(self, x):
        return x[:, :1] * 0 + 2.0


def test_tiled_prediction_covers_any_shape():
    prob = unet.predict_probability(_ConstantNet(), np.zeros((3, 300, 700), np.float32),
                                    dev=torch.device("cpu"), tile=256)
    assert prob.shape == (300, 700)
    np.testing.assert_allclose(prob, 1 / (1 + np.exp(-2.0)), rtol=1e-5)


@pytest.mark.parametrize("variant", sorted(config.INPUT_CHANNELS))
def test_unet_shapes(variant):
    net = unet.build_unet(variant, pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.zeros(1, len(config.INPUT_CHANNELS[variant]), 64, 64))
    assert out.shape == (1, 1, 64, 64)
