"""LightGBM pixel classifier: radar values + local texture, one pixel at a time.

Sits between the threshold and the U-Net in complexity. It sees more than a
single dB value — each pixel's neighbourhood mean and standard deviation at
two scales, which capture how smooth (water) or speckled (land) the area
around it is — but none of the learned spatial shapes a U-Net uses.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from . import config
from .data import Chip, load_chip

WINDOWS = (5, 15)  # neighbourhood sizes in pixels (50 m and 150 m)


def _local_stats(band: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    filled = np.where(np.isfinite(band), band, np.nanmean(band) if np.isfinite(band).any() else 0.0)
    mean = uniform_filter(filled, size=size, mode="reflect")
    mean_sq = uniform_filter(filled * filled, size=size, mode="reflect")
    return mean, np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def pixel_features(chip: Chip) -> pd.DataFrame:
    """One row per pixel, in row-major order."""
    vv = np.clip(chip.s1[0], *config.DB_CLIP["vv"])
    vh = np.clip(chip.s1[1], *config.DB_CLIP["vh"])
    columns = {"vv": vv, "vh": vh, "ratio": vh - vv}
    for name, band in (("vv", vv), ("vh", vh)):
        for size in WINDOWS:
            mean, std = _local_stats(band, size)
            columns[f"{name}_mean_{size}"] = mean
            columns[f"{name}_std_{size}"] = std
    return pd.DataFrame({k: v.ravel().astype(np.float32) for k, v in columns.items()})


def training_table(names: list[str], pixels_per_chip: int = 20_000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """A random sample of labelled pixels from each chip (all chips weighted
    equally, so one large, easy chip can't dominate)."""
    rng = np.random.default_rng(seed)
    frames = []
    for name in names:
        chip = load_chip(name)
        features = pixel_features(chip)
        label = chip.label.ravel()
        valid = np.flatnonzero(label != config.NODATA_LABEL)
        if len(valid) == 0:
            continue
        take = rng.choice(valid, size=min(pixels_per_chip, len(valid)), replace=False)
        frame = features.iloc[take].copy()
        frame["water"] = (label[take] == 1).astype(np.int8)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def train(names: list[str], pixels_per_chip: int = 20_000):
    from lightgbm import LGBMClassifier

    table = training_table(names, pixels_per_chip)
    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=config.RANDOM_SEED, verbosity=-1)
    model.fit(table.drop(columns="water"), table["water"])
    return model


def predictor(model, threshold: float = 0.5):
    def predict(chip: Chip) -> np.ndarray:
        prob = model.predict_proba(pixel_features(chip))[:, 1]
        return (prob >= threshold).reshape(chip.label.shape) & np.isfinite(chip.s1).all(axis=0)

    return predict
