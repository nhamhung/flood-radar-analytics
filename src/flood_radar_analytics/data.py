"""Loading Sen1Floods11 chips and turning them into model inputs.

`model_input` is the single place raw bands become a network input, so
training, cross-event evaluation and the app all see identical inputs.

Any chip file or split list that isn't on disk is fetched on demand from the
dataset's public bucket over HTTPS (no account) and cached under
`config.SEN1FLOODS11_DIR`. That's what lets the deployed app show any chip
without shipping the 2 GB dataset: it downloads only the chips a viewer opens.
"""

import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from . import config


def _ensure(path: Path, remote: str) -> Path:
    """`path`, downloading it from the public bucket first if it's missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(config.BUCKET_URL + urllib.parse.quote(remote), tmp)
        tmp.rename(path)
    return path


def split_names(split: str) -> list[str]:
    """S1 chip filenames of an official split ('train', 'valid', 'test', 'bolivia')."""
    filename = config.SPLIT_FILES[split]
    path = _ensure(config.SPLITS_DIR / filename, f"v1.1/splits/flood_handlabeled/{filename}")
    return pd.read_csv(path, header=None)[0].tolist()


def all_names() -> list[str]:
    """Every hand-labelled chip (all 446): the union of the four official splits, sorted."""
    return sorted({name for split in config.SPLIT_FILES for name in split_names(split)})


def event_of(name: str) -> str:
    """'Sri-Lanka_120804_S1Hand.tif' -> 'Sri-Lanka'."""
    return name.rsplit("_", 2)[0]


def _layer_path(name: str, layer: str):
    return config.SEN1FLOODS11_DIR / layer / name.replace("S1Hand", layer)


def _read(name: str, layer: str) -> np.ndarray:
    filename = name.replace("S1Hand", layer)
    path = _ensure(_layer_path(name, layer), f"v1.1/data/flood_events/HandLabeled/{layer}/{filename}")
    with rasterio.open(path) as src:
        return src.read()


@dataclass
class Chip:
    name: str
    s1: np.ndarray                  # (2, H, W) VV, VH in dB (NaN = no data)
    label: np.ndarray               # (H, W) -1 / 0 / 1
    permanent_water: np.ndarray     # (H, W) bool, JRC permanent water
    s2: np.ndarray | None = None    # (13, H, W) reflectance 0..1, if loaded

    @property
    def event(self) -> str:
        return event_of(self.name)


def load_chip(name: str, with_s2: bool = False) -> Chip:
    s1 = _read(name, "S1Hand").astype(np.float32)
    label = _read(name, "LabelHand")[0].astype(np.int8)
    label[~np.isfinite(s1).all(axis=0)] = config.NODATA_LABEL  # no radar -> can't classify
    permanent = _read(name, "JRCWaterHand")[0] == 1
    s2 = None
    if with_s2:
        s2 = _read(name, "S2Hand").astype(np.float32) / config.S2_SCALE
    return Chip(name=name, s1=s1, label=label, permanent_water=permanent, s2=s2)


def ndwi(s2: np.ndarray) -> np.ndarray:
    """Normalised difference water index (green - NIR) / (green + NIR)."""
    green, nir = s2[config.S2_BAND_INDEX["B3"]], s2[config.S2_BAND_INDEX["B8"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(green + nir > 0, (green - nir) / (green + nir), 0.0).astype(np.float32)


def ndvi(s2: np.ndarray) -> np.ndarray:
    """Normalised difference vegetation index (NIR - red) / (NIR + red)."""
    red, nir = s2[config.S2_BAND_INDEX["B4"]], s2[config.S2_BAND_INDEX["B8"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(red + nir > 0, (nir - red) / (nir + red), 0.0).astype(np.float32)


def raw_channels(chip: Chip, variant: str) -> np.ndarray:
    """Un-normalised channels for a variant: VV, VH, VH-VV in dB, then (for
    's1s2') the selected optical reflectances and NDWI."""
    vv = np.clip(chip.s1[0], *config.DB_CLIP["vv"])
    vh = np.clip(chip.s1[1], *config.DB_CLIP["vh"])
    channels = [vv, vh, vh - vv]
    if variant == "s1s2":
        if chip.s2 is None:
            raise ValueError("variant 's1s2' needs a chip loaded with with_s2=True")
        channels += [np.clip(chip.s2[config.S2_BAND_INDEX[b]], 0, 1) for b in config.S2_BANDS]
        channels.append(ndwi(chip.s2))
    return np.stack(channels).astype(np.float32)


def model_input(chip: Chip, variant: str) -> np.ndarray:
    """Normalised (C, H, W) float32 input; missing values become 0 (the training mean)."""
    x = raw_channels(chip, variant)
    mean = np.concatenate([config.S1_MEAN, config.S2_MEAN]) if variant == "s1s2" else config.S1_MEAN
    std = np.concatenate([config.S1_STD, config.S2_STD]) if variant == "s1s2" else config.S1_STD
    return np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
