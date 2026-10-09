"""Metrics, evaluation, and the non-learned baselines — no PyTorch, no LightGBM.

Methods compared in this project, simplest to most complex (each is a
function chip -> water mask):

- **VV threshold** — water where VV backscatter is below one dB value learned
  on training chips. The simplest possible radar water detector.
- **Authors' Otsu** — the Sen1Floods11 authors' own scene-level Otsu maps
  (Otsu picks, per scene, the threshold that best splits dark and bright
  pixels). Reported as published; not re-fit here.
- **LightGBM pixel classifier** (`pixel_model.py`) — radar values plus local
  texture statistics, one pixel at a time.
- **U-Net (radar)** and **U-Net (radar + optical)** (`unet.py`) — a
  convolutional segmentation network that sees spatial context.

Metric: water intersection-over-union (IoU), pooled over all valid pixels of
a set of chips — the convention of the Sen1Floods11 paper.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import config
from .data import Chip, load_chip

Predictor = Callable[[Chip], np.ndarray]


# --- Metrics -----------------------------------------------------------------


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted: np.ndarray, label: np.ndarray, mask: np.ndarray | None = None) -> "Confusion":
        valid = label != config.NODATA_LABEL
        if mask is not None:
            valid &= mask
        p, t = predicted[valid].astype(bool), label[valid].astype(bool)
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())
        self.tn += int((~p & ~t).sum())
        return self

    @property
    def iou(self) -> float:
        return self.tp / max(self.tp + self.fp + self.fn, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        return 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)

    def as_dict(self) -> dict:
        return {"IoU": self.iou, "F1": self.f1, "precision": self.precision, "recall": self.recall,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


@dataclass
class Evaluation:
    """Pooled and per-event confusion of one method over a set of chips."""

    pooled: Confusion = field(default_factory=Confusion)
    by_event: dict[str, Confusion] = field(default_factory=dict)

    def add(self, chip: Chip, predicted: np.ndarray) -> None:
        self.pooled.add(predicted, chip.label)
        self.by_event.setdefault(chip.event, Confusion()).add(predicted, chip.label)


def evaluate(predictors: dict[str, Predictor], names: list[str], with_s2: bool = False) -> dict[str, Evaluation]:
    out = {name: Evaluation() for name in predictors}
    for chip_name in names:
        chip = load_chip(chip_name, with_s2=with_s2)
        for method, predict in predictors.items():
            out[method].add(chip, predict(chip))
    return out


# --- Baselines -------------------------------------------------------------


def fit_threshold(names: list[str], candidates=np.arange(-25.0, -10.0, 0.25)) -> float:
    """The VV dB threshold maximising pooled water IoU on `names`."""
    values, water = [], []
    for name in names:
        chip = load_chip(name)
        valid = chip.label != config.NODATA_LABEL
        values.append(chip.s1[0][valid])
        water.append(chip.label[valid] == 1)
    vv, water = np.concatenate(values), np.concatenate(water)
    order = np.argsort(vv)
    vv_sorted, water_sorted = vv[order], water[order]
    cum_water = np.cumsum(water_sorted)
    total_water = water_sorted.sum()
    best, best_iou = float(candidates[0]), -1.0
    for t in candidates:
        k = np.searchsorted(vv_sorted, t)  # pixels predicted water: vv < t
        tp = cum_water[k - 1] if k else 0
        iou = tp / max(k + total_water - tp, 1)
        if iou > best_iou:
            best, best_iou = float(t), iou
    return best


def threshold_predictor(threshold: float) -> Predictor:
    return lambda chip: np.isfinite(chip.s1[0]) & (chip.s1[0] < threshold)


def otsu_water(vv_db: np.ndarray) -> np.ndarray:
    """Per-image Otsu on VV. Unreliable on single 5 km chips with little water
    (it always finds *some* split) — kept for illustration and tests."""
    from skimage.filters import threshold_otsu  # imported here so the app doesn't need scikit-image

    valid = np.isfinite(vv_db)
    if valid.sum() < 2 or np.ptp(vv_db[valid]) == 0:
        return np.zeros_like(vv_db, dtype=bool)
    return valid & (vv_db < threshold_otsu(vv_db[valid]))


def authors_otsu_predictor() -> Predictor:
    import rasterio

    def predict(chip: Chip) -> np.ndarray:
        path = config.SEN1FLOODS11_DIR / "S1OtsuLabelHand" / chip.name.replace("S1Hand", "S1OtsuLabelHand")
        with rasterio.open(path) as src:
            return src.read(1) == 1

    return predict
