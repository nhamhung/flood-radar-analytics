#!/usr/bin/env python
"""Leave-one-event-out: does a method learned on some floods work on a new one?

Usage:
    python scripts/cross_event.py                  # everything (~4.5 h on an Apple-silicon GPU)
    python scripts/cross_event.py --stage classic  # VV threshold, authors' Otsu, LightGBM (~25 min)
    python scripts/cross_event.py --stage unet     # both U-Nets, 22 trainings (~4 h)

For each of the 11 flood events: train on every chip of the other 10 events,
score on every chip of the held-out one. The U-Net picks its best epoch on
10% of the *training* chips — the held-out event is never seen in training.
Also records each event's oracle threshold: the single VV cut-off that would
have been best for that event, using its own labels — how far it moves
between floods is the clearest picture of why fixed rules transfer badly.

Writes results/cross_event.json and every method's held-out predictions to
results/predictions/<method>.npz (bit-packed), for the error analysis.
Stages run as separate processes (LightGBM and PyTorch can't share one).
"""

import argparse
import functools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_radar_analytics import config, data, evaluation  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

PREDICTIONS_DIR = config.PREDICTIONS_DIR


def method_slug(method: str) -> str:
    """'LightGBM (radar + texture)' -> 'lightgbm_radar_plus_texture'."""
    keep = method.lower().replace("+", "plus").replace("'", "")
    return "_".join("".join(c if c.isalnum() else " " for c in keep).split())


def folds() -> list[tuple[str, list[str], list[str]]]:
    names = data.all_names()
    return [(event, [n for n in names if data.event_of(n) != event], [n for n in names if data.event_of(n) == event])
            for event in config.EVENTS]


def run(stage: str, epochs: int) -> dict:
    started = time.time()
    rng = np.random.default_rng(config.RANDOM_SEED)
    results = {"events": {}, "oracle_threshold_db": {}, "fold_threshold_db": {}}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for event, train_names, test_names in folds():
        print(f"=== {event}: {len(test_names)} held-out chips, {len(train_names)} training chips ({stage})")
        if stage == "classic":
            from flood_radar_analytics import pixel_model

            threshold = evaluation.fit_threshold(train_names)
            results["fold_threshold_db"][event] = threshold
            results["oracle_threshold_db"][event] = evaluation.fit_threshold(test_names)
            predictors = {
                "VV threshold": evaluation.threshold_predictor(threshold),
                "Authors' Otsu": evaluation.authors_otsu_predictor(),
                "LightGBM (radar + texture)": pixel_model.predictor(pixel_model.train(train_names)),
            }
        else:
            from flood_radar_analytics import unet

            shuffled = list(rng.permutation(train_names))
            n_valid = max(1, len(shuffled) // 10)
            predictors = {}
            for variant, label in (("s1", "U-Net (radar)"), ("s1s2", "U-Net (radar + optical)")):
                net, _ = unet.train_unet(shuffled[n_valid:], shuffled[:n_valid], variant, epochs=epochs,
                                         log=lambda _: None)
                predictors[label] = unet.unet_predictor(net, variant)

        results["events"][event] = {}
        for method, predict in predictors.items():
            confusion = evaluation.Confusion()
            for name in test_names:
                chip = data.load_chip(name, with_s2=(stage == "unet"))
                predicted = predict(chip)
                confusion.add(predicted, chip.label)
                predictions.setdefault(method, {})[name] = np.packbits(predicted.astype(bool))
            results["events"][event][method] = confusion.as_dict()
            print(f"  {method:<28} IoU {confusion.iou:.4f}  P {confusion.precision:.3f}  R {confusion.recall:.3f}"
                  f"   ({(time.time() - started) / 60:.0f} min)")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for method, by_chip in predictions.items():
        np.savez_compressed(PREDICTIONS_DIR / f"{method_slug(method)}.npz", **by_chip)
    return results


def merge() -> None:
    parts = [json.loads((config.RESULTS_DIR / f"cross_event_{s}.json").read_text()) for s in ("classic", "unet")]
    merged = {**parts[0], "events": {e: {**parts[0]["events"][e], **parts[1]["events"][e]} for e in config.EVENTS}}
    (config.RESULTS_DIR / "cross_event.json").write_text(json.dumps(merged, indent=2))
    print(f"Wrote {config.RESULTS_DIR / 'cross_event.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "classic", "unet", "merge"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.stage == "merge":
        merge()
        return
    if args.stage == "all":
        for stage in ("classic", "unet"):
            subprocess.run([sys.executable, __file__, "--stage", stage, "--epochs", str(args.epochs)], check=True)
        merge()
        return
    result = run(args.stage, args.epochs)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"cross_event_{args.stage}.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {config.RESULTS_DIR / f'cross_event_{args.stage}.json'}")


if __name__ == "__main__":
    main()
