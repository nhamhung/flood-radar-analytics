#!/usr/bin/env python
"""Train every method on the official Sen1Floods11 split and score it.

Usage:
    python scripts/train.py                     # everything (~30 min on an Apple-silicon GPU)
    python scripts/train.py --stage baselines   # VV threshold, authors' Otsu, LightGBM (~3 min)
    python scripts/train.py --stage unet        # both U-Nets
    python scripts/train.py --stage unet --reuse-existing   # skip U-Nets already saved in models/

Writes models/lightgbm.joblib, models/unet_s1.pt, models/unet_s1s2.pt and
results/benchmark.json: pooled and per-event IoU / F1 / precision / recall
for every method on the validation, test and held-out Bolivia splits.

The two stages run as separate processes: LightGBM and PyTorch must never
share one (see the flood_radar_analytics package docstring).
"""

import argparse
import functools
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_radar_analytics import config, data, evaluation  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

EVAL_SPLITS = ("valid", "test", "bolivia")
UNET_VARIANTS = {"s1": "U-Net (radar)", "s1s2": "U-Net (radar + optical)"}


def score(predictors: dict, with_s2: bool) -> dict:
    out = {}
    for split in EVAL_SPLITS:
        evaluations = evaluation.evaluate(predictors, data.split_names(split), with_s2=with_s2)
        out[split] = {method: {"pooled": e.pooled.as_dict(),
                               "by_event": {k: c.as_dict() for k, c in e.by_event.items()}}
                      for method, e in evaluations.items()}
        for method, e in evaluations.items():
            d = e.pooled.as_dict()
            print(f"  {split:<8} {method:<28} IoU {d['IoU']:.4f}  P {d['precision']:.3f}  R {d['recall']:.3f}")
    return out


def stage_baselines() -> dict:
    import joblib

    from flood_radar_analytics import pixel_model

    train_names = data.split_names("train")
    threshold = evaluation.fit_threshold(train_names)
    print(f"VV threshold: water where VV < {threshold:.2f} dB")
    lgbm = pixel_model.train(train_names)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm, config.model_path("lightgbm.joblib"))
    predictors = {
        "VV threshold": evaluation.threshold_predictor(threshold),
        "Authors' Otsu": evaluation.authors_otsu_predictor(),
        "LightGBM (radar + texture)": pixel_model.predictor(lgbm),
    }
    return {"vv_threshold_db": threshold, "splits": score(predictors, with_s2=False)}


def stage_unet(epochs: int, reuse_existing: bool) -> dict:
    from flood_radar_analytics import unet

    train_names, valid_names = data.split_names("train"), data.split_names("valid")
    predictors, history = {}, {}
    for variant, label in UNET_VARIANTS.items():
        if reuse_existing and config.model_path(f"unet_{variant}.pt").exists():
            print(f"Reusing saved {label} ({config.model_path(f'unet_{variant}.pt').name})")
            net, history[variant] = unet.load_unet(variant), None  # training curve not re-recorded
        else:
            print(f"Training {label} for {epochs} epochs ({unet.device()}) ...")
            net, history[variant] = unet.train_unet(train_names, valid_names, variant, epochs=epochs, log=print)
            unet.save_unet(net, variant)
        predictors[label] = unet.unet_predictor(net, variant)
    return {"history": history, "splits": score(predictors, with_s2=True)}


def merge() -> None:
    parts = [json.loads((config.RESULTS_DIR / f"benchmark_{s}.json").read_text()) for s in ("baselines", "unet")]
    merged = {"vv_threshold_db": parts[0]["vv_threshold_db"], "history": parts[1]["history"], "splits": {}}
    for split in EVAL_SPLITS:
        merged["splits"][split] = {**parts[0]["splits"][split], **parts[1]["splits"][split]}
    (config.RESULTS_DIR / "benchmark.json").write_text(json.dumps(merged, indent=2))
    print(f"Wrote {config.RESULTS_DIR / 'benchmark.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "baselines", "unet", "merge"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Load U-Nets already saved in models/ instead of retraining them.")
    args = parser.parse_args()

    started = time.time()
    if args.stage == "merge":
        merge()
        return
    if args.stage == "all":
        extra = ["--reuse-existing"] if args.reuse_existing else []
        for stage in ("baselines", "unet"):
            subprocess.run([sys.executable, __file__, "--stage", stage, "--epochs", str(args.epochs), *extra], check=True)
        merge()
    else:
        result = stage_baselines() if args.stage == "baselines" else stage_unet(args.epochs, args.reuse_existing)
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.RESULTS_DIR / f"benchmark_{args.stage}.json").write_text(json.dumps(result, indent=2))
    print(f"Done ({args.stage}) in {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
