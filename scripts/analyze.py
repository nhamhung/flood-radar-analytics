#!/usr/bin/env python
"""Turn trained-model outputs into the small result tables the app, notebook and report read.

Usage:
    python scripts/analyze.py

Needs the data (scripts/download_data.py) and the held-out predictions from
scripts/cross_event.py. Writes to results/:
- event_separability.csv — per event: water share, VV contrast, VV AUC
- error_anatomy_<method>.json — recall by water type and by vegetation (NDVI)
"""

import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_radar_analytics import analysis, config  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    separability = analysis.event_separability()
    separability.to_csv(config.RESULTS_DIR / "event_separability.csv")
    print(f"event_separability.csv: {len(separability)} events")

    for path in sorted(analysis.PREDICTIONS_DIR.glob("*.npz")):
        slug = path.stem
        tables = analysis.error_anatomy(slug)
        out = {
            "by_water_type": tables["by_water_type"].reset_index().to_dict(orient="records"),
            "by_vegetation": tables["by_vegetation"].reset_index(names="ndvi").to_dict(orient="records"),
            "by_event_and_water_type": tables["by_event_and_water_type"].to_dict(orient="records"),
        }
        (config.RESULTS_DIR / f"error_anatomy_{slug}.json").write_text(json.dumps(out, indent=2))
        recall = tables["by_water_type"]["recall"]
        print(f"error_anatomy_{slug}.json: recall permanent water {recall['permanent water']:.3f}, "
              f"new floodwater {recall['new floodwater']:.3f}")


if __name__ == "__main__":
    main()
