#!/usr/bin/env python
"""Download the Sen1Floods11 hand-labelled data (public, no account needed).

Usage:
    python scripts/download_data.py            # everything (~1.9 GB)
    python scripts/download_data.py --skip-s2  # radar + labels only (~0.9 GB)

Fetched over plain HTTPS from the public `sen1floods11` Google Cloud bucket
(no `gsutil`, no login). Already-downloaded files are skipped, so it is safe
to re-run after an interruption.
"""

import argparse
import functools
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_radar_analytics import config  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

BUCKET = "sen1floods11"
PREFIX = "v1.1"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
FILE_URL = f"https://storage.googleapis.com/{BUCKET}/"


def list_objects(prefix: str) -> list[str]:
    names, token = [], None
    while True:
        query = {"prefix": prefix, "fields": "items(name),nextPageToken"}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(f"{LIST_URL}?{urllib.parse.urlencode(query)}") as response:
            page = json.load(response)
        names += [item["name"] for item in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return names


def fetch(name: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(FILE_URL + urllib.parse.quote(name), tmp)
    tmp.rename(target)


def download_layer(layer: str) -> None:
    names = [n for n in list_objects(f"{PREFIX}/data/flood_events/HandLabeled/{layer}/") if n.endswith(".tif")]
    targets = [config.SEN1FLOODS11_DIR / layer / Path(n).name for n in names]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch, names, targets))
    print(f"  {layer}: {len(names)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-s2", action="store_true", help="Skip the optical Sentinel-2 chips (~1 GB).")
    args = parser.parse_args()

    for layer in config.LAYERS:
        if layer == "S2Hand" and args.skip_s2:
            continue
        download_layer(layer)
    for name in list_objects(f"{PREFIX}/splits/flood_handlabeled/"):
        fetch(name, config.SPLITS_DIR / Path(name).name)
    print(f"  splits: {len(list(config.SPLITS_DIR.glob('*.csv')))} files")


if __name__ == "__main__":
    main()
