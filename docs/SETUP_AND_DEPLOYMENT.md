# Setup and Deployment

## Local setup

```bash
git clone https://github.com/nhamhhung/flood-radar-analytics.git
cd flood-radar-analytics
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r app/requirements.txt
pytest tests/
python scripts/smoke_app.py
streamlit run app/streamlit_app.py
```

To retrain everything from scratch (not needed for the app):
`python scripts/download_data.py`, then `train.py`, `cross_event.py` and
`analyze.py` — see the README.

## Deploy your fork

1. Fork or clone this repository and replace the values in `docs/deployment-config.yml`.
2. Push to your own public GitHub repository with `main` as the default branch.
3. In Settings → Pages, select **GitHub Actions** as the source.
4. In Streamlit Community Cloud, create a new app from your repository, branch `main`, file `app/streamlit_app.py`, Python 3.12.
5. **No secrets are needed.** The app reads the bundled results in `results/` and the Chip Explorer downloads only the chips a viewer opens (a few MB each) from the public Sen1Floods11 bucket. Model weights and the 2 GB dataset are not needed. `app/requirements.txt` is the minimal pinned set (streamlit, matplotlib, rasterio).
6. Wait for CI and Pages to pass, then record the URLs and revision in `docs/DEPLOYMENT_ACCEPTANCE.md`.

To rehearse the cloud exactly — no local data at all — point the data
directory at an empty folder: `FLOOD_RADAR_DATA_DIR=$(mktemp -d) streamlit run app/streamlit_app.py`.

Do not copy another owner's tokens, application URL, Pages URL, or secret values. GitHub Pages hosts static documentation; Streamlit Community Cloud runs the Python app.

## Required checks

Protect `main` after the initial bootstrap. Require `ruff`, `pytest (3.11)`, `pytest (3.12)`, `smoke (3.11)`, `smoke (3.12)`, and `pages-build`. Disable force pushes and direct pushes; use up-to-date pull requests and squash merges.

## Rollback

Use the most recent SHA recorded as known-good in `DEPLOYMENT_ACCEPTANCE.md`. Revert later commits on a `rollback/<date>` branch, pass all required checks, and merge normally. Verify both Pages and Streamlit before closing the rollback.
