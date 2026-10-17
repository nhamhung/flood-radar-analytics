# Start here: run the Flood Radar Analytics notebooks and app on your computer

This guide is for anyone, including people who have never installed Python. You **don't** need
Python, Git, VS Code, a Google Cloud account, a Hugging Face account or a Streamlit account. You
double-click one file, and it sets everything up.

**What you get**

- **Two notebooks**: step-by-step walk-throughs that you can run, change and re-run in your web
  browser.
  - `01_flood_radar_analytics.ipynb`: the satellite radar data, how flood maps are scored, the simple
    methods (a radar threshold, Otsu) and the LightGBM model, and what happens on a flood the model
    has never seen.
  - `02_unet_walkthrough.ipynb`: the U-Net neural network: what it is, how it is trained (a small demo
    training runs in front of you), and the results of the full-size training.
- **The app**: the interactive Flood Radar Analytics app (chip explorer, transfer across floods, why
  floods differ, the official benchmark) running on your own computer.

**What you need**

- **Windows 10 or 11** (64-bit, Intel or AMD processor), or **macOS 14 (Sonoma) or newer on a Mac
  with Apple silicon** (M1, M2, M3, M4...).
  - **Intel Macs**: needs macOS 15 or newer, and only the app and notebook 01 work. PyTorch, which
    notebook 02 needs, no longer publishes versions for Intel Macs, so the setup leaves it out there.
  - Windows on ARM (for example Surface Pro X / Copilot+ PCs) is not supported.
- An internet connection the first time:
  - about **400 MB** for Python and the packages (PyTorch alone is about 125 MB),
  - about **1.8 GB** for the satellite images the notebooks use (the app doesn't need them),
  - about **85 MB** for the U-Net's pretrained starting weights (notebook 02).
- About **6 GB** of free disk space (the packages take about 1.4 GB once installed, plus a download
  cache of the same size, plus the 1.8 GB of images).

---

## Windows

### 1. Download the project

1. Go to <https://github.com/nhamhung/flood-radar-analytics>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder, right-click `flood-radar-analytics-main.zip` and choose
   **Extract All...**. Extract it to a simple place such as
   `C:\Users\<you>\Documents\flood-radar-analytics-main`.

> Don't run it from inside the ZIP file, and avoid folders synced by OneDrive if you can
> (they work, but setup is slower, and the 1.8 GB of images would be uploaded to OneDrive).

### 2. Start it

1. Open the extracted folder and double-click **`start_windows.bat`**.
2. If Windows shows **"Windows protected your PC"**, click **More info**, then **Run anyway**.
   (This appears because the file was downloaded from the internet.)
3. A black window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 3–10 minutes,
   depending on your internet. Later starts take a few seconds.
4. When it's ready, you see a menu:

   ```
   === Flood Radar Analytics ===
     1  Open the notebooks (JupyterLab)
     2  Open the app (Streamlit)
     3  Quit
   Type 1, 2 or 3 and press Enter:
   ```

5. If Windows Firewall asks whether to allow Python, click **Allow** (it only talks to your own
   computer).

### 3. Stop it

Click the black window and press **Ctrl + C**. If it asks *"Terminate batch job (Y/N)?"*, type
**Y** and press Enter. Or simply close the black window.

---

## macOS

### 1. Download the project

1. Go to <https://github.com/nhamhung/flood-radar-analytics>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder and double-click `flood-radar-analytics-main.zip` to unzip it.
   Move the `flood-radar-analytics-main` folder somewhere simple, such as your **Documents** folder.

### 2. Start it

1. Open the folder. **The first time**, right-click (or Control-click) **`start_mac.command`**,
   choose **Open**, then click **Open** again in the warning. (macOS asks because the file was
   downloaded from the internet. After this, a normal double-click works.)
   - If macOS says it *"cannot be opened because Apple cannot check it"* and there is no Open
     button: open **System Settings → Privacy & Security**, scroll down, click **Open Anyway**
     next to `start_mac.command`, then try again.
2. A Terminal window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 3–10 minutes,
   depending on your internet. Later starts take a few seconds.
   On most Macs the first start also prints *"One-time fix: downloading the
   OpenMP runtime..."*: that is a small library (about 300 kB) the model needs on macOS, and it is
   stored inside the project folder.
3. When it's ready, you see the same menu as on Windows: type **1** (notebooks) or **2** (app)
   and press Return.

### 3. Stop it

Click the Terminal window and press **Control + C**, then close the window.

---

## Using the notebooks (choice 1)

**The first time**, before JupyterLab opens, the launcher checks for the satellite images the
notebooks read (446 image tiles, "chips", from 11 floods: the public *Sen1Floods11* dataset).
They are not part of the ZIP, so it asks:

```
Download them now? [y/N]
```

Type **y** and press Enter. This downloads about **1.8 GB** from the dataset's free public storage
(no account needed) and takes roughly **5–30 minutes**, depending on your internet. It prints a
line after each of the 5 image layers, so it can look stuck for a few minutes: that's normal.
If it is interrupted (or you close the window), just choose **1** again: it skips what is already
downloaded and continues. Right after, it downloads the U-Net's pretrained starting weights for
notebook 02 (about 85 MB, once). After that, the notebooks work without internet.

Then JupyterLab opens in your web browser.

1. In the file list on the left, double-click **`01_flood_radar_analytics.ipynb`**. Start with
   this one: notebook 02 builds on it.
2. Run it from top to bottom with **Run → Run All Cells**. Notebook 01 takes about 3–5 minutes.
   A `[*]` next to a cell means it is still running.
3. To run just one cell, click it and press **Shift + Enter**.
4. **Experiment freely.** Change a number in a cell and run it again. Each notebook ends with
   suggested exercises.
5. If things get into a strange state, use **Kernel → Restart Kernel and Run All Cells**.

**Notebook 02 (the U-Net)** takes about 1.5 minutes on a recent Apple-silicon Mac, which trains on
the Mac's graphics chip. Windows computers here train on the processor (CPU) instead, which is
much slower: expect tens of minutes for the demo training cell. The notebook still works the same.

**Run the two notebooks one after the other, not mixed in one notebook.** They use two
machine-learning libraries (LightGBM in notebook 01, PyTorch in notebook 02) that crash or freeze
if they are loaded together on a Mac. Each notebook you open in JupyterLab gets its own separate
Python process ("kernel"), so having both open is fine. Just don't copy code that loads one
library into the other notebook. (Notebook 02's text mentions a kernel named *Flood Radar
Analytics*: the launcher's standard **Python 3 (ipykernel)** kernel is the right one.)

Your edits are saved in the notebook files. To keep the originals, first use **File → Save
Notebook As...** and work on the copy (or download the ZIP again later).

## Using the app (choice 2)

The app needs no download: it shows the project's saved results, and fetches each satellite image
tile from the internet when you open it (or reads it from your computer if the notebooks'
download is already there). It opens in your browser by itself (if it doesn't, copy the
`http://localhost:...` address shown in the black/Terminal window into your browser). Pages:

- **Chip Explorer Demo**: pick an image tile from any flood and compare each method's water map
  with the hand-drawn label.
- **Transfer Across Floods**: how well each method works on a flood it has never seen.
- **Why Floods Differ**: what makes some floods hard for radar.
- **Official Benchmark**: the scores on the dataset's official test split.

Nothing you do in the app changes the project files.

## Next time

Just double-click `start_windows.bat` or `start_mac.command` again. No internet needed once the
first setup and the image download have finished (the app's Chip Explorer needs internet only for
tiles that aren't downloaded).

---

## Optional: use VS Code instead of the browser

If you prefer VS Code for the notebooks:

1. Run the start file once (above) and choose **1**, so the project's Python environment (the
   `.venv` folder), the images and the pretrained weights are all in place. You can then close
   the browser tab and stop it.
2. Install VS Code from <https://code.visualstudio.com>.
3. In VS Code: **File → Open Folder...** and choose the project folder. Accept the prompt to
   install the recommended **Python** and **Jupyter** extensions.
4. Open a notebook in the `notebooks` folder. Click **Select Kernel** (top right) →
   **Python Environments** → the one marked **`.venv`** (Recommended).

That is the only kernel step, and you do it once per notebook.

On a Mac, VS Code doesn't get the launcher's OpenMP fix: if a cell fails with *"Library not
loaded: @rpath/libomp.dylib"*, either use the launcher's JupyterLab instead, or install
[Homebrew](https://brew.sh) and run `brew install libomp` in Terminal once.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| macOS: *"Library not loaded: @rpath/libomp.dylib"* or *"brew install libomp"* | The launcher normally fixes this by itself (it needs the internet once). If it says the fix didn't work, install [Homebrew](https://brew.sh), run `brew install libomp` in Terminal, and start again. |
| Installing uv fails (often on work computers that block scripts) | Install uv another way, then double-click the start file again. **Windows:** open *Command Prompt* and run `winget install --id=astral-sh.uv -e`. **macOS:** if you use Homebrew, run `brew install uv` in Terminal. Or follow <https://docs.astral.sh/uv/getting-started/installation/>. |
| The setup fails with *"no wheels"* or *"unsatisfiable"* on a Mac | Your macOS is too old for the image library: update to macOS 14 or newer (Apple silicon) or macOS 15 or newer (Intel). Apple menu → **About This Mac** shows your version. |
| macOS: *"permission denied"* or the file opens in a text editor | Open **Terminal**, type `bash ` (with a space), drag `start_mac.command` into the Terminal window, and press Return. |
| The browser didn't open | Copy the `http://localhost:...` address from the black/Terminal window into your browser. |
| *"Address already in use"* | Another copy is probably running. Close the other black/Terminal window, or just pick the option again: the launcher picks a free port. |
| The image download stopped or failed | Choose **1** again: it continues where it stopped. Work networks sometimes block large downloads; try another network. |
| Notebook 02 says *"Pretrained weights unavailable ... using random starting weights instead"* | The weights couldn't be downloaded (no internet the first time). The notebook still runs; the demo model just learns more slowly. Choose **1** again once you're online to download them. |
| Notebook 02 on an Intel Mac: *"No module named 'torch'"* | Expected: PyTorch isn't available for Intel Macs (see *What you need*). Notebook 01 and the app work. |
| Errors after editing a notebook | **Kernel → Restart Kernel and Run All Cells**. Still broken? Download the ZIP again for a fresh copy (copy its `data` folder over to skip the image download). |
| Something else is broken | Delete the `.venv` folder inside the project and double-click the start file again: it rebuilds a clean setup (the images in `data` are kept). |

## Uninstall

1. Delete the project folder (this also deletes the 1.8 GB of images).
2. Optional, to remove uv, the downloaded Python and the pretrained weights too:
   - **Windows:** delete `uv.exe` and `uvx.exe` from `C:\Users\<you>\.local\bin`, the folders
     `%APPDATA%\uv` and `%LOCALAPPDATA%\uv`, and the folder
     `C:\Users\<you>\.cache\huggingface\hub\models--smp-hub--resnet34.imagenet`.
   - **macOS:** in Terminal, run `rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv ~/.cache/uv ~/.cache/huggingface/hub/models--smp-hub--resnet34.imagenet`.

---

## For developers

The project is a standard [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` lists the
direct dependencies at the exact versions it was tested with, and `uv.lock` pins everything else.

```bash
uv sync                                     # create .venv (add --no-dev to skip pytest/ruff)
uv run python scripts/download_data.py      # the Sen1Floods11 chips (~1.8 GB, resumable)
uv run pytest                               # tests (synthetic data, no download needed)
uv run jupyter lab notebooks                # notebooks
uv run streamlit run app/streamlit_app.py   # app
```

`scripts/launch.py` is what the start files run. `requirements.txt` and `app/requirements.txt`
stay for CI and Streamlit Community Cloud. To publish your own changes to GitHub, see the setup
manual at <https://github.com/nhamhung/nhamhung.github.io/blob/main/README.md> (Git, VS Code,
clone, push).
