# SatFIN dashboard guide

The dashboard is a Streamlit web app for running SatFIN on satellite images, comparing it with other methods, and
watching the live Himawari-9 feed.

## Starting it

```bash
.venv\Scripts\activate                                   # Linux/macOS: source .venv/bin/activate
streamlit run app/dashboard.py                           # then open http://localhost:8501
python -m satfin.live --ckpt runs/gpu-20k/best.pt        # optional, second terminal: needed for "Live feed"
```

The server only accepts connections from this computer (`.streamlit/config.toml`, `address = "localhost"`).
To open it from another device on your network, change that to `"0.0.0.0"` and restart.

You need:
- a trained model in `runs/<name>/best.pt`
- processed sequences in `data/processed/` for held-out mode
- evaluation results in `outputs/**/results.json` for the bottom section

The README explains how to create each of these.

## Page layout

```
┌ Sidebar ─────────────┐ ┌ Main page ──────────────────────────────────────────┐
│ Model                │ │ Animation (original vs SatFIN vs ground truth)      │
│ Frames: mode picker  │ │ Frame by frame + difference map                     │
│ Upsampling factor    │ │ Download buttons                                    │
│ Mode-specific inputs │ │ This clip: metrics            (held-out mode only)  │
│                      │ │ Test-set evaluation                                 │
└──────────────────────┘ └─────────────────────────────────────────────────────┘
```

In **Live feed** mode the main page is different; see [Live feed](#mode-3-live-feed).

---

## Sidebar

### Model
Shows which trained model is used, currently **gpu-20k**. Only models that have evaluation results are offered.
With a single model this is a plain label. A **Model** dropdown appears only if two or more models have been
evaluated (`python -m satfin.evaluate --ckpt ...`).
Live feed mode ignores this setting, because the live loop loads its own model from its `--ckpt` argument.

### Frames (mode picker)
| Option | Use it to |
|---|---|
| **Held-out sequence (has 1-min truth)** | Test SatFIN on stored data where the real in-between frames exist, so its output can be scored |
| **Upload files** | Run SatFIN on your own satellite files (no ground truth) |
| **Live feed** | Watch the real-time Himawari-9 pipeline |

### Upsampling factor (held-out and upload modes)
Options: 2, 3, 4, 5, 6, 10. A factor of **N** inserts **N − 1** SatFIN frames between each pair of input frames.

| Factor | GOES mesoscale held-out (1-min data) | Himawari held-out (2.5-min data) | Uploaded 30-min data |
|---|---|---|---|
| 2 | 2-min input → 1-min | 5-min → 2.5-min | 30 → 15 min |
| 4 | 4-min → 1-min | 10-min → 2.5-min | 30 → 7.5 min |
| 10 | 10-min → 1-min | 25-min → 2.5-min | 30 → 3 min |

In held-out mode, the factor also sets how many real frames are hidden and used as ground truth: every Nth frame is input, the rest are truth.
Larger factors mean bigger gaps and harder interpolation.

---

## Mode 1: Held-out sequence

### Sidebar controls
| Control | What it does |
|---|---|
| **Sequence** | Which processed recording to use. Only sequences long enough for the chosen factor are listed. Full-disk scenes are hidden because they are too slow here (use `python -m satfin.infer` for those). |
| **Input pairs** | How many consecutive gaps to interpolate. For example, factor 10 with 3 pairs makes a 30-min clip from 4 input frames. |
| **Start frame** | Index of the first input frame, i.e. where in the recording the clip starts. |

**Which sequence to pick:** below the title, a caption shows the sequence's split.
- **test** (`G19_..._20250610...`): the model never saw this data, so the scores are fair.
- **train / val** (other GOES days): the model has seen this data, so the scores are optimistic.
- **unassigned** (Himawari): a different satellite that was never used in training, so it is a fair cross-satellite test. Its truth is at 2.5-min cadence, not 1-min.

### Animation
A looping GIF with the columns side by side:
- **original**: what you would see without SatFIN. The last real frame is held until the next one arrives, so it jumps.
- **SatFIN**: the smooth, interpolated sequence.
- **ground truth**: the real frames that were hidden.

Each frame is labelled with its UTC time. A `*` marks a frame that SatFIN generated.
Brightness is inverted brightness temperature: **bright = cold cloud tops** (storms), dark = warm ground or sea.

### Frame by frame
| Element | Meaning |
|---|---|
| **Frame slider** | Steps through the clip. Frame 0 and every Nth frame are real inputs. The ones in between are SatFIN's. |
| original (held) | The last real input frame at this point |
| SatFIN | SatFIN's frame for this time |
| ground truth | The real frame for this time |
| **Difference map** | \|SatFIN − truth\| in kelvin, colour scale 0 K (black) to 10 K (yellow), with the mean absolute error (MAE) in the caption. Bright areas are where SatFIN got the temperature wrong. These usually sit on fast-moving or fast-growing cloud edges. |

### Download
| Button | File | Contents |
|---|---|---|
| **NetCDF (BT in K, all frames)** | `satfin_interpolated.nc` | `bt(time, y, x)` brightness temperature in kelvin for every frame, `time`, `interpolated` (1 = SatFIN, 0 = real), and source metadata. Opens in Python (xarray/netCDF4), Panoply, QGIS and similar tools. |
| **Comparison MP4** | `satfin_comparison.mp4` | The side-by-side animation as a video, for slides or sharing |
| **Interpolated GIF** | `satfin_interpolated.gif` | SatFIN's sequence only |

### This clip: all methods vs ground truth
Scores SatFIN, **OpenCV DIS optical flow** (a classical motion-estimation method) and **linear blending** (a simple cross-fade) on the hidden frames of this clip.

| Metric | Meaning | Better |
|---|---|---|
| **PSNR (dB)** | Overall pixel accuracy on a log scale. +3 dB ≈ half the squared error. | Higher |
| **SSIM** | Structural similarity (shapes and edges), from 0 to 1 | Higher |
| **MAE (K)** | Average temperature error in kelvin | Lower |
| **Cold-cloud MAE (K)** | MAE only where the truth is colder than 235 K (deep convection / storm tops). Shows "n/a" if the clip has no such pixels. | Lower |

- **Metric cards:** each card shows SatFIN's value, and below it the difference from linear blending. Green means SatFIN is better.
- **Table:** the mean of each metric for all three methods.
- **PSNR vs t chart:** accuracy at each position inside a gap (t = 0 is the first input frame, t = 1 the next). All methods dip mid-gap, where the frame is farthest from both inputs.

---

## Mode 2: Upload files

### Sidebar control: file uploader
Drag files in or click **Browse files**. Accepted input:

| Input | Notes |
|---|---|
| GOES ABI L1b `.nc` files | One file per scan, e.g. from `noaa-goes19`. IR bands 7–16. |
| Himawari HSD `.DAT` / `.DAT.bz2` | Segments of the same scan are joined automatically. Target-area window jitter is corrected. |
| One `.npy` file | A (frames, height, width) array of brightness temperature **in kelvin**. It has no timestamps, so frames are labelled 10 min apart starting at 2000-01-01. |

- **Frame count:** you need at least 2 frames, all from the same sector.
- **Order:** files are sorted by scan time, so upload order doesn't matter.
- **Size:** Streamlit limits uploads to 200 MB in total.
- **Large frames:** frames over 1024 px are processed in tiles, but full-disk uploads are slow.
- **Errors:** if a file can't be read, a red message explains why; the app doesn't crash.

A caption confirms what was read: number of frames, size, and time range.

### Animation, Frame by frame, Download
These work like Mode 1, with two differences because there is **no ground truth**:
- There is no ground-truth column and no "This clip" metrics section.
- The difference map shows **\|SatFIN − linear blend\|** in kelvin. It highlights where SatFIN moved clouds instead of just cross-fading them. It measures **how different** the methods are, not how accurate either one is.

---

## Mode 3: Live feed

Shows the output of the background live loop (`python -m satfin.live`). The loop checks for new Himawari-9 Target scans every 30 s (a new scan comes every 2.5 min) and inserts 4 SatFIN frames between each pair, giving one frame every 30 s.
It keeps the last 6 h in `outputs/live/`. If the loop isn't running, the page shows the command to start it.

### Sidebar control: Animation span
30, 60, 120 or 360 min: how much recent history the video covers. Longer spans take a few seconds more to build.

### Status cards
| Card | Meaning |
|---|---|
| **Latest scan (UTC)** | Observation time of the newest real scan processed |
| **Scan age** | How old that scan is now. Normally about 5–8 min: NOAA delivers files 4–6 min after the scan, processing takes about 1 s, and a new scan arrives every 2.5 min. |
| **Frames in window** | Real + SatFIN frames kept (up to about 720 over 6 h) |
| **Last poll** | Seconds since the loop last checked S3 |

Warnings:
- **"No poll for N min"**: the live loop has stopped or hung. Restart it.
- **"Last error: …"**: the most recent problem (e.g. a network hiccup or an unreadable file), with its time. The loop skips the scan and continues.

### Video and latest scan
- **Video:** loops automatically. Each frame is labelled with its UTC time and **OBSERVED** (real scan) or **SatFIN** (generated).
- **Latest observed scan:** the newest real image.
- **Latest NetCDF:** downloads that scan as brightness temperature in kelvin.

The Live view refreshes itself every **60 s**; there's no need to reload the page.
SatFIN frames between two scans can only be made once the second scan arrives, so the smooth part of the video trails the newest scan by one interval (2.5 min).

---

## Test-set evaluation (bottom of the page, held-out and upload modes)

Saved benchmark results produced by `python -m satfin.evaluate`. They don't change with the clip you selected above.

| Control / element | What it does |
|---|---|
| **Results set** | Picks which benchmark to show:<br>• **GOES test day (full frame)**: 500×500 frames<br>• **GOES test day (256 crop)**: centre crop, broken down by gap<br>• **Himawari cross-satellite (full frame)**: a satellite the model never trained on |
| Model cards | Parameters, training steps, best validation PSNR, and number of test triplets (a triplet is two input frames plus one hidden truth frame) |
| Caption | Test data, gap sizes, crop, cold-cloud threshold, and the device used for timing. Gaps are counted in frames: 1 frame = 1 min on GOES, 2.5 min on Himawari. |
| **Overall** table | Every method's mean PSNR / SSIM / MAE / cold-cloud MAE, and speed in ms per frame |
| **PSNR by gap / MAE by gap** charts | How each method degrades as the gap grows |
| **Per-gap tables and error maps** (click to expand) | Full table per gap. **Error maps** show ground truth and every method on the top row, and their \|error\| in kelvin below (0–10 K). |

A yellow warning appears if these results were produced by a different model than the one in the sidebar.

---

## Tips and troubleshooting

| Situation | What to do |
|---|---|
| Slow first load after changing a setting | Normal (about 5–10 s): SatFIN runs and the animation and downloads are built. After that, the slider and page are instant because everything is cached per selection. |
| "No processed sequence has more than N frames" | Lower the upsampling factor, or preprocess longer sequences |
| Page shows an old version | Refresh the browser. Streamlit picks up code changes on the next rerun. |
| Can't reach the page from another device | By design it only listens on localhost; see [Starting it](#starting-it) |
| Live feed says the loop isn't running | Run `python -m satfin.live --ckpt runs/gpu-20k/best.pt` in a second terminal |
| Scan age keeps growing past about 10 min | NOAA delivery is delayed, or the loop stopped. Check "Last poll". |
