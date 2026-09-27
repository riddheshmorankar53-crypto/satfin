# SatFIN codebase guide

A map of the repository for answering "where is X / how does Y work / why was it done this way".
Every section names the file and the line where the code lives (`file:line`).

**Contents**
1. [One-paragraph summary](#1-one-paragraph-summary)
2. [Folder map](#2-folder-map)
3. [Pipeline: which file does what, in order](#3-pipeline-which-file-does-what-in-order)
4. [Quick lookup: "where is…?"](#4-quick-lookup-where-is)
5. [File-by-file details](#5-file-by-file-details)
6. [Key numbers](#6-key-numbers)
7. [Likely interview questions](#7-likely-interview-questions)

---

## 1. One-paragraph summary

SatFIN raises the **temporal resolution** of geostationary weather-satellite imagery. Given two infrared images
taken minutes apart, it generates realistic frames *in between*: 10-min data becomes 1-min, and 2.5-min data becomes 30-s.
It is a RIFE-style network:
1. It estimates **optical flow** (how every pixel moves) from time *t* back to both input frames.
2. It **warps** both frames to time *t* and blends them with a learned mask.
3. A small **U-Net** refines the result.

It was trained on real 1-min **GOES-19** mesoscale data, where the true in-between frames exist. It was tested on an unseen GOES day and on **Himawari-9** (a different satellite).
It runs **live** on the Himawari-9 feed from AWS. On GOES it beats classical optical flow (OpenCV DIS) by +3.5 dB PSNR.

---

## 2. Folder map

```
satfin/                         repository root
├── README.md                   project overview, setup, commands, results, limitations
├── requirements.txt            Python dependencies
├── configs/default.yaml        ALL settings: data paths, bounds, splits, model size, loss weights, training
├── .streamlit/config.toml      dashboard server: localhost only
│
├── satfin/                     the Python package (python -m satfin.<module>)
│   ├── config.py               load the YAML config
│   ├── env_check.py            print GPU/CPU info, pick the device
│   │
│   ├── data/                   STAGE 1-2: getting and preparing data
│   │   ├── download.py         list + download files from the NOAA S3 buckets
│   │   ├── loaders/
│   │   │   ├── __init__.py     format dispatch: .nc → GOES, .DAT(.bz2) → Himawari
│   │   │   ├── goes.py         GOES ABI NetCDF → brightness temperature (Planck)
│   │   │   └── himawari.py     Himawari HSD binary → brightness temperature (hand-written parser)
│   │   ├── preprocess.py       raw files → normalized sequences (frames.npy/times.npy/meta.json)
│   │   └── dataset.py          PyTorch Dataset of (I0, It, I1, t) training triplets + date split
│   │
│   ├── models/                 STAGE 3: the neural network
│   │   ├── warp.py             backward warping (grid_sample)
│   │   ├── ifnet.py            IFNet: coarse-to-fine flow + mask estimator
│   │   ├── fusion.py           refinement U-Net (residual correction)
│   │   └── satfin.py           full model = IFNet + warp + blend + U-Net
│   │
│   ├── losses.py               STAGE 4: training losses
│   ├── train.py                STAGE 4: training loop
│   │
│   ├── baselines.py            STAGE 5: linear blend + OpenCV optical-flow baselines
│   ├── metrics.py              STAGE 5: PSNR, SSIM, MAE (K), cold-cloud MAE
│   ├── evaluate.py             STAGE 5: benchmark SatFIN vs baselines → results.md/.json + error maps
│   │
│   ├── infer.py                STAGE 6: run the model on any sequence (tiled), write NetCDF/PNG/GIF/MP4
│   ├── visualize.py            STAGE 6: image/GIF/MP4/error-map helpers
│   └── live.py                 STAGE 7: real-time loop on the Himawari-9 feed
│
├── app/dashboard.py            STAGE 8: Streamlit web UI
├── scripts/                    quick_demo.sh/.ps1 (whole pipeline end-to-end), download_goes.sh
├── tests/                      pytest suite (19 tests)
└── docs/
    ├── DASHBOARD.md            how to use every control in the web UI
    └── CODEBASE.md             this file

Not in git (regenerable, .gitignore):
data/raw/        downloaded satellite files        data/processed/  normalized sequences
runs/<name>/     checkpoints + TensorBoard logs    outputs/         results, figures, animations, live frames
```

---

## 3. Pipeline: which file does what, in order

```
 NOAA S3 (GOES-19, Himawari-9)
        │  download.py
        ▼
 raw .nc / .DAT.bz2 files ──loaders/goes.py, loaders/himawari.py──► brightness temperature (K)
        │  preprocess.py: normalize 180–330 K → [0,1], split into continuous sequences, align
        ▼
 data/processed/<seq>/frames.npy, times.npy, meta.json
        │  dataset.py: (I0, It, I1, t) triplets, date-based train/val/test split, augmentation
        ▼
 models/satfin.py  ◄── ifnet.py (flow + mask) ── warp.py ── fusion.py (U-Net residual)
        │  train.py + losses.py → runs/gpu-20k/best.pt
        ▼
 evaluate.py (+ metrics.py, baselines.py, visualize.py) → outputs/results.md/.json, error_maps/
 infer.py    (tiled, any cadence)                       → outputs/interpolated.nc, GIF/MP4
 live.py     (Himawari-9 every 2.5 min)                 → outputs/live/frames/*.nc/.png
        │
        ▼
 app/dashboard.py (Streamlit): held-out / upload / live views
```

| Stage | Command | File(s) |
|---|---|---|
| 0 Environment | `python -m satfin.env_check` | `env_check.py` |
| 1 Download | `python -m satfin.data.download --start 2025-06-01T18:00 --hours 3 --sector M1` | `data/download.py` |
| 2 Preprocess | `python -m satfin.data.preprocess` | `data/preprocess.py`, `data/loaders/*` |
| 3 Model check | `python -m satfin.models.satfin` | `models/*` |
| 4 Train | `python -m satfin.train --run gpu-20k` | `train.py`, `losses.py`, `data/dataset.py` |
| 5 Evaluate | `python -m satfin.evaluate --ckpt runs/gpu-20k/best.pt` | `evaluate.py`, `metrics.py`, `baselines.py` |
| 6 Inference | `python -m satfin.infer --ckpt … --seq … --every 10 --k 9` | `infer.py`, `visualize.py` |
| 7 Live | `python -m satfin.live --ckpt runs/gpu-20k/best.pt` | `live.py` |
| 8 UI | `streamlit run app/dashboard.py` | `app/dashboard.py` |
| Tests | `python -m pytest` | `tests/` |
| All at once | `bash scripts/quick_demo.sh` | `scripts/` |

---

## 4. Quick lookup: "where is…?"

| Question | Answer |
|---|---|
| Radiance → brightness temperature (GOES)? | `satfin/data/loaders/goes.py:10` `rad_to_bt` (inverse Planck with the file's constants) |
| Himawari binary format parsing? | `satfin/data/loaders/himawari.py:35` `read_segment` |
| Bad/invalid pixel handling? | GOES DQF: `loaders/goes.py:16`. Himawari error counts: `himawari.py:35`. NaN fill: `preprocess.py:72` |
| Normalization? | `satfin/data/preprocess.py:19` `normalize` (fixed 180–330 K → [0,1]) |
| How sequences are split / cadence gaps? | `preprocess.py:58` `split_sequences` (gap > 1.5× median cadence) |
| Himawari jitter correction? | `preprocess.py:41` `align`, `preprocess.py:32` `same_place` |
| How training triplets are built? | `satfin/data/dataset.py:30` `TripletDataset` |
| Train/val/test split (no leakage)? | `dataset.py:11` `split_dirs` (by date, GOES only) + `configs/default.yaml` `splits` |
| Data augmentation? | `dataset.py:54` `__getitem__` (crop, flips, rot90, time reversal) |
| Model architecture? | `satfin/models/satfin.py:16` `SatFIN.forward` |
| Optical-flow estimator? | `satfin/models/ifnet.py:36` `IFNet`, one scale = `ifnet.py:13` `IFBlock` |
| Warping? | `satfin/models/warp.py:6` `backward_warp` |
| Refinement network? | `satfin/models/fusion.py:21` `FusionUNet` |
| Where is *t* (time) fed in? | `ifnet.py:52`: t is broadcast to a constant image channel |
| Loss function? | `satfin/losses.py:40` `total_loss` |
| Storm/convective weighting? | `losses.py:14` `convective_weight` |
| Optimizer, LR schedule, AMP, clipping? | `satfin/train.py:65` `main` |
| Checkpointing / resume? | `train.py:117` `save`, `--resume` in `train.py:65` |
| Classical baselines? | `satfin/baselines.py:6` `linear_blend`, `:27` `flow_interp` |
| Metrics? | `satfin/metrics.py:13` `scores` (PSNR, SSIM, MAE K, cold MAE); `:7` `psnr` (training) |
| Benchmark script? | `satfin/evaluate.py:41` `main` |
| Tiled inference for full disk? | `satfin/infer.py:59` `interpolate` (+ `_ramp` `:41`, `_starts` `:51`) |
| Inserting K frames into a whole sequence? | `infer.py:85` `interpolate_sequence` |
| NetCDF output? | `infer.py:100` `save_netcdf` |
| GIF/MP4/error-map images? | `satfin/visualize.py` |
| Real-time loop? | `satfin/live.py:51` `LiveRunner` (`poll` `:69`, `add` `:91`, `prune` `:119`) |
| Waiting for all full-disk segments? | `live.py:38` `complete_scans` |
| Dashboard caching? | `app/dashboard.py:131` `run`, `:137` `artifacts` |
| All hyperparameters? | `configs/default.yaml` |
| Results? | `README.md` → Results; raw files `outputs/**/results.md` |

---

## 5. File-by-file details

### 5.1 Configuration and environment

#### `configs/default.yaml`
The single source of settings. There are no hard-coded paths.

| Section | Key settings |
|---|---|
| `data` | `bt_min/bt_max` 180/330 K; `gaps` [2, 4, 10] frames; `crop` 256; `split_platforms` [G19]; `splits` train 06-01..03 / val 06-05 / test 06-10 |
| `model` | `channels` 1; `ifnet_widths` [96, 64, 48]; `scales` [4, 2, 1]; `fusion_width` 32 |
| `loss` | `cold_bt` 235 K, `cold_weight` 4, `grad_thresh_K` 5, `grad_weight` 2, `smooth_weight` 0.01, `edge_weight` 0.1 |
| `train` | batch 8, lr 3e-4, AdamW wd 1e-4, 20k steps, grad clip 1.0, AMP, val every 1000 steps on 256 fixed triplets |
| `cpu` | overrides merged in when there is no GPU: crop 128, batch 4, 2000 steps |

#### `satfin/config.py`
`load_config()` (`:9`) reads the YAML into a dict.

#### `satfin/env_check.py`
`get_device()` (`:8`) returns CUDA if available, else CPU. `python -m satfin.env_check` prints the Python, PyTorch and GPU versions.
*Story:* this check revealed that a CPU-only PyTorch was installed on a machine with an RTX 4060. Fixing that made the full training run possible.

---

### 5.2 Data (`satfin/data/`)

#### `download.py`: get raw files from AWS
- **Source:** the public, anonymous NOAA buckets `noaa-goes19` and `noaa-himawari9`, accessed with `s3fs` (`anon=True`).
- **`parse_start_time` (`:15`):** scan time from a GOES filename (`_sYYYYDDDHHMMSS`).
- **`list_files` (`:21`):** lists GOES keys per hour folder, filtered by sector (`-RadM1-`), band (`C13_`) and time.
- **`himawari_time` (`:40`) and `list_himawari` (`:47`):** the Himawari equivalents. Himawari folders are per 10 min, and Target sub-scans R301–R304 are 2.5 min apart.
- **`download` (`:62`):**
  - skips files that already exist with the right size
  - writes to `.part` and then renames, so an interrupted download never leaves a truncated file

#### `loaders/goes.py`: GOES ABI NetCDF → brightness temperature
- **`rad_to_bt` (`:10`):** `BT = (fk2 / ln(fk1/L + 1) − bc1) / bc2`, the inverse Planck function from the GOES-R Product User's Guide. The constants `planck_fk1/fk2/bc1/bc2` come from each file.
- **`read_goes_bt` (`:16`):**
  - reads `Rad` and the `DQF` quality flag; DQF ≥ 2 (bad / no value) becomes NaN
  - also returns metadata: time, platform, scene, band, sector centre
- **Why `netCDF4` and not `xarray`:** xarray took about 5 s to open each ABI file; netCDF4 is near-instant.

#### `loaders/himawari.py`: Himawari Standard Data (HSD) → brightness temperature
- **Parser:** hand-written, so there's no dependency on `satpy`. An HSD file is 11 binary header blocks followed by uint16 counts. `_blocks` (`:21`) walks the blocks using each block's own length field.
- **`read_segment` (`:35`):**
  - counts → radiance (`gain·count + offset`, from block 5)
  - radiance → effective temperature (Planck, using the central wavelength)
  - effective temperature → BT with the correction `c0 + c1·Te + c2·Te²`
  - error and outside-scan counts → NaN
  - reads the projection (sub-longitude, COFF/LOFF), the time (MJD) and the segment number
- **`read_himawari_bt` (`:77`):** stacks the full-disk segments (10 per scan) in segment order.
- **How it was checked:**
  - the byte offsets were checked against a real file
  - band 13 gave 186–298 K, which is physically plausible
  - a unit test builds a synthetic HSD file and checks that the Planck round trip reproduces the radiance

#### `loaders/__init__.py`: format dispatch
- `find_files` (`:12`) returns the supported files in a folder.
- `group_scans` (`:17`) groups Himawari segments of one scan.
- `read_scan` (`:25`) sends `.nc` files to the GOES reader and `.DAT(.bz2)` files to the Himawari reader.

#### `preprocess.py`: raw files → normalized sequences
- **`normalize` / `denormalize` (`:19` / `:24`):** a fixed physical scale, 180–330 K → [0,1].
  *Why fixed, not per-image min/max:* the same temperature always maps to the same value, so outputs can be converted back to kelvin exactly.
- **`same_place` (`:32`) and `align` (`:41`):**
  - The Himawari Target window shifts about 5 px between scans (COFF alternates 180.5/185.5).
  - Without correction, that shift would look like fake cloud motion.
  - `align` crops every frame so each pixel is the same fixed-grid location.
  - It was checked on real data: mean frame-to-frame difference is 3.5 K aligned vs 7.7 K with the wrong sign.
- **`split_sequences` (`:58`):** starts a new sequence when the scene moves or a gap exceeds 1.5× the median cadence. This works for 1-min GOES and 2.5-min Himawari alike.
- **`process_dir` (`:72`):** writes `frames.npy` (float16), `times.npy` and `meta.json` per sequence. Rare NaN pixels are filled with the frame mean.

#### `dataset.py`: training triplets
- **`split_dirs` (`:11`):**
  - assigns sequences to train/val/test **by date**, and asserts that no date is in two splits
  - only `split_platforms` (GOES-19) is used, so Himawari stays an unseen test set
  - *Why by date:* neighbouring minutes are nearly identical, so a random-frame split would leak test data into training.
- **`TripletDataset` (`:30`):**
  - **Triplets:** from 1-min sequences, every (I0, It, I1) with gaps of 2, 4 and 10 frames. This simulates low-cadence data where the missing frames are known.
  - **`t` from real timestamps**, not k/gap, because scan times jitter by a few seconds.
  - **Dropped frames:** a triplet is skipped if its span exceeds (gap + 0.5) × cadence.
  - **Augmentation (`__getitem__` `:54`):** random crop, horizontal/vertical flip, 90° rotation, and **time reversal** (swap I0/I1 and set t → 1 − t).
  - **Counts:** 6,708 train, 1,456 val and 1,456 test triplets.

---

### 5.3 Model (`satfin/models/`), 1.14 M parameters

```
I0, I1, t ──► IFNet (3 scales: 1/4 → 1/2 → 1) ──► flow F(t→0), F(t→1)  +  mask
                                                     │
               backward_warp(I0, F(t→0)) = W0 ◄──────┤
               backward_warp(I1, F(t→1)) = W1 ◄──────┘
               merged = mask·W0 + (1 − mask)·W1
               pred   = merged + FusionUNet(W0, W1, merged, mask, flow, warped context)
```

#### `warp.py`: `backward_warp` (`:6`)
For every output pixel *p*, it samples the source image at *p + flow(p)* with `grid_sample` (bilinear, border padding).
*Why backward warping:* every output pixel gets exactly one value, with no holes or collisions. Forward warping (splatting) leaves gaps.
It is tested: zero flow gives the identity, and an integer shift gives an exact pixel shift.

#### `ifnet.py`: flow estimator (0.65 M parameters)
- **`IFBlock` (`:13`), one scale:**
  - downsamples the input by `scale`, then twice more with stride-2 convolutions
  - predicts a 4-channel flow update plus a 1-channel mask update, then upsamples to full resolution
- **`IFNet` (`:36`):**
  - three blocks at scales 4, 2 and 1, coarse to fine
  - the first block sees (I0, I1, t)
  - later blocks also see both frames warped with the current flow, the mask and the flow, and add *residual* updates
  - *Why coarse-to-fine:* the coarse scale catches large motion cheaply; finer scales fix details.
- **Time *t*:** fed as a constant extra image channel (`:52`), so one model handles any t ∈ (0, 1).

#### `fusion.py`: refinement (0.49 M parameters)
- **`ContextNet` (`:10`):** feature maps of each input frame, warped to time *t*.
- **`FusionUNet` (`:21`):**
  - a 3-level U-Net over (W0, W1, merged, mask, flow, context)
  - outputs a `tanh` **residual** added to the blended frame
  - *Why a residual:* the warp already gets most of it right; the U-Net only fixes occlusions and blend artefacts.

#### `satfin.py`: the full model
- **`SatFIN.forward` (`:26`):**
  1. **Per-sample standardization:** subtract the mean, divide by the std of I0 and I1.
     *Why:* normalized BT has a std of only about 0.03, and the network trained poorly on such tiny signals. The output is transformed back and clamped to [0,1].
  2. **Replicate-pad** to a multiple of 16, so any H × W works.
  3. Run IFNet, warp, blend with the sigmoid mask, add the U-Net residual, and crop back.
  4. Return `pred`, `merged`, `flow` and `mask`, which the losses use.
- **`build_model` (`:47`):** builds the model from config.
- **`python -m satfin.models.satfin`:** prints the parameter count and runs a dummy forward pass.
- **Multispectral:** set `channels: N` for N-band input (the model supports it and it's tested; the data pipeline is single-band).

---

### 5.4 Training

#### `losses.py`
- **`charbonnier` (`:5`):** `sqrt((pred − gt)² + ε²)`, a smooth L1. It is robust to outliers and doesn't blur like L2.
- **`convective_weight` (`:14`):** each pixel's weight is
  `1 + 4·[any frame colder than 235 K] + 2·[|BT(I1) − BT(I0)| > 5 K]`, up to 7×.
  *Why:* cold cloud tops mark deep convection (storms), the part forecasters care about, and fast-changing pixels are where interpolation is hard. Plain L1 would be dominated by the easy, static background.
- **`flow_smoothness` (`:28`, × 0.01):** penalizes jagged flow fields.
- **`edge_loss` (`:34`, × 0.1):** matches image gradients, for sharper cloud edges.
- **`total_loss` (`:40`):** the sum of these; the parts are returned for TensorBoard.

#### `train.py`
- **Setup:** AdamW (lr 3e-4, weight decay 1e-4) with cosine LR decay; mixed precision (AMP) on CUDA; gradient clipping at 1.0; 20k steps at batch 8 on 256 × 256 crops.
- **`validate` (`:46`):** every 1000 steps it logs val loss and PSNR for **SatFIN and linear blending**, plus an image grid (`I0 | GT | pred | I1 | error`), to TensorBoard.
- **Checkpoints (`save` `:117`):** `best.pt` (best val PSNR) and `last.pt`, both with optimizer, scheduler and scaler state. `--resume` continues from `last.pt`.
- **`--overfit 20`:** a sanity check that it can memorize 20 triplets, which proves the pipeline can learn.
- **`merge` (`:26`):** applies the `cpu:` overrides when there is no GPU.
- **Run:** RTX 4060 Laptop, about 10.6 it/s, about 35 min in total. Best val PSNR was **44.65 dB at step 12k** (linear blending: 38.25 dB).

---

### 5.5 Evaluation

#### `baselines.py`: what SatFIN is compared against
- **`linear_blend` (`:6`):** `(1 − t)·I0 + t·I1`, a cross-fade that ignores motion.
- **`flow_interp` (`:27`):**
  - classical bidirectional optical flow, OpenCV **DIS** or **Farnebäck**
  - intermediate flows from the Super SloMo approximation: `F(t→0) = −(1−t)t·F01 + t²·F10`
  - then warp and blend

#### `metrics.py`
- **`psnr` (`:7`):** the torch version, used during training.
- **`scores` (`:13`):**
  - **PSNR:** pixel accuracy on a log scale
  - **SSIM:** structure, from scikit-image
  - **MAE in kelvin:** normalized error × 150 K
  - **Cold-cloud MAE:** MAE only on pixels where the truth is below 235 K (convection)

#### `evaluate.py`
- **`main` (`:41`):**
  - for each gap, scores `--n` evenly spaced triplets (deterministic) with all four methods
  - measures ms/frame
  - saves error-map figures
  - writes `results.md` (tables) and `results.json` (for the dashboard)
- **Options:**
  - `--gap 10`: 10-frame gaps only (9 hidden frames)
  - `--dirs "data/processed/Himawari*"`: any sequences, used for the cross-satellite test
  - `--crop 0`: full frames

---

### 5.6 Inference and outputs

#### `infer.py`
- **`load_model` (`:32`):** the model and its training config from a checkpoint.
- **`interpolate` (`:59`):** one pair of frames, any list of t values.
  - Frames larger than `tile` (1024) are split into **overlapping tiles** (128 px overlap).
  - Each tile's prediction is weighted by a **linear ramp** (`_ramp` `:41`) that fades to 0 at tile edges, then accumulated and divided by the weight sum. This blends away seams.
  - The t values are batched through the GPU.
  - A Himawari full disk (5500²) takes about 10 s per pair. A test checks that tiled output equals single-pass output.
- **`interpolate_sequence` (`:85`):** inserts *k* frames between every consecutive pair; returns the frames, their times and a flag marking interpolated frames.
- **`save_netcdf` (`:100`):** CF-style NetCDF with `bt(time, y, x)` in kelvin, `time`, an `interpolated` flag, and source metadata including the Himawari projection.
- **`main` (`:153`):**
  - input is a processed sequence (`--seq`, `--every N` to simulate low cadence) or raw files (`--files`)
  - writes NetCDF, PNG, GIF, MP4 and a side-by-side comparison
- **Cadence examples:** 30 → 7.5 min is `--k 3`; 10 → 1 min is `--k 9`.

#### `visualize.py`
- `to_u8` (`:9`): BT → 8-bit, inverted so cold clouds are bright, the meteorological convention.
- `preview` (`:14`): downscales big scenes for animations.
- `label` (`:20`): text strip on an image.
- `error_map` (`:30`): inferno colormap.
- `error_figure` (`:36`): ground truth and each method, with their error maps.
- `save_gif` / `save_mp4` (`:52` / `:57`) and `side_by_side` (`:65`).

---

### 5.7 Live pipeline: `satfin/live.py`

- **`LiveRunner.poll` (`:69`):**
  - every 30 s, lists new Himawari-9 Target files on S3
  - downloads each to a temp dir, reads it, and deletes it, so **no raw storage**
  - a bad file is logged and skipped, not retried forever
- **`complete_scans` (`:38`):** full-disk scans arrive as 10 segments, and a scan is processed only once all are present. It's tested.
- **`add` (`:91`):**
  - aligns the frame to a fixed reference grid (20 px margin, so every frame is 460 × 460)
  - interpolates 4 frames against the previous scan, giving 30-s cadence
  - restarts cleanly if the target area moves or a gap exceeds 8 min
- **`write` (`:112`) / `prune` (`:119`):** one NetCDF + PNG per frame; anything older than 6 h is deleted (about 350 MB steady state).
- **`save_state` (`:125`):** writes `state.json` atomically (temp file + rename), so the dashboard never reads a half-written file.
- **Latency:** NOAA publishes about 4–6 min after the scan, and processing takes about 1 s per scan. Interpolated frames need the *next* scan, so the smooth video trails by one interval (2.5 min).
- **Bug found in live testing:** `s3fs` caches directory listings, so new files in an already-listed folder were invisible and scans were silently skipped. Fixed with `use_listings_cache=False` (`:58`).

---

### 5.8 Dashboard: `app/dashboard.py`

A Streamlit app with three modes: held-out (with ground truth), upload, and live. Usage is in [DASHBOARD.md](DASHBOARD.md).
- **Caching:**
  - `run` (`:131`) and `artifacts` (`:137`) are cached by a cheap selection *key*
  - arguments starting with `_` are not hashed, because hashing the big frame arrays on every rerun was slow
  - result: moving the frame slider went from rebuilding everything to about 0.13 s
- **`live_view` (`:57`):** a `st.fragment(run_every=60)` that refreshes only the live panel.
- **`clip_scores` (`:227`):** scores all methods on the selected clip.
- **Safeguards:** only sequences long enough for the factor are listed, and full-disk scenes are hidden from held-out mode. The server binds to localhost (`.streamlit/config.toml`).

---

### 5.9 Tests (`tests/`, 19 tests, about 2 s)

| File | Checks |
|---|---|
| `test_data.py` | Planck round trip; normalize/denormalize; dataset shapes, ranges and t ∈ (0, 1); date split, no shared dates, platform filter |
| `test_download.py` | GOES filename → time |
| `test_loaders.py` | synthetic HSD file: segments stacked, invalid counts → NaN, **Planck forward reproduces the radiance**; Himawari time; alignment |
| `test_model.py` | warp identity and integer shift; forward pass for 1 and 3 channels, odd sizes, output in [0,1], gradients reach every parameter |
| `test_losses.py` | Charbonnier is ~0 for identical images; convective weights hit cold/fast pixels; total loss backpropagates; on a moving blob, DIS and Farneback flow interpolation beat linear blending |
| `test_infer.py` | tiled output = single pass (with a stand-in model); NetCDF round trip |
| `test_live.py` | full-disk scans wait for all 10 segments; target sub-scan ordering |

### 5.10 Scripts
- `scripts/quick_demo.sh` / `.ps1`: download → preprocess → short training → evaluate → infer. `STEPS=20 N=8` runs a 2-minute smoke test; results go to `outputs/demo/`.
- `scripts/download_goes.sh`: the default 3-h GOES sample.

---

## 6. Key numbers

| Item | Value |
|---|---|
| Data | GOES-19 mesoscale M1/M2, band 13 (10.3 µm IR), 1-min, 500 × 500 px (~2 km); 5 days (3 train, 1 val, 1 test) |
| Triplets | 6,708 train / 1,456 val / 1,456 test |
| Model | 1.14 M params (IFNet 0.65 M + fusion 0.49 M) |
| Training | 20k steps, batch 8, 256 crops, ~35 min on an RTX 4060 Laptop (8 GB); best at step 12k |
| GOES test, full frame | SatFIN **47.57 dB** / 0.424 K MAE vs DIS 44.08 dB / 0.574 K vs linear 42.12 dB |
| GOES 10-min gap | SatFIN 40.73 vs DIS 39.87 vs linear 33.70 dB |
| Himawari (never trained on) | 5-min: SatFIN **45.67** vs DIS 43.82 dB · 10-min: SatFIN 39.68 vs DIS **40.73** dB |
| Speed | ~20–70 ms per 500² frame (GPU); full disk 5500² ~10 s per pair |
| Live | Scan every 2.5 min → frame every 30 s; ~5 min scan-to-screen; 6-h window ≈ 350 MB |
| Earlier CPU-only run | 1.5k steps, 1 training day: *below* DIS everywhere |

---

## 7. Likely interview questions

**Q: What problem does this solve?**
Geostationary satellites image a region every 10–30 min, which is too coarse to follow fast-developing storms. SatFIN synthesizes the missing frames, for example 10-min → 1-min.
Operational users get smoother loops and a better sense of convective growth without new hardware.

**Q: Where does ground truth come from?**
GOES mesoscale sectors scan every **1 min**. We hide frames to *simulate* low cadence (keep every 10th frame) and score the model on the hidden real frames.
This is `dataset.py` for training and `evaluate.py` / `infer.py --every` for testing.

**Q: Why optical flow instead of predicting pixels directly?**
Clouds mostly *move*. Estimating motion and warping keeps the texture sharp, whereas regressing pixels directly blurs.
The mask handles regions visible in only one frame, and the U-Net residual fixes what motion can't explain, such as clouds growing or dissipating.

**Q: How is leakage prevented?**
The split is by **date** (`split_dirs`), with an assert that no date is in two splits. Himawari is excluded from the splits entirely (`split_platforms`), so it's a true unseen-satellite test.

**Q: Why the convective-weighted loss?**
Most pixels are clear sky that barely changes; an unweighted loss would optimize for them. Cold tops (< 235 K) and fast-changing pixels are the meteorologically important and hard cases, so they get up to 7× weight.
It shows in the results: SatFIN's largest relative gains are in **cold-cloud MAE**.

**Q: Why does SatFIN lose to DIS on Himawari at 10 min?**
That scene (a typhoon region) moves far more pixels per frame than the training data: linear blending scores only 28 dB there, against 34 dB on GOES at 10 min.
The model was trained only on 1-min GOES gaps of up to 10 frames, so very large displacements are out of distribution. It still wins on cold-cloud MAE.
The fix would be more varied training data, some Himawari, and larger gaps.

**Q: How do you handle full-disk images (5500 × 5500)?**
Tiled inference (`infer.py:59`): 1024 tiles with 128 px overlap, blended with linear ramp weights so there are no seams. A test checks that tiling equals a single pass.
Three full-disk scans take 31 s on the GPU.

**Q: How does the live system work, and do you store images?**
`live.py` polls S3 every 30 s. Raw files are downloaded to a temp dir, read and deleted, so nothing raw is stored.
Only the previous frame is kept in memory, and outputs keep a 6-h rolling window.
Interpolation needs the next scan, so output trails real time by one scan interval.

**Q: What bugs did you find, and how?**
1. A **CPU-only PyTorch on a GPU machine**, found with `env_check`. Fixing it made the full GPU training possible, turning a model that lost to DIS everywhere into one that beats it on GOES.
2. **Himawari window jitter** (5 px), found by inspecting COFF across scans. It would have been fake motion; fixed with `align`.
3. **Date-split leakage**: the Himawari data shared a date with the GOES train split. Fixed with `split_platforms`.
4. **An s3fs listing cache** that hid new files in the live loop, found by watching scans go missing. Fixed with `use_listings_cache=False`.
5. **A dashboard crash** on a 3-frame sequence; now only valid sequences are listed.

**Q: How do you know the Himawari reader is right without a library?**
- The byte offsets were checked against a real file.
- The BT range (186–298 K) is physically plausible.
- Correct alignment cut frame-to-frame differences from 7.7 K to 3.5 K.
- A unit test builds a synthetic file and checks that the Planck round trip reproduces the input radiance.

**Q: Limitations?**
- One IR band, 9 h of training data, one region and one season.
- Large motion is weak.
- No per-pixel lat/lon or GeoTIFF output.
- The model is 1.14 M params; RIFE uses ~10 M, so there's room to scale.

**Q: What would you do next?**
- More days, regions and seasons, mixing GOES and Himawari, with larger gaps and possibly a fourth coarse flow scale for big motion.
- Uncertainty or quality flags on the generated frames.
- A GeoTIFF / lat-lon export.
