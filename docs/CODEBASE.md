# SatFIN in plain words

## What does it do?

Weather satellites take a picture of the clouds every **10 minutes** or so. Storms change a lot in 10 minutes,
so the picture sequence looks jumpy, like a flipbook with missing pages.

**SatFIN draws the missing pages.** It looks at two pictures, works out how the clouds moved,
and creates the pictures in between. Ten-minute pictures become one-minute pictures.

```
  real          ?      ?      ?          real
 12:00  ───►  12:01  12:02  12:03 … ───► 12:10
              └──── SatFIN fills these in ────┘
```

## How does it work? (5 steps)

1. **Download** satellite pictures from the internet (free NOAA archive).
2. **Clean** them: turn raw sensor numbers into temperatures, then into values between 0 and 1.
3. **Teach** the model: show it pictures where we *know* the in-between frames (one satellite takes a picture every minute),
   hide some, and let it practise guessing them.
4. **Test** it on days it has never seen, against two simple methods, and measure how close it gets.
5. **Use** it on new pictures, including a **live** feed that updates every few minutes.

## How does the model guess a picture? (3 ideas)

1. **Find the movement.** For every pixel, estimate which way the cloud moved (this is called *optical flow*).
2. **Slide the pictures.** Move both real pictures along that movement to the in-between moment, then mix them.
3. **Touch up.** A small second network fixes spots the movement can't explain, like clouds that grow or disappear.

## The two satellites

| Satellite | What we use it for |
|---|---|
| **GOES-19** (over the Americas) | Teaching and testing. It takes a picture **every minute**, so we always have the right answer. |
| **Himawari-9** (over Asia and the Pacific) | A test on a satellite the model never saw, and the **live** feed (new picture every 2.5 minutes). |

## The files: what is where

Everything is in four groups, in the order the pipeline runs.

**1. Get the data** (`satfin/data/`)
| File | In one line |
|---|---|
| `download.py` | Downloads satellite files from the internet |
| `loaders/goes.py` | Reads a GOES file and turns it into temperatures |
| `loaders/himawari.py` | Reads a Himawari file and turns it into temperatures |
| `preprocess.py` | Cleans the pictures and saves them as ready-to-use sequences |
| `dataset.py` | Cuts sequences into practice questions: "here are 2 pictures, guess the middle one" |

**2. The model and teaching it**
| File | In one line |
|---|---|
| `models/ifnet.py` | Finds how the clouds moved |
| `models/warp.py` | Slides a picture along that movement |
| `models/fusion.py` | The touch-up network |
| `models/satfin.py` | Joins the three parts into one model |
| `losses.py` | Scores each guess during practice, and counts storm areas extra |
| `train.py` | The practice loop: guess, check, improve, repeat 20,000 times |

**3. Check how good it is**
| File | In one line |
|---|---|
| `baselines.py` | Two simple methods to compare against: cross-fade, and classic motion-tracking (OpenCV) |
| `metrics.py` | The scores: PSNR, SSIM, error in degrees |
| `evaluate.py` | Runs the full test and writes the results table |

**4. Use it**
| File | In one line |
|---|---|
| `infer.py` | Fills in frames for any picture sequence and saves the results |
| `live.py` | Watches the live satellite feed and fills in frames as new pictures arrive |
| `visualize.py` | Makes images, GIFs and videos |
| `app/dashboard.py` | The web page for using all of the above (see [DASHBOARD.md](DASHBOARD.md)) |

**Other**
| File / folder | In one line |
|---|---|
| `configs/default.yaml` | Every setting in one place (paths, sizes, training length) |
| `tests/` | Automatic checks that the code works (run: `python -m pytest`) |
| `scripts/quick_demo.sh` | Runs the whole pipeline on a small sample |

## Where do I find…?

| Question | File → function |
|---|---|
| How raw numbers become temperatures | `loaders/goes.py` → `rad_to_bt` |
| How the practice questions are made | `data/dataset.py` → `TripletDataset` |
| How test days are kept separate from practice days | `data/dataset.py` → `split_dirs` |
| The model | `models/satfin.py` → `SatFIN` |
| Where the "in-between time" goes into the model | `models/ifnet.py` → `IFNet.forward` |
| How guesses are scored in practice | `losses.py` → `total_loss` |
| Why storms count extra | `losses.py` → `convective_weight` |
| Training settings | `configs/default.yaml` → `train:` |
| How huge pictures are handled | `infer.py` → `interpolate` (cuts into overlapping tiles) |
| How the live feed works | `live.py` → `LiveRunner` |

## Results in plain words

The scores are PSNR in dB: higher means closer to the real picture.

| Test | SatFIN | Classic motion-tracking | Cross-fade |
|---|---|---|---|
| GOES, a day it never saw | **47.6** | 44.1 | 42.1 |
| Himawari, 5-minute gaps | **45.7** | 43.8 | 31.5 |
| Himawari, 10-minute gaps | 39.7 | **40.7** | 28.2 |

- **SatFIN wins** on GOES, and on Himawari with short gaps.
- **It loses** on Himawari with long gaps, where clouds moved much further than anything it practised on.
- **It is fast:** about 1 second per new live picture.

## Words you will hear

| Word | Meaning |
|---|---|
| **Geostationary satellite** | A satellite that stays over the same spot and keeps photographing it |
| **Cadence** | How often a picture is taken (every 1, 2.5, or 10 minutes) |
| **Brightness temperature** | The cloud-top temperature the satellite measures, in kelvin (K). Colder means higher, stormier clouds. |
| **Optical flow** | An arrow for every pixel showing where it moved |
| **Interpolation** | Creating the frames in between two real ones |
| **Triplet** | One practice question: first picture, hidden middle picture, last picture |
| **PSNR** | Similarity to the real picture. Higher is better; +3 is a clear improvement. |
| **MAE (K)** | Average temperature error in kelvin. Lower is better. |
| **Checkpoint** | The saved, trained model (`runs/gpu-20k/best.pt`) |

## Likely interview questions (short answers)

- **Why is this useful?** Storms grow fast. Smoother loops help forecasters see them develop, without launching a new satellite.
- **How do you know the right answer?** GOES takes a picture every minute. We hide frames, guess them, and compare with the real ones.
- **Why not just cross-fade?** A cross-fade makes ghostly double images. Clouds *move*, so we track the movement.
- **How do you stop cheating in the test?** Practice and test use **different days**, and Himawari is never used for practice at all.
- **Why count storms extra?** Most of the picture is calm sky. Without extra weight the model would ignore the storms, which are the part that matters.
- **What was hard?** A computer set up without its graphics card, a Himawari picture frame that shifts 5 pixels between shots, and a live-download cache that hid new files. Each was found by testing and then fixed.
- **Weakness?** Very fast motion over long gaps. The fix is more varied practice data.
- **Do you store the live pictures?** No. Each one is downloaded, used, then deleted. Only the last 6 hours of results are kept.
