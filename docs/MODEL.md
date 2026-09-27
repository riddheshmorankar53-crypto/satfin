# The SatFIN model, in detail

This page explains exactly how SatFIN turns two satellite pictures into the pictures between them:
what goes in, every stage inside, what comes out, how it learned, and why it was built this way.
All numbers come from the real trained model (`runs/gpu-20k/best.pt`).

Code: `satfin/models/` (`satfin.py` joins everything; `ifnet.py`, `warp.py` and `fusion.py` are the parts).

---

## 1. The big picture

```
 picture at 12:00 (I0) ─┐
 picture at 12:10 (I1) ─┼─► ① IFNet ──► movement arrows (flow) + trust map (mask)
 wanted time  t = 0.3  ─┘                  │
                                           ▼
                        ② Warp: slide I0 and I1 along the arrows to 12:03
                                           │
                                           ▼
                        ③ Blend: mix the two slid pictures using the trust map
                                           │
                                           ▼
                        ④ Fusion U-Net: small corrections (growing / fading clouds)
                                           │
                                           ▼
                             predicted picture at 12:03
```

Every stage runs inside one network, so all of it is learned together.
The design follows **RIFE** (Real-time Intermediate Flow Estimation), a well-known video frame-interpolation method,
adapted for single-channel infrared satellite images.

**Size: 1.14 million learned numbers (parameters).** That's small: RIFE itself is about 10 M.

---

## 2. What goes in and what comes out

### Inputs
| Input | Shape | Meaning |
|---|---|---|
| `I0` | 1 × H × W | First real picture |
| `I1` | 1 × H × W | Second real picture |
| `t` | one number in (0, 1) | Which moment to create: 0 = time of I0, 1 = time of I1, 0.5 = halfway |

H × W can be any size. The model was trained on 256 × 256 crops and is used on 500 × 500 scenes (and bigger, in tiles).

### How the pictures are prepared (before and inside the model)
1. **Raw sensor value → temperature.** The satellite records radiance (or raw counts for Himawari), which is
   converted to *brightness temperature* in kelvin, roughly 190–320 K for clouds and ground.
2. **Temperature → 0…1.** A fixed scale where 180 K is 0 and 330 K is 1 (`preprocess.normalize`).
   It's fixed, not per picture, so the same temperature always means the same number.
3. **Inside the model: standardize** (`SatFIN.forward`). Real scenes use only a thin slice of the 0…1 range
   (typical spread ≈ 0.03). The model subtracts the average of both pictures and divides by their spread, so the network sees clear signals of about ±1.
   *Why:* before this step, training barely learned at all. The effect is undone at the end.
4. **Pad** the edges by copying border pixels, up to a multiple of 16 (500 → 512), because the network halves the size several times.
   The padding is cut off at the end.

### Outputs (`SatFIN.forward` returns a dictionary)
| Output | Shape | Meaning |
|---|---|---|
| `pred` | 1 × H × W | **The final predicted picture**, in 0…1 (clamped) |
| `merged` | 1 × H × W | The blended picture *before* the U-Net correction (useful for debugging) |
| `flow` | 4 × H × W | Two movement fields (x and y arrows each): where to look in I0, and where to look in I1 |
| `mask` | 1 × H × W | Trust map, 0…1: how much to use I0 (1) versus I1 (0) at each pixel |

---

## 3. Stage ① IFNet: finding the movement (`models/ifnet.py`)

### The job
For every pixel of the picture we *want* (at time t), find two arrows:
- **F(t→0):** where that piece of cloud was in I0
- **F(t→1):** where it will be in I1

It also produces the **mask**: which of the two pictures to trust more at each pixel.

### Why "coarse to fine"
Clouds can move tens of pixels between pictures, and a small filter can't see that far. So IFNet looks three times, going from blurry to sharp:

| Pass | Sees the picture at | Feature grid (for 512² input) | Width | Parameters | Job |
|---|---|---|---|---|---|
| Block 1 | **¼ size** | 32 × 32 | 96 | 380 k | Big, rough movement. Large moves look small here. |
| Block 2 | **½ size** | 64 × 64 | 64 | 172 k | Correct the rough guess |
| Block 3 | **full size** | 128 × 128 | 48 | 98 k | Fine detail at cloud edges |

Each block **adds a correction** to the previous arrows instead of starting over (a residual update).
The biggest block works on the smallest picture, where computing is cheapest.

### Inside one block (`IFBlock`)
```
 input channels ─► shrink by "scale" ─► conv (stride 2) ─► conv (stride 2) ─► 4 × conv ─(+ skip)─► conv ─► 5 channels ─► enlarge back
                                        └──── ¼ of that size ────┘    └── "body" ──┘           4 flow + 1 mask
```
- **conv** = a 3 × 3 filter layer followed by PReLU, an activation that lets negative values through at a learned slope.
- The block works at 1/(4 × scale) of full resolution, then enlarges its answer back to full size.
  The flow numbers are multiplied up to match (`× scale × 4`), so the arrows are measured in full-size pixels.
- **What each block sees:**
  - Block 1: `I0, I1, t`, which is **3 channels**.
  - Blocks 2–3: `I0, I1`, both pictures **already slid** with the current arrows, `t`, the current mask, and the current arrows. That's **10 channels**.
    Seeing the slid pictures lets a block check "do they line up yet?" and fix what doesn't.

### How the time t gets in
`t` is turned into a whole extra picture filled with that one number (`IFNet.forward`), and stacked with the inputs.
So one trained model can create **any** moment (t = 0.1, 0.33, 0.9…), not just the halfway point.

---

## 4. Stage ② Warping: sliding the pictures (`models/warp.py`)

`backward_warp(image, flow)`: for each pixel *p* of the new picture, take the colour of the source picture at
*p + arrow(p)*. Between pixels it uses bilinear interpolation, a weighted mix of the 4 nearest.

```
  new picture pixel (10, 20), arrow = (+3, −1)  →  copy source pixel (13, 19)
```

- **Why "backward"** (pull the colour *from* the source) and not "forward" (push pixels *to* the target)?
  Pulling gives every new pixel exactly one value. Pushing leaves holes where nothing lands, and collisions where two pixels land in the same spot.
- **Tested:** zero arrows return the identical picture, and a shift of 2 moves it by exactly 2 pixels (`tests/test_model.py`).

The result is two candidate pictures at time t: **W0** (I0 slid forward) and **W1** (I1 slid backward).

---

## 5. Stage ③ Blending with the trust map

```
merged = mask × W0 + (1 − mask) × W1
```

The mask comes from IFNet (squashed to 0…1 with a sigmoid).
- **Why it's needed:** some areas are visible in only one picture, like a cloud that just formed, or ground that a cloud has just uncovered.
  There, the model learns to trust the picture that actually shows it.
- **Near t = 0** the mask leans toward I0; **near t = 1**, toward I1. Measured on a 10-minute test pair, the average mask is almost exactly 1 − t:

  | t | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
  |---|---|---|---|---|---|
  | average mask | 0.92 | 0.71 | 0.50 | 0.29 | 0.09 |

  The model worked this rule out by itself. Its real value is in the places where it breaks the rule, such as newly visible areas.

---

## 6. Stage ④ Fusion U-Net: final corrections (`models/fusion.py`)

Sliding only explains *movement*. Clouds also **grow, shrink, cool and warm**, and nothing slides into those changes.
A small U-Net learns a **correction** that is added to the blended picture:

```
pred = merged + correction        (correction limited to −1…+1 by tanh)
```

### What it looks at (40 channels)
- W0, W1 and merged (3)
- the mask (1) and the arrows (4)
- **context features** (32): a small 2-layer network (`ContextNet`, 16 feature maps per picture) describes the texture of I0 and I1, and those descriptions are **also slid** to time t with the same arrows.

### Its shape: shrink, then grow back, with shortcuts
| Layer | Resolution (512² input) | Width | Parameters |
|---|---|---|---|
| e1 (encoder) | 512 × 512 | 32 | 21 k |
| e2 | 256 × 256 | 64 | 56 k |
| e3 (bottom) | 128 × 128 | 128 | 222 k |
| d2 (decoder, + shortcut from e2) | 256 × 256 | 64 | 148 k |
| d1 (decoder, + shortcut from e1) | 512 × 512 | 32 | 37 k |
| output conv | 512 × 512 | 1 | 0.3 k |

Shrinking lets it see a wider area. The shortcuts ("skip connections") carry fine detail straight across, so edges stay sharp.

**Why a correction instead of a whole new picture:** the slid-and-blended picture is already mostly right.
Learning only the small difference is much easier than redrawing everything.

---

## 7. Finishing

1. Undo the standardization from step 3 of section 2 (multiply the spread back, add the average back).
2. Cut off the padding (512 → 500).
3. Clamp to 0…1.
4. Outside the model: convert back to kelvin (`denormalize`: × 150 + 180) for NetCDF files and error numbers.

---

## 8. Where the 1.14 million parameters are

| Part | Parameters | Share |
|---|---|---|
| IFNet block 1 (¼ size) | 379,925 | 33 % |
| IFNet block 2 (½ size) | 172,357 | 15 % |
| IFNet block 3 (full size) | 98,165 | 9 % |
| Fusion U-Net | 483,105 | 43 % |
| Context network | 2,512 | < 1 % |
| **Total** | **1,136,064** | |

The settings live in `configs/default.yaml` → `model:` (`ifnet_widths: [96, 64, 48]`, `scales: [4, 2, 1]`, `fusion_width: 32`, `channels: 1`).
Setting `channels: N` makes the model accept N satellite bands at once. The model supports it and it's tested, but the data pipeline currently prepares one band.

---

## 9. How it learned

### What it practised on
- **Data:** GOES-19 takes a picture **every minute**. From 3 days (9 hours) of those, we made **6,708 practice questions**, each: "here is minute 0 and minute N, guess minute k."
- **Gaps:** N is 2, 4 or 10 minutes, and `t = k / N` comes from the real timestamps.
- **Augmentation:** each question is randomly cropped to 256 × 256, flipped, rotated, or played **backwards** (swap I0 and I1, and t becomes 1 − t). This makes the model see more variety.

### How a guess is scored (the loss, `losses.py`)
```
loss =  weighted Charbonnier(pred, truth)
      + 0.01 × flow smoothness
      + 0.1  × edge loss
```
| Part | What it does | Why |
|---|---|---|
| **Charbonnier** | ≈ absolute error, smoothed near 0 | Handles outliers well and doesn't blur like squared error |
| **Weights** | each pixel counts `1 + 4·[colder than 235 K] + 2·[changed > 5 K]`, up to 7× | Storm tops are cold, and fast-changing pixels are the hard part. Without weights, calm sky would dominate. |
| **Flow smoothness** | penalizes jagged arrows | Neighbouring cloud pieces usually move together |
| **Edge loss** | matches the picture's edges (gradients) | Keeps cloud edges sharp |

### Training settings (`train.py`)
| Setting | Value |
|---|---|
| Optimizer | AdamW, learning rate 3e-4, weight decay 1e-4 |
| Schedule | Cosine: the learning rate slowly decays to 0 |
| Steps / batch | 20,000 steps × 8 questions |
| Speed tricks | Mixed precision (16-bit maths on GPU), gradient clipping at 1.0 for stability |
| Checks | Every 1,000 steps, on 256 fixed questions from a separate day; the best version is kept |
| Hardware / time | RTX 4060 Laptop GPU, ~10.6 steps/s, ~35 min |
| Best result | Step 12,000: validation PSNR **44.65 dB** (cross-fade: 38.25 dB) |

**Sanity check first:** before the real run, the model was trained on just 20 fixed questions until it memorized them (`--overfit 20`), which proves the whole pipeline can learn.

---

## 10. Using the trained model

- **Any moment:** ask for t = 0.1, 0.2, … 0.9 to turn a 10-minute gap into 1-minute steps. All the t values run together in one batch.
- **Big pictures:** a 5500 × 5500 full-disk image is cut into 1024 × 1024 tiles that overlap by 128 pixels.
  Each tile's result fades out toward its edges and the overlaps are averaged, so no seams show (`infer.interpolate`).
- **Speed:** about 20–70 ms per 500 × 500 picture on the GPU, and about 10 s per pair for a full disk.

---

## 11. Why these choices

| Choice | Reason | Alternative considered |
|---|---|---|
| Estimate movement, then slide | Clouds mostly move; sliding keeps their texture sharp | Predicting pixels directly: blurry results |
| Arrows from time t (not 0→1) | We directly get what's needed to build the picture at t | Estimate 0→1 motion and approximate t (the OpenCV baseline does this, and is less accurate) |
| Coarse-to-fine, 3 passes | Catches big moves cheaply, then refines | A single pass at full size: misses big moves, costs more |
| Trust mask | Handles areas visible in only one picture | Plain 50/50 mix: ghosting |
| U-Net correction | Clouds grow and fade, which sliding can't explain | No correction: stuck with pure motion |
| Standardize inside the model | Satellite pictures are low-contrast; the network needs clear signals | Per-picture 0…1 stretch: breaks the fixed temperature meaning |
| Small model (1.1 M) | Fits an 8 GB laptop GPU, trains in 35 min, runs live | Bigger model: more accurate on big motion, but slower and needs more data |

---

## 12. What it's good and bad at

**Good**
- On a GOES day it never saw, it beats classic motion tracking (OpenCV DIS): **47.6 vs 44.1 dB**.
- It is most accurate on storm tops, the area the loss weighted.
- It works on another satellite (Himawari) it was never trained on, and wins at 5-minute gaps (45.7 vs 43.8 dB).

**Weak**
- **Very large motion:** on Himawari 10-minute gaps it loses to DIS (39.7 vs 40.7 dB). The clouds moved further than in any practice question.
- **Brand-new clouds** appearing between two pictures can only be approximated, because neither input shows them.
- **Only seen:** one infrared band, 9 hours of training data, one region and one season.

**How to improve it:** more days, regions and seasons; some Himawari practice data; bigger gaps in practice; possibly a fourth, even coarser pass for big motion; a larger model.

---

## 13. One example, start to finish

Two pictures at **12:00** and **12:10**; we want **12:03** (t = 0.3).

1. Both pictures are converted to 0…1 temperatures, standardized, and padded from 500 to 512.
2. **IFNet block 1** looks at ¼-size copies and finds that the storm moved about 12 pixels to the east.
   Block 2 at ½ size refines the arrows around the storm edge; block 3 at full size fixes small details.
3. **Warp:** I0 is slid 30 % of the way forward (≈ 3.6 px east) and I1 is slid 70 % of the way back, so both now show the storm where it was at 12:03.
4. **Blend:** where a new cloud appears only in I1, the mask trusts W1; elsewhere it mixes both, with more weight on I0 since 12:03 is closer to 12:00.
5. **U-Net:** adds a small cooling to the storm top, which was growing.
6. Undo the standardization, cut back to 500 × 500, clamp: that's the predicted 12:03 picture.
   On the GOES test day, the average error at 10-minute gaps is about 0.8 K (cross-fade: 1.75 K).

(The 12-pixel movement in this example is an illustration, not a measured value.)
