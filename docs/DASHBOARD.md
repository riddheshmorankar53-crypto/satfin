# How to use the SatFIN web page

## Open it

```bash
streamlit run app/dashboard.py        # then open http://localhost:8501
python -m satfin.live --ckpt runs/gpu-20k/best.pt    # only needed for the Live feed (run in a second window)
```

## The left side: pick what to look at

**Frames**: choose one of three modes.

| Mode | Use it when you want to… |
|---|---|
| **Held-out sequence** | See how good SatFIN is. It uses stored pictures where we know the real answer. |
| **Upload files** | Try SatFIN on your own satellite files. |
| **Live feed** | Watch the real satellite, updated every few minutes. |

**Upsampling factor**: how many pictures to create. Factor **N** makes **N − 1** new pictures between each real pair.
Example: factor 10 turns 10-minute pictures into 1-minute pictures.

The **Model** line just shows which trained model is in use. There's nothing to change.

## Mode 1: Held-out sequence

**Left side**
- **Sequence:** which stored recording to use. Pick the one ending in `20250610`: the model never saw that day, so the test is fair.
- **Input pairs:** how long the clip is.
- **Start frame:** where in the recording the clip starts.

**Main page, top to bottom**
1. **Animation:** three videos side by side.
   - **original**: jumpy, real pictures only.
   - **SatFIN**: smooth.
   - **ground truth**: the real answer.
   - A `*` marks pictures SatFIN made.
2. **Frame slider:** step through one picture at a time. The coloured **difference map** shows where SatFIN was wrong: black is right, yellow is off by 10 degrees.
3. **Download buttons**
   - **NetCDF:** the data, as temperatures, for scientists.
   - **MP4:** the side-by-side video.
   - **GIF:** SatFIN's pictures only.
4. **Scores for this clip:** SatFIN compared with two simple methods. A green number means SatFIN did better.
5. **Test results** (bottom): the saved full test.
   - Use **Results set** to switch between GOES and Himawari.
   - Click **Per-gap tables and error maps** for the details.

## Mode 2: Upload files

- **Drop files in the box.** Accepted: GOES `.nc` files, Himawari `.DAT` / `.DAT.bz2` files, or one `.npy` array of temperatures. You need at least 2 pictures.
- **Order doesn't matter;** they're sorted by time automatically.
- **Same page as Mode 1, but there's no real answer.** So there are no scores, and the difference map shows how SatFIN differs from a plain cross-fade, not whether it's right.

## Mode 3: Live feed

- **Four boxes** at the top:
  - **Latest scan:** the time of the newest real picture.
  - **Scan age:** how old it is. About 5–8 minutes is normal.
  - **Frames in window:** how many pictures are kept (the last 6 hours).
  - **Last poll:** when the page last checked for new pictures.
- **Video:** the last 30 minutes to 6 hours; pick with **Animation span**. Each picture is labelled **OBSERVED** (real) or **SatFIN** (made).
- **Latest observed scan:** the newest real picture, with a **Latest NetCDF** download button.
- **It refreshes by itself** every minute.

## The scores, explained

| Score | Meaning | Good is |
|---|---|---|
| **PSNR** | How close to the real picture | Higher |
| **SSIM** | How similar the shapes are (0 to 1) | Higher |
| **MAE (K)** | Average temperature mistake, in degrees | Lower |
| **Cold-cloud MAE** | The same, but only in storm areas | Lower |

In the pictures, **bright means cold, high clouds (storms)** and dark means warm ground or sea.

## If something goes wrong

| Problem | Fix |
|---|---|
| First load is slow | Normal: 5–10 seconds while it computes. After that it's instant. |
| "No processed sequence has more than N frames" | Pick a smaller upsampling factor. |
| Live feed says the loop isn't running | Start it: `python -m satfin.live --ckpt runs/gpu-20k/best.pt` |
| "No poll for N min" warning | The live loop stopped. Start it again. |
| Page looks old | Refresh the browser. |
| Can't open it from another device | It only works on this computer, on purpose. |
