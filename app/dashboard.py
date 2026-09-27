"""SatFIN dashboard: streamlit run app/dashboard.py"""
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from satfin.baselines import flow_interp, linear_blend  # noqa: E402
from satfin.config import load_config  # noqa: E402
from satfin.data.loaders import group_scans, read_scan  # noqa: E402
from satfin.data.preprocess import align, normalize  # noqa: E402
from satfin.infer import interpolate_sequence, load_model, save_netcdf  # noqa: E402
from satfin.metrics import scores  # noqa: E402
from satfin.visualize import error_map, label, save_gif, save_mp4, side_by_side, to_u8  # noqa: E402

st.set_page_config(page_title="SatFIN", layout="wide")
st.title("SatFIN: satellite frame interpolation")

results = sorted(ROOT.glob("outputs/**/results.json"))
evaluated = {(ROOT / json.loads(r.read_text())["meta"]["ckpt"]).resolve() for r in results}
ckpts = sorted(ROOT.glob("runs/*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
ckpts = [c for c in ckpts if c.resolve() in evaluated] or ckpts[:1]  # evaluated models; else the newest run
if not ckpts:
    st.error("Need a checkpoint in runs/*/best.pt. Run scripts/quick_demo first.")
    st.stop()
if len(ckpts) > 1:
    ckpt = st.sidebar.selectbox("Model", ckpts, format_func=lambda p: p.parent.name)
else:
    ckpt = ckpts[0]
    st.sidebar.caption(f"Model: **{ckpt.parent.name}**")
source = st.sidebar.radio("Frames", ["Held-out sequence (has 1-min truth)", "Upload files", "Live feed"])
LIVE = ROOT / "outputs" / "live"


@st.cache_data(max_entries=4)
def live_video(names: tuple[str, ...]) -> bytes:
    """Labeled MP4 of the given live frame PNGs (cached by file list, so it is rebuilt only when frames change)."""
    imgs = [label(imageio.imread(LIVE / "frames" / n), f"{n[:10]} {n[11:13]}:{n[13:15]}:{n[15:17]} UTC  "
                  + ("SatFIN" if n.endswith("_int.png") else "OBSERVED")) for n in names]
    out = Path(tempfile.mkdtemp()) / "live.mp4"
    save_mp4(imgs, out, fps=10)
    return out.read_bytes()


if source == "Live feed":
    span_min = st.sidebar.select_slider("Animation span", [30, 60, 120, 360], value=60, format_func=lambda m: f"{m} min")

    @st.fragment(run_every=60)
    def live_view() -> None:
        state_f = LIVE / "state.json"
        if not state_f.exists():
            st.info("The live loop is not running. Start it in a terminal:\n\n"
                    "`python -m satfin.live --ckpt runs/gpu-20k/best.pt`")
            return
        s = json.loads(state_f.read_text())
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        poll_age = (now - datetime.fromisoformat(s["last_poll"])).total_seconds() / 60
        st.caption(f"{s['satellite']} {s['sector']} band {s['band']}: {s['k']} SatFIN frames between scans, "
                   f"{s['window_hours']} h rolling window. Refreshes every minute.")
        if poll_age > 3:
            st.warning(f"No poll for {poll_age:.0f} min: is `python -m satfin.live` still running?")
        if s.get("last_error"):
            st.caption(f"Last error: {s['last_error']}")
        pngs = sorted(p.name for p in (LIVE / "frames").glob("*.png"))
        if not pngs:
            st.info("Waiting for the first scans...")
            return
        latest = datetime.strptime(pngs[-1][:17], "%Y-%m-%dT%H%M%S")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest scan (UTC)", s["latest_scan"][11:19] if s["latest_scan"] else "-")
        c2.metric("Scan age", f"{(now - datetime.fromisoformat(s['latest_scan'])).total_seconds() / 60:.1f} min"
                  if s["latest_scan"] else "-", help="S3 delivery (~4-6 min) + processing")
        c3.metric("Frames in window", f"{len(pngs):,}", help="observed + interpolated")
        c4.metric("Last poll", f"{poll_age * 60:.0f} s ago")
        recent = tuple(n for n in pngs if datetime.strptime(n[:17], "%Y-%m-%dT%H%M%S") >= latest - timedelta(minutes=span_min))
        l, r = st.columns([3, 2])
        l.markdown(f"**Last {span_min} min** ({len(recent)} frames, 10 fps)")
        l.video(live_video(recent), autoplay=True, loop=True, muted=True)
        obs = [n for n in pngs if n.endswith("_obs.png")]
        r.markdown("**Latest observed scan**")
        r.image(str(LIVE / "frames" / obs[-1]), caption=obs[-1][:17], width="stretch")
        r.download_button("Latest NetCDF", (LIVE / "frames" / obs[-1].replace(".png", ".nc")).read_bytes(),
                          obs[-1].replace(".png", ".nc"))

    live_view()
    st.stop()

factor =st.sidebar.select_slider("Upsampling factor", [2, 3, 4, 5, 6, 10], value=10,
                                  help="factor N inserts N-1 frames per pair, e.g. 30 min / 4 = 7.5 min")


@st.cache_resource
def model_for(path: str):
    return load_model(path)


model, cfg = model_for(str(ckpt))
dc = {**load_config()["data"], **cfg["data"]}
span = dc["bt_max"] - dc["bt_min"]
cold = (cfg["loss"]["cold_bt"] - dc["bt_min"]) / span


@st.cache_data
def read_uploads(blobs: tuple[tuple[str, bytes], ...]) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """Uploaded raw satellite files, or one .npy (T,H,W) BT stack in K, -> normalized frames, times, meta."""
    d = Path(tempfile.mkdtemp())
    for name, data in blobs:
        (d / name).write_bytes(data)
    if len(blobs) == 1 and blobs[0][0].endswith(".npy"):
        bt = np.load(d / blobs[0][0]).astype(np.float32)
        times = np.datetime64("2000-01-01T00:00:00", "s") + np.arange(len(bt)) * np.timedelta64(600, "s")
        return [normalize(b, dc["bt_min"], dc["bt_max"]) for b in bt], times, {"source": blobs[0][0]}
    scans = sorted((read_scan(g) for g in group_scans(list(d.iterdir()))),
                   key=lambda s: s[1]["time"])
    frames = [normalize(np.where(np.isnan(b), np.nanmean(b), b), dc["bt_min"], dc["bt_max"]).astype(np.float32)
              for b in align([b for b, _ in scans], [m for _, m in scans])]
    meta = {k: v for k, v in scans[0][1].items() if k != "time"}
    return frames, np.array([m["time"] for _, m in scans], "datetime64[s]"), meta


# `key` identifies the selection; `_`-prefixed args are not hashed (hashing big frame arrays every rerun is slow)
@st.cache_data(max_entries=8)
def run(ckpt_path: str, key: str, k: int, _frames: list, _times: np.ndarray) -> tuple[list, np.ndarray, np.ndarray]:
    """Cached SatFIN interpolation of a whole sequence."""
    return interpolate_sequence(model_for(ckpt_path)[0], _frames, _times, k)


@st.cache_data(max_entries=8)
def artifacts(ckpt_path: str, key: str, k: int, _seq: list, _seq_t: np.ndarray, _flag: np.ndarray,
              _cols: dict, _labels: list, _meta: dict) -> dict[str, bytes]:
    """Comparison GIF and the downloadable files, built once per selection."""
    d = Path(tempfile.mkdtemp(prefix="satfin_dash_"))
    comp = side_by_side(_cols, _labels)
    save_gif(comp, d / "comparison.gif", fps=k + 1)
    save_mp4(comp, d / "comparison.mp4", fps=k + 1)
    save_gif(_seq, d / "interpolated.gif", fps=k + 1)
    save_netcdf(d / "satfin.nc", _seq, _seq_t, _flag, (dc["bt_min"], dc["bt_max"]), _meta)
    return {f.name: f.read_bytes() for f in d.iterdir()}


gt, split = None, None
if source.startswith("Held"):
    seqs = sorted(p.parent for p in ROOT.glob("data/processed/*/frames.npy")
                  if json.loads((p.parent / "meta.json").read_text())["n_frames"] > factor)  # room for 1 pair
    if not seqs:
        st.error(f"No processed sequence has more than {factor} frames. Lower the upsampling factor.")
        st.stop()
    seq = st.sidebar.selectbox("Sequence", seqs, format_func=lambda p: p.name)
    fr = np.load(seq / "frames.npy", mmap_mode="r")
    tm = np.load(seq / "times.npy")
    n_pairs = st.sidebar.number_input("Input pairs", 1, max(1, (len(fr) - 1) // factor), 1)
    i0 = st.sidebar.number_input("Start frame", 0, len(fr) - 1 - factor * n_pairs, 0)
    keep = np.arange(i0, i0 + factor * n_pairs + 1, factor)
    key = f"{seq}|{keep.tolist()}"
    frames = [fr[i].astype(np.float32) for i in keep]
    times, meta = tm[keep], json.loads((seq / "meta.json").read_text())
    gt = [fr[i].astype(np.float32) for i in range(keep[0], keep[-1] + 1)]
    in_splits = meta.get("platform") in (cfg["data"].get("split_platforms") or [meta.get("platform")])
    split = next((k for k, d in cfg["data"]["splits"].items() if in_splits and meta["start"][:10] in d), "unassigned")
    st.caption(f"Sequence split: **{split}**" + ("" if split == "test" else " (the model may have seen this data)"))
else:
    ups = st.sidebar.file_uploader("GOES .nc, Himawari .DAT(.bz2) segments, or one (T,H,W) BT .npy in K",
                                   accept_multiple_files=True)
    if not ups:
        st.info("Upload at least two frames (any cadence) in the sidebar.")
        st.stop()
    key = hashlib.sha1(b"".join(u.name.encode() + u.getvalue() for u in ups)).hexdigest()
    try:
        frames, times, meta = read_uploads(tuple((u.name, u.getvalue()) for u in ups))
    except Exception as e:  # unreadable upload: show the reason, don't crash the app
        st.error(f"Could not read the uploads: {e}")
        st.stop()
    if len(frames) < 2:
        st.error("Need at least two frames.")
        st.stop()
    st.caption(f"{len(frames)} frames, {frames[0].shape[0]}x{frames[0].shape[1]} px, "
               f"{str(times[0])} to {str(times[-1])}")

k = factor - 1
with st.spinner("Interpolating..."):
    seq, seq_t, flag = run(str(ckpt), key, k, frames, np.asarray(times))
held = [frames[i // factor] for i in range(len(seq))]  # last real frame
labels = [f"{str(t)[11:19]}{' *' if f else ''}" for t, f in zip(seq_t, flag)]
cols = {"original": held, "SatFIN": seq} | ({"ground truth": gt} if gt is not None else {})
with st.spinner("Rendering animation and downloads..."):
    files = artifacts(str(ckpt), key, k, seq, seq_t, flag, cols, labels, meta)
st.subheader(f"Animation: original vs SatFIN{' vs ground truth' if gt is not None else ''} (* = interpolated)")
st.image(files["comparison.gif"])

st.subheader("Frame by frame")
i = st.slider("Frame", 0, len(seq) - 1, min(len(seq) - 1, k // 2 + 1))
c = st.columns(4 if gt is not None else 3)
c[0].image(to_u8(held[i]), caption=f"original (held) {labels[i]}", width="stretch")
c[1].image(to_u8(seq[i]), caption=f"SatFIN {labels[i]}", width="stretch")
if gt is not None:
    c[2].image(to_u8(gt[i]), caption="ground truth", width="stretch")
    e = np.abs(seq[i] - gt[i]) * span
    c[3].image(error_map(e, 10), caption=f"|SatFIN - truth|, 0-10 K, MAE {e.mean():.2f} K", width="stretch")
else:
    j0, t = divmod(i, factor)  # difference vs linear blend of the bracketing real frames
    lin = held[i] if not t else linear_blend(frames[j0], frames[j0 + 1], t / factor)
    e = np.abs(seq[i] - lin) * span
    c[2].image(error_map(e, 10), caption=f"|SatFIN - linear blend|, 0-10 K, mean {e.mean():.2f} K",
               width="stretch")

st.subheader("Download")
d1, d2, d3 = st.columns(3)
d1.download_button("NetCDF (BT in K, all frames)", files["satfin.nc"], "satfin_interpolated.nc")
d2.download_button("Comparison MP4", files["comparison.mp4"], "satfin_comparison.mp4")
d3.download_button("Interpolated GIF", files["interpolated.gif"], "satfin_interpolated.gif")

if gt is not None:
    st.subheader("This clip: all methods vs ground truth")

    @st.cache_data(max_entries=8)
    def clip_scores(ckpt_path: str, key: str, factor: int) -> pd.DataFrame:
        ts = [j / factor for j in range(1, factor)]
        rows = []
        for p, (a, b) in enumerate(zip(frames, frames[1:])):
            preds = {"SatFIN": seq[p * factor + 1:(p + 1) * factor],
                     "OpenCV DIS flow": [flow_interp(a, b, t, "dis") for t in ts],
                     "Linear blend": [linear_blend(a, b, t) for t in ts]}
            g = gt[p * factor + 1:(p + 1) * factor]
            rows += [{"method": m, "t": t, **dict(zip(("psnr", "ssim", "mae", "cold_mae"),
                                                     scores(np.clip(x, 0, 1).astype(np.float32), y, span, cold)))}
                     for m, xs in preds.items() for t, x, y in zip(ts, xs, g)]
        return pd.DataFrame(rows)

    df = clip_scores(str(ckpt), key, factor)
    mean = df.groupby("method", sort=False)[["psnr", "ssim", "mae", "cold_mae"]].mean()
    s, lin = mean.loc["SatFIN"], mean.loc["Linear blend"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("PSNR (dB)", f"{s.psnr:.2f}", f"{s.psnr - lin.psnr:+.2f} vs linear")
    k2.metric("SSIM", f"{s.ssim:.4f}", f"{s.ssim - lin.ssim:+.4f} vs linear")
    k3.metric("MAE (K)", f"{s.mae:.3f}", f"{s.mae - lin.mae:+.3f} vs linear", delta_color="inverse")
    k4.metric("Cold-cloud MAE (K)", "n/a" if np.isnan(s.cold_mae) else f"{s.cold_mae:.3f}",
              None if np.isnan(s.cold_mae) else f"{s.cold_mae - lin.cold_mae:+.3f} vs linear", delta_color="inverse")
    l, r = st.columns([2, 3])
    l.dataframe(mean.style.format({"psnr": "{:.2f}", "ssim": "{:.4f}", "mae": "{:.3f}", "cold_mae": "{:.3f}"}))
    r.caption("PSNR vs t, averaged over pairs (dips mid-gap, where frames are farthest from both inputs)")
    r.line_chart(df.groupby(["t", "method"]).psnr.mean().unstack(), x_label="t", y_label="PSNR (dB)")

st.subheader("Test-set evaluation")
if not results:
    st.info("Run `python -m satfin.evaluate --ckpt <ckpt>` to generate outputs/results.json.")
    st.stop()
set_names = {"outputs": "GOES test day (256 crop)", "goes_fullframe": "GOES test day (full frame)",
         "himawari": "Himawari cross-satellite (full frame)"}
res = st.selectbox("Results set", results,
                   format_func=lambda p: set_names.get(p.parent.name, p.parent.relative_to(ROOT).as_posix()))
R = json.loads(res.read_text())
M = R["meta"]
if Path(M["ckpt"]).resolve() != ckpt.resolve():
    st.warning(f"These results are for `{M['ckpt']}`, not the selected checkpoint.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Parameters", f"{M['params'] / 1e6:.2f} M")
m2.metric("Training steps", f"{M['train_steps']:,}")
m3.metric("Best val PSNR (dB)", f"{M['best_val_psnr']:.2f}")
m4.metric("Test triplets", f"{M['n_total']:,}")
st.caption(f"`{M['split']}`, gaps {M['gaps_min']} frames (1 frame = 1 min on GOES mesoscale, 2.5 min on the "
           f"Himawari target area), {M['crop']}, "
           f"cold cloud = ground-truth BT < {M['cold_bt_K']:.0f} K, timing on {M['device']}.")
names = {"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑", "mae": "MAE (K) ↓", "cold_mae": "Cold-cloud MAE (K) ↓",
         "ms": "ms/frame"}
fmt = {"PSNR (dB) ↑": "{:.2f}", "SSIM ↑": "{:.4f}", "MAE (K) ↓": "{:.3f}", "Cold-cloud MAE (K) ↓": "{:.3f}",
       "ms/frame": "{:.1f}"}
tbl = lambda d: pd.DataFrame(d).T.rename(columns=names).style.format(fmt)
st.markdown("**Overall**")
st.dataframe(tbl(R["overall"]))
gap = pd.DataFrame([{"gap": g, "method": m, **v} for g, r in R["per_gap"].items() for m, v in r.items()])
g1, g2 = st.columns(2)
g1.markdown("**PSNR by gap** (higher is better)")
g1.bar_chart(gap, x="gap", y="psnr", color="method", stack=False)
g2.markdown("**MAE by gap, K** (lower is better)")
g2.bar_chart(gap, x="gap", y="mae", color="method", stack=False)
with st.expander("Per-gap tables and error maps"):
    for g, r in R["per_gap"].items():
        st.markdown(f"**Gap {g}**")
        st.dataframe(tbl(r))
    for f in sorted((res.parent / "error_maps").glob("*.png")):
        st.image(str(f), caption=f.stem)
