"""SatFIN dashboard: streamlit run app/dashboard.py"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from satfin.baselines import flow_interp, linear_blend  # noqa: E402
from satfin.evaluate import scores  # noqa: E402
from satfin.infer import interpolate, load_model, save_gif  # noqa: E402

st.set_page_config(page_title="SatFIN", layout="wide")
st.title("SatFIN: satellite frame interpolation")

ckpts = sorted(ROOT.glob("runs/*/best.pt"))
seqs = sorted(p.parent for p in ROOT.glob("data/processed/*/frames.npy"))
if not ckpts or not seqs:
    st.error("Need a checkpoint in runs/*/best.pt and a processed sequence in data/processed/. Run scripts/quick_demo first.")
    st.stop()

ckpt = st.sidebar.selectbox("Checkpoint", ckpts, format_func=lambda p: p.parent.name)
seq = st.sidebar.selectbox("Sequence", seqs, format_func=lambda p: p.name)
frames = np.load(seq / "frames.npy", mmap_mode="r")
factor = st.sidebar.select_slider("Upsampling factor", [2, 4, 5, 10], value=10)
i0 = st.sidebar.number_input("Start frame", 0, len(frames) - 1 - factor, 0)
i1 = i0 + factor


@st.cache_resource
def model_for(path: str):
    return load_model(path)


@st.cache_data
def clip(ckpt_path: str, seq_dir: str, i0: int, factor: int):
    """Predictions of every method and per-frame scores for one clip (cached per selection)."""
    model, cfg = model_for(ckpt_path)
    dc = cfg["data"]
    span = dc["bt_max"] - dc["bt_min"]
    cold = (cfg["loss"]["cold_bt"] - dc["bt_min"]) / span
    fr = np.load(Path(seq_dir) / "frames.npy", mmap_mode="r")
    a, b = fr[i0].astype(np.float32), fr[i0 + factor].astype(np.float32)
    ts = [k / factor for k in range(1, factor)]
    gts = [fr[i].astype(np.float32) for i in range(i0 + 1, i0 + factor)]
    preds = {"SatFIN": interpolate(model, a, b, ts),
             "OpenCV DIS flow": [flow_interp(a, b, t, "dis") for t in ts],
             "Linear blend": [linear_blend(a, b, t) for t in ts]}
    rows = [{"method": m, "t": t, **dict(zip(("psnr", "ssim", "mae", "cold_mae"),
                                             scores(np.clip(p, 0, 1).astype(np.float32), g, span, cold)))}
            for m, ps in preds.items() for t, p, g in zip(ts, ps, gts)]
    return a, b, preds["SatFIN"], gts, pd.DataFrame(rows)


a, b, preds, gts, df = clip(str(ckpt), str(seq), int(i0), factor)
split = next((k for k, d in model_for(str(ckpt))[1]["data"]["splits"].items()
              if json.loads((seq / "meta.json").read_text())["start"][:10] in d), "unassigned")
st.caption(f"Sequence split: **{split}**" + ("" if split == "test" else " (the model may have seen this data)"))

tmp = Path(tempfile.gettempdir())
save_gif([a, b], tmp / "satfin_orig.gif", fps=1)
save_gif([a, *preds, b], tmp / "satfin_interp.gif", fps=factor)
save_gif([a, *gts, b], tmp / "satfin_gt.gif", fps=factor)
c1, c2, c3 = st.columns(3)
c1.subheader(f"Original ({factor}-min)")
c1.image(str(tmp / "satfin_orig.gif"))
c2.subheader("SatFIN (1-min)")
c2.image(str(tmp / "satfin_interp.gif"))
c3.subheader("Ground truth (1-min)")
c3.image(str(tmp / "satfin_gt.gif"))

st.subheader("This clip")
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
r.caption("PSNR per interpolated frame (dips mid-gap, where frames are farthest from both inputs)")
r.line_chart(df.pivot(index="t", columns="method", values="psnr"), x_label="t", y_label="PSNR (dB)")

res = ROOT / "outputs" / "results.json"
st.subheader("Test-set evaluation")
if not res.exists():
    st.info("Run `python -m satfin.evaluate --ckpt <ckpt>` to generate outputs/results.json.")
    st.stop()
R = json.loads(res.read_text())
M = R["meta"]
if Path(M["ckpt"]).resolve() != ckpt.resolve():
    st.warning(f"These results are for `{M['ckpt']}`, not the selected checkpoint.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Parameters", f"{M['params'] / 1e6:.2f} M")
m2.metric("Training steps", f"{M['train_steps']:,}")
m3.metric("Best val PSNR (dB)", f"{M['best_val_psnr']:.2f}")
m4.metric("Test triplets", f"{M['n_total']:,}")
st.caption(f"Split `{M['split']}`, gaps {M['gaps_min']} min, {M['crop']}x{M['crop']} center crop, "
           f"cold cloud = ground-truth BT < {M['cold_bt_K']:.0f} K, timing on {M['device']}.")
cols = {"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑", "mae": "MAE (K) ↓", "cold_mae": "Cold-cloud MAE (K) ↓",
        "ms": "ms/frame"}
fmt = {"PSNR (dB) ↑": "{:.2f}", "SSIM ↑": "{:.4f}", "MAE (K) ↓": "{:.3f}", "Cold-cloud MAE (K) ↓": "{:.3f}",
       "ms/frame": "{:.1f}"}
tbl = lambda d: pd.DataFrame(d).T.rename(columns=cols).style.format(fmt)
st.markdown("**Overall**")
st.dataframe(tbl(R["overall"]))
gap = pd.DataFrame([{"gap (min)": g, "method": m, **v} for g, r in R["per_gap"].items() for m, v in r.items()])
g1, g2 = st.columns(2)
g1.markdown("**PSNR by gap** (higher is better)")
g1.bar_chart(gap, x="gap (min)", y="psnr", color="method", stack=False)
g2.markdown("**MAE by gap, K** (lower is better)")
g2.bar_chart(gap, x="gap (min)", y="mae", color="method", stack=False)
with st.expander("Per-gap tables"):
    for g, r in R["per_gap"].items():
        st.markdown(f"**Gap {g} min**")
        st.dataframe(tbl(r))
