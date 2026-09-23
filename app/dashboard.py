"""SatFIN dashboard: streamlit run app/dashboard.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
    return load_model(path)[0]


a, b = frames[i0].astype(np.float32), frames[i1].astype(np.float32)
ts = [k / factor for k in range(1, factor)]
preds = interpolate(model_for(str(ckpt)), a, b, ts)
gts = [frames[i].astype(np.float32) for i in range(i0 + 1, i1)]

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

span = 150.0  # bt_max - bt_min from configs/default.yaml
m = np.mean([scores(p, g, span) for p, g in zip(preds, gts)], 0)
k1, k2, k3 = st.columns(3)
k1.metric("PSNR (dB)", f"{m[0]:.2f}")
k2.metric("SSIM", f"{m[1]:.4f}")
k3.metric("MAE (K)", f"{m[2]:.3f}")
res = ROOT / "outputs" / "results.md"
if res.exists():
    st.subheader("Test-set results")
    st.markdown(res.read_text(encoding="utf-8"))
