"""Frames to images, animations (GIF/MP4) and error-map figures. Frames are HxW normalized BT in [0,1]."""
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def to_u8(x: np.ndarray) -> np.ndarray:
    """Normalized BT to 8-bit, inverted so cold cloud tops are bright."""
    return ((1 - np.clip(x, 0, 1)) * 255).astype(np.uint8)


def preview(x: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Downscale (area averaging) so the longest side is at most max_side; for animations of large scenes."""
    f = max_side / max(x.shape[:2])
    return x if f >= 1 else cv2.resize(x, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Image (HxW or HxWx3 uint8) with a text strip on top."""
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    scale = max(img.shape[1] / 500, 0.4)
    strip = np.zeros((int(28 * scale), img.shape[1], 3), np.uint8)
    cv2.putText(strip, text, (4, int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255),
                max(1, int(scale)), cv2.LINE_AA)
    return np.vstack([strip, img])


def error_map(err_K: np.ndarray, vmax_K: float) -> np.ndarray:
    """|error| in kelvin -> RGB (inferno, 0..vmax_K)."""
    u8 = (np.clip(err_K / vmax_K, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def error_figure(gt: np.ndarray, preds: dict[str, np.ndarray], span: float, path: Path,
                 vmax_K: float = 10.0, title: str = "") -> None:
    """Top row: ground truth and each prediction. Bottom row: |pred - gt| in K (inferno, 0..vmax_K)."""
    top = [label(to_u8(gt), f"ground truth {title}")]
    bot = [label(np.zeros((*gt.shape, 3), np.uint8), f"|error| K, 0-{vmax_K:g}")]
    for name, p in preds.items():
        e = np.abs(p - gt) * span
        top.append(label(to_u8(p), name))
        bot.append(label(error_map(e, vmax_K), f"{name}: MAE {e.mean():.2f} K"))
    imageio.imwrite(path, np.vstack([np.hstack(top), np.hstack(bot)]))


def _rgb(f: np.ndarray) -> np.ndarray:
    return f if f.ndim == 3 else cv2.cvtColor(to_u8(f), cv2.COLOR_GRAY2RGB)


def save_gif(frames: list[np.ndarray], path: Path, fps: float = 5) -> None:
    """Normalized frames (or RGB uint8 images) to a looping GIF."""
    imageio.mimsave(path, [_rgb(f) for f in frames], duration=1 / fps, loop=0)


def save_mp4(frames: list[np.ndarray], path: Path, fps: float = 5) -> None:
    """Normalized frames (or RGB uint8 images) to H.264 MP4 (padded to even size)."""
    imgs = [_rgb(f) for f in frames]
    h, w = imgs[0].shape[:2]
    imgs = [np.pad(i, ((0, h % 2), (0, w % 2), (0, 0))) for i in imgs]
    imageio.mimsave(path, imgs, fps=fps, codec="libx264", macro_block_size=1, quality=8)


def side_by_side(columns: dict[str, list[np.ndarray]], labels: list[str] | None = None) -> list[np.ndarray]:
    """Equal-length frame lists -> one labeled RGB frame per step, columns left to right."""
    n = len(next(iter(columns.values())))
    out = []
    for i in range(n):
        row = [label(to_u8(c[i]), f"{name}" + (f"  {labels[i]}" if labels else "")) for name, c in columns.items()]
        out.append(np.hstack(row))
    return out
