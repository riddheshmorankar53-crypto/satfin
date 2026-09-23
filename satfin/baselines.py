"""Classical interpolation baselines on single HxW float frames in [0,1]."""
import cv2
import numpy as np


def linear_blend(i0: np.ndarray, i1: np.ndarray, t: float) -> np.ndarray:
    """Pixel-wise linear blend in time."""
    return (1 - t) * i0 + t * i1


def _flow(a: np.ndarray, b: np.ndarray, method: str) -> np.ndarray:
    a8, b8 = [np.clip(x * 255, 0, 255).astype(np.uint8) for x in (a, b)]
    if method == "dis":
        return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a8, b8, None)
    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    raise ValueError(method)


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = img.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), xs + flow[..., 0], ys + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flow_interp(i0: np.ndarray, i1: np.ndarray, t: float, method: str = "dis") -> np.ndarray:
    """Bidirectional OpenCV flow, intermediate flows by the Super SloMo approximation, warp and blend."""
    f01, f10 = _flow(i0, i1, method), _flow(i1, i0, method)
    ft0 = -(1 - t) * t * f01 + t * t * f10
    ft1 = (1 - t) ** 2 * f01 - t * (1 - t) * f10
    return (1 - t) * _warp(i0, ft0) + t * _warp(i1, ft1)
