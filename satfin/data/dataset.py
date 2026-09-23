"""Triplet dataset: (I0, It, I1, t) sampled from 1-min sequences with simulated gaps."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def split_dirs(processed_dir: str | Path, splits: dict[str, list[str]]) -> dict[str, list[Path]]:
    """Assign sequence folders to splits by the UTC date of their first frame."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(splits[a]) & set(splits[b]), f"splits {a}/{b} share dates"
    out: dict[str, list[Path]] = {k: [] for k in splits}
    for m in sorted(Path(processed_dir).glob("*/meta.json")):
        date = json.loads(m.read_text())["start"][:10]
        for k, dates in splits.items():
            if date in dates:
                out[k].append(m.parent)
    return out


class TripletDataset(Dataset):
    """Every (start, gap, intermediate) triplet from the given sequences.

    Gaps are in frames; a triplet is skipped if its real time span exceeds gap minutes + 30 s
    (dropped frames). t comes from real timestamps, not k/gap.
    """

    def __init__(self, seq_dirs: list[Path], gaps: list[int], crop: int, train: bool) -> None:
        self.crop, self.train = crop, train
        self.frames = [np.load(Path(d) / "frames.npy", mmap_mode="r") for d in seq_dirs]
        self.times = [np.load(Path(d) / "times.npy").astype("int64") for d in seq_dirs]  # seconds
        self.index = [
            (s, i0, k, i0 + g)
            for s, tm in enumerate(self.times)
            for g in gaps
            for i0 in range(len(tm) - g)
            if tm[i0 + g] - tm[i0] <= g * 60 + 30
            for k in range(i0 + 1, i0 + g)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s, i0, k, i1 = self.index[idx]
        tm = self.times[s]
        t = float(tm[k] - tm[i0]) / float(tm[i1] - tm[i0])
        x = np.stack([self.frames[s][i] for i in (i0, k, i1)]).astype(np.float32)  # (3,H,W)
        H, W = x.shape[1:]
        c = min(self.crop, H, W)
        if self.train:
            y0, x0 = random.randint(0, H - c), random.randint(0, W - c)
        else:
            y0, x0 = (H - c) // 2, (W - c) // 2
        x = x[:, y0:y0 + c, x0:x0 + c]
        if self.train:
            if random.random() < 0.5:
                x = x[:, :, ::-1]
            if random.random() < 0.5:
                x = x[:, ::-1, :]
            x = np.rot90(x, random.randint(0, 3), axes=(1, 2))
            if random.random() < 0.5:  # time reversal
                x, t = x[::-1], 1.0 - t
        x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1)  # (3,1,c,c)
        return {"I0": x[0], "It": x[1], "I1": x[2], "t": torch.tensor([t], dtype=torch.float32)}
