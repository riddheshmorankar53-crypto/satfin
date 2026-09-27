"""Train SatFIN.

python -m satfin.train                         # full run (config `train`, `cpu` overrides without a GPU)
python -m satfin.train --overfit 20 --steps 500  # sanity check: memorize 20 fixed triplets
python -m satfin.train --resume runs/<name>/last.pt
"""
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from satfin.config import DEFAULT_CONFIG, load_config
from satfin.data.dataset import TripletDataset, split_dirs
from satfin.env_check import get_device
from satfin.losses import total_loss
from satfin.metrics import psnr
from satfin.models.satfin import build_model, count_params


def merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into `base` (in place)."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def even_subset(ds: TripletDataset, n: int) -> Subset:
    """n evenly spaced items: a fixed, deterministic subset."""
    return Subset(ds, np.linspace(0, len(ds) - 1, min(n, len(ds))).round().astype(int).tolist())


def to_device(batch: dict, dev: torch.device) -> dict:
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model, batches: list[dict], bounds, loss_cfg: dict, dev) -> dict:
    """Mean val loss and PSNR of the model and of linear blending, plus an image grid."""
    model.eval()
    loss, p_model, p_lin, rows = 0.0, [], [], []
    for b in batches:
        b = to_device(b, dev)
        out = model(b["I0"], b["I1"], b["t"])
        loss += total_loss(out, b, bounds, loss_cfg)[0].item()
        t = b["t"].view(-1, 1, 1, 1)
        p_model.append(psnr(out["pred"], b["It"]))
        p_lin.append(psnr((1 - t) * b["I0"] + t * b["I1"], b["It"]))
        if len(rows) < 4:
            err = ((out["pred"] - b["It"]).abs() * 5).clamp(0, 1)
            rows.append(torch.cat([b["I0"][0], b["It"][0], out["pred"][0], b["I1"][0], err[0]], -1))
    model.train()
    return {"loss": loss / len(batches), "psnr": torch.cat(p_model).mean().item(),
            "psnr_linear": torch.cat(p_lin).mean().item(), "grid": torch.cat(rows, -2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--overfit", type=int, default=0, help="train and validate on N fixed train triplets")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--run", default=None, help="run name (default: timestamp)")
    p.add_argument("--resume", type=Path, default=None)
    a = p.parse_args()

    dev = get_device()
    ckpt = torch.load(a.resume, map_location=dev, weights_only=False) if a.resume else None
    cfg = ckpt["cfg"] if ckpt else load_config(a.config)
    if ckpt:
        a.overfit = ckpt.get("overfit", 0)
    over = cfg.pop("cpu", {})
    if dev.type == "cpu":
        merge(cfg, over)
    if a.steps:
        cfg["train"]["steps"] = a.steps
    dc, tc = cfg["data"], cfg["train"]
    bounds = (dc["bt_min"], dc["bt_max"])
    random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"])

    splits = split_dirs(dc["processed_dir"], dc["splits"], dc.get("split_platforms"))
    if a.overfit:
        train_ds = val_ds = even_subset(TripletDataset(splits["train"], dc["gaps"], dc["crop"], train=False), a.overfit)
    else:
        train_ds = TripletDataset(splits["train"], dc["gaps"], dc["crop"], train=True)
        val_ds = even_subset(TripletDataset(splits["val"], dc["gaps"], dc["crop"], train=False), tc["val_samples"])
    loader = DataLoader(train_ds, tc["batch_size"], shuffle=True, drop_last=len(train_ds) >= tc["batch_size"],
                        num_workers=tc["num_workers"], pin_memory=dev.type == "cuda",
                        persistent_workers=tc["num_workers"] > 0)
    val_batches = list(DataLoader(val_ds, tc["batch_size"]))

    model = build_model(cfg["model"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["steps"])
    use_amp = tc["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best = 0, -float("inf")
    if ckpt:
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"]); scaler.load_state_dict(ckpt["scaler"])
        step, best = ckpt["step"], ckpt["best"]
        run_dir = a.resume.parent
    else:
        run_dir = Path(tc["out_dir"]) / (a.run or datetime.now().strftime("%Y%m%d-%H%M%S") + ("-overfit" if a.overfit else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir)
    print(f"device {dev} | {count_params(model) / 1e6:.2f} M params | train {len(train_ds)} val {len(val_ds)} "
          f"| crop {dc['crop']} bs {tc['batch_size']} steps {tc['steps']} | {run_dir}")

    def save(name: str) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best": best, "cfg": cfg,
                    "overfit": a.overfit}, run_dir / name)

    model.train()
    bar = tqdm(total=tc["steps"], initial=step, desc="train")
    while step < tc["steps"]:
        for batch in loader:
            batch = to_device(batch, dev)
            with torch.autocast(dev.type, enabled=use_amp):
                out = model(batch["I0"], batch["I1"], batch["t"])
            loss, parts = total_loss({k: v.float() for k, v in out.items()}, batch, bounds, cfg["loss"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            bar.update()
            if step % tc["log_every"] == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                for k, v in parts.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/grad_norm", gnorm.item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                bar.set_postfix(loss=f"{loss.item():.4f}")
            if step % tc["val_every"] == 0 or step == tc["steps"]:
                v = validate(model, val_batches, bounds, cfg["loss"], dev)
                writer.add_scalar("val/loss", v["loss"], step)
                writer.add_scalars("val/psnr", {"satfin": v["psnr"], "linear": v["psnr_linear"]}, step)
                writer.add_image("val/I0_gt_pred_I1_err", v["grid"], step)
                if v["psnr"] > best:
                    best = v["psnr"]
                    save("best.pt")
                save("last.pt")
                tqdm.write(f"step {step}: val loss {v['loss']:.4f} | PSNR satfin {v['psnr']:.2f} dB, "
                           f"linear {v['psnr_linear']:.2f} dB | best {best:.2f}")
            if step >= tc["steps"]:
                break
    bar.close()
    writer.close()


if __name__ == "__main__":
    main()
