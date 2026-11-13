"""Curriculum training loop.

Curriculum: clean demo frames are always used. Recovery frames are unlocked by
noise level as training progresses: level i of N becomes available once
progress >= (i - 1) / N * unlock_fraction, so by `unlock_fraction` of the run
every level is in. A WeightedRandomSampler keeps recovery frames at roughly
`recovery_ratio` of each epoch regardless of how many are unlocked.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.common import ACTION_DIM, PROPRIO_DIM, dumps_flips
from training.dataset_loader import CurriculumDataset, DemoIndex
from training.policy import VisionProprioPolicy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unlocked_level(epoch: int, total_epochs: int, num_levels: int, unlock_fraction: float) -> int:
    if num_levels == 0:
        return 0
    progress = epoch / max(total_epochs - 1, 1)
    span = max(unlock_fraction, 1e-6)
    return int(min(num_levels, 1 + np.floor(progress / span * num_levels)))


def make_sampler(ds: CurriculumDataset, recovery_ratio: float, num_samples: int) -> WeightedRandomSampler:
    kinds = np.array([ds.index.demos[i]["kind"] == "recovery" for i, _ in ds.samples])
    n_rec, n_clean = int(kinds.sum()), int((~kinds).sum())
    w = np.ones(len(kinds), dtype=np.float64)
    if n_rec > 0 and n_clean > 0:
        w[kinds] = recovery_ratio / n_rec
        w[~kinds] = (1 - recovery_ratio) / n_clean
    return WeightedRandomSampler(torch.from_numpy(w), num_samples=num_samples, replacement=True)


def loss_fn(pred, target):
    # translation/rotation regression + gripper as a sign target
    l_move = F.mse_loss(pred[:, :6], target[:, :6])
    l_grip = F.mse_loss(pred[:, 6], torch.sign(target[:, 6]))
    return l_move + 0.5 * l_grip


def run_epoch(model, loader, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for b in loader:
            img = b["image"].to(device, non_blocking=True)
            pro = b["proprio"].to(device, non_blocking=True)
            tid = b["task_id"].to(device, non_blocking=True)
            act = b["action"].to(device, non_blocking=True)
            loss = loss_fn(model(img, pro, tid), act)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot += loss.item() * img.size(0)
            n += img.size(0)
    return tot / max(n, 1)


def train_with_curriculum(
    dataset_dir: str,
    checkpoint_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.1,
    recovery_ratio: float = 0.3,
    unlock_fraction: float = 0.6,
    samples_per_epoch: int = 0,
    num_workers: int = 4,
    save_every: int = 10,
    seed: int = 0,
    device: str | None = None,
    clean_action_offset: int = 0,
) -> List[Dict[str, Any]]:
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = Path(checkpoint_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    index = DemoIndex(dataset_dir)
    train_ids, val_ids = index.split(val_fraction, seed)
    train_ds = CurriculumDataset(index, train_ids, clean_action_offset=clean_action_offset)
    val_ds = CurriculumDataset(index, val_ids, proprio_stats=train_ds.proprio_stats,
                               clean_action_offset=clean_action_offset) if val_ids else None
    print(f"tasks={len(index.task_names)} levels={index.num_levels} image_flips={index.image_flips}")
    print(f"train demos={len(train_ids)} val demos={len(val_ids)} device={dev}")

    model = VisionProprioPolicy(PROPRIO_DIM, ACTION_DIM, num_tasks=len(index.task_names)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    meta = {
        "task_names": index.task_names,
        "image_flips": index.image_flips,
        "image_flips_json": dumps_flips(index.image_flips),
        "proprio_mean": train_ds.proprio_stats["mean"].tolist(),
        "proprio_std": train_ds.proprio_stats["std"].tolist(),
        "proprio_dim": PROPRIO_DIM,
        "action_dim": ACTION_DIM,
        "num_levels": index.num_levels,
        "clean_action_offset": clean_action_offset,
    }

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=dev.type == "cuda",
                     persistent_workers=False, drop_last=False)
    history, best = [], float("inf")

    def save(path, epoch, val_loss):
        torch.save({"model_state_dict": model.state_dict(), "meta": meta, "epoch": epoch, "val_loss": val_loss}, path)

    for epoch in range(epochs):
        level = unlocked_level(epoch, epochs, index.num_levels, unlock_fraction)
        train_ds.set_max_level(level)
        n_samples = samples_per_epoch or len(train_ds)
        loader = DataLoader(train_ds, sampler=make_sampler(train_ds, recovery_ratio, n_samples), **loader_kw)
        t0 = time.time()
        train_loss = run_epoch(model, loader, dev, optimizer)
        sched.step()

        val_loss = None
        if val_ds is not None:
            val_ds.set_max_level(index.num_levels)  # always validate on everything
            val_loss = run_epoch(model, DataLoader(val_ds, shuffle=False, **loader_kw), dev)

        rec = {"epoch": epoch, "level": level, "counts": train_ds.count(), "loss": train_loss,
               "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)

        if val_loss is not None and val_loss < best:
            best = val_loss
            save(ckpt_dir / "checkpoint_best.pt", epoch, val_loss)
        if epoch % save_every == 0 or epoch == epochs - 1:
            save(ckpt_dir / f"checkpoint_epoch_{epoch}.pt", epoch, val_loss)
        save(ckpt_dir / "checkpoint_last.pt", epoch, val_loss)
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=1))

    train_ds.close()
    if val_ds is not None:
        val_ds.close()
    return history
