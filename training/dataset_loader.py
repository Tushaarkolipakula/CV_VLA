"""Dataset over generated files: clean demo frames + recovery frames, split by demo.

Each sample:
    image   uint8 tensor (2*3, H, W)  agentview + wrist, dataset orientation, never flipped here
    proprio float32 (8,)               normalized with stats from the train split
    task_id int64
    action  float32 (7,)
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from training.common import CAMERAS, loads_flips, proprio_from_dataset, task_name_from_file


class DemoIndex:
    """Lists every demo across files with its kind / level / length."""

    def __init__(self, dataset_dir: str):
        self.files = sorted(glob.glob(os.path.join(os.path.expanduser(dataset_dir), "*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"No .hdf5 files in {dataset_dir}")
        self.task_names = sorted({task_name_from_file(f) for f in self.files})
        self.demos: List[Dict] = []
        self.image_flips = None
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                if "image_flips" not in f["data"].attrs:
                    raise RuntimeError(f"{path} has no image_flips attr: regenerate it with the new generator")
                flips = loads_flips(f["data"].attrs["image_flips"])
                if self.image_flips is None:
                    self.image_flips = flips
                elif flips != self.image_flips:
                    raise RuntimeError(f"Inconsistent image_flips across files: {flips} vs {self.image_flips}")
                tid = self.task_names.index(task_name_from_file(path))
                for k in f["data"].keys():
                    g = f["data"][k]
                    self.demos.append(
                        {
                            "file": fi,
                            "key": k,
                            "task_id": tid,
                            "kind": str(g.attrs.get("kind", "clean")),
                            "level": int(g.attrs.get("noise_level", 0)),
                            "source": str(g.attrs.get("source_demo", k)),
                            "length": int(g["actions"].shape[0]),
                        }
                    )
        self.num_levels = max([d["level"] for d in self.demos] + [0])

    def split(self, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
        """Split by source demo so a recovery sample never lands in the other split from its clean demo."""
        groups = sorted({(d["file"], d["source"]) for d in self.demos})
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        n_val = int(round(len(groups) * val_fraction)) if val_fraction > 0 else 0
        val_groups = set(groups[:n_val])
        train, val = [], []
        for i, d in enumerate(self.demos):
            (val if (d["file"], d["source"]) in val_groups else train).append(i)
        return train, val


class CurriculumDataset(Dataset):
    def __init__(self, index: DemoIndex, demo_ids: List[int], proprio_stats: Optional[Dict[str, np.ndarray]] = None,
                 clean_action_offset: int = 0):
        if clean_action_offset not in (0, 1):
            raise ValueError("clean_action_offset must be 0 or 1")
        self.clean_action_offset = clean_action_offset
        self.index = index
        self.demo_ids = list(demo_ids)
        self._handles: Dict[Tuple[int, int], h5py.File] = {}
        self.proprio_stats = proprio_stats or self.compute_proprio_stats()
        self.set_max_level(index.num_levels)

    def compute_proprio_stats(self) -> Dict[str, np.ndarray]:
        vals = []
        for i in self.demo_ids:
            d = self.index.demos[i]
            if d["kind"] != "clean":
                continue
            vals.append(proprio_from_dataset(self._file(d["file"])["data"][d["key"]]["obs"], slice(None)))
        allv = np.concatenate(vals, axis=0)
        return {"mean": allv.mean(0).astype(np.float32), "std": (allv.std(0) + 1e-6).astype(np.float32)}

    def set_max_level(self, max_level: int):
        """Curriculum hook: only clean frames and recovery frames with level <= max_level are used."""
        self.max_level = max_level
        self.samples = []
        for i in self.demo_ids:
            d = self.index.demos[i]
            if d["kind"] == "clean" or d["level"] <= max_level:
                # Original LIBERO exports may store post-action observations.
                # For these, pair obs[t] with actions[t+1]; recovery data already
                # stores pre-action observations and must retain offset zero.
                offset = self.clean_action_offset if d["kind"] == "clean" else 0
                self.samples.extend((i, t) for t in range(max(0, d["length"] - offset)))

    def count(self) -> Dict[str, int]:
        c = {"clean": 0, "recovery": 0}
        for i, _ in self.samples:
            c[self.index.demos[i]["kind"]] += 1
        return c

    def _file(self, fi: int) -> h5py.File:
        key = (os.getpid(), fi)
        if key not in self._handles:
            self._handles[key] = h5py.File(self.index.files[fi], "r")
        return self._handles[key]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        di, t = self.samples[idx]
        d = self.index.demos[di]
        g = self._file(d["file"])["data"][d["key"]]
        obs = g["obs"]
        image = np.concatenate([obs[ds_key][t] for ds_key, _ in CAMERAS], axis=-1)  # H, W, 6
        proprio = (proprio_from_dataset(obs, t) - self.proprio_stats["mean"]) / self.proprio_stats["std"]
        offset = self.clean_action_offset if d["kind"] == "clean" else 0
        action = np.asarray(g["actions"][t + offset], dtype=np.float32)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "proprio": torch.from_numpy(proprio.astype(np.float32)),
            "task_id": torch.tensor(d["task_id"], dtype=torch.long),
            "action": torch.from_numpy(action),
        }

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_handles"] = {}
        return s

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles = {}
