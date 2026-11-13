"""Single source of truth for observation layout.

Dataset generation, training and evaluation all import from here, so the
policy sees images and proprioception in exactly the same layout in all three.

Image convention
----------------
Images are stored and fed to the model in the SAME orientation as the original
LIBERO .hdf5 files. Nothing in training ever flips an image. The generator
measures which flip (normally none) maps a raw env camera frame onto the
dataset frame by rendering a dataset state and comparing pixels, writes it into
the output files, training copies it into the checkpoint, and evaluation
applies that same flip to live env frames. That one stored value is the only
place orientation is decided.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# (dataset obs key, raw env obs key)
CAMERAS: Tuple[Tuple[str, str], ...] = (
    ("agentview_rgb", "agentview_image"),
    ("eye_in_hand_rgb", "robot0_eye_in_hand_image"),
)

# proprio = gripper_qpos(2) + eef_pos(3) + eef_axis_angle(3)
PROPRIO_DIM = 8
ACTION_DIM = 7

FLIP_MODES = ("none", "ud", "lr", "rot180")


def apply_flip(img: np.ndarray, mode: str) -> np.ndarray:
    """Flip an (..., H, W, C) image. Returns a contiguous copy."""
    if mode == "none":
        out = img
    elif mode == "ud":
        out = np.flip(img, axis=-3)
    elif mode == "lr":
        out = np.flip(img, axis=-2)
    elif mode == "rot180":
        out = np.flip(img, axis=(-3, -2))
    else:
        raise ValueError(f"unknown flip mode {mode!r}")
    return np.ascontiguousarray(out)


def detect_flip(env_img: np.ndarray, dataset_img: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Return the flip that makes env_img look most like dataset_img (mean abs pixel error)."""
    a = env_img.astype(np.float32)
    b = dataset_img.astype(np.float32)
    if a.shape != b.shape:
        raise ValueError(f"env image {a.shape} vs dataset image {b.shape}: set camera size to the dataset size")
    errs = {m: float(np.abs(apply_flip(a, m) - b).mean()) for m in FLIP_MODES}
    return min(errs, key=errs.get), errs


def quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Same formula as robosuite.utils.transform_utils.quat2axisangle (xyzw input)."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return q[:3] * 2.0 * np.arccos(w) / den


def proprio_from_env_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        ]
    )


def proprio_from_dataset(obs_grp, idx) -> np.ndarray:
    """obs_grp is an h5py group with gripper_states / ee_pos / ee_ori. idx may be an int or slice."""
    return np.concatenate(
        [
            np.asarray(obs_grp["gripper_states"][idx], dtype=np.float32),
            np.asarray(obs_grp["ee_pos"][idx], dtype=np.float32),
            np.asarray(obs_grp["ee_ori"][idx], dtype=np.float32),
        ],
        axis=-1,
    )


def env_obs_to_images(obs: Dict[str, np.ndarray], flips: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Raw env camera frames -> dataset orientation, keyed by dataset obs key."""
    return {ds_key: apply_flip(np.asarray(obs[env_key]), flips[ds_key]) for ds_key, env_key in CAMERAS}


def task_name_from_file(path) -> str:
    stem = Path(path).stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem


def dumps_flips(flips: Dict[str, str]) -> str:
    return json.dumps(flips, sort_keys=True)


def loads_flips(s) -> Dict[str, str]:
    if isinstance(s, bytes):
        s = s.decode()
    return json.loads(s)
