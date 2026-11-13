"""Precompute gripper-derived temporal boundary metadata for LIBERO demos.

This module is intentionally isolated from the augmentation pipeline. It only reads
existing HDF5 demonstrations, extracts the gripper signal from ``actions[:, 6]``,
identifies hard temporal boundaries, and computes the future-distance signal D(t)
for each timestep.

The output is saved as lightweight NumPy archives so the augmentation stage can load
precomputed metadata later without recomputing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import numpy as np


def find_hard_boundaries(gripper_actions: np.ndarray) -> np.ndarray:
    """Detect hard temporal boundaries from the observed gripper signal.

    We do not assume the gripper is encoded as +/-1. Any change between consecutive
    gripper values is treated as a candidate hard boundary.
    """
    gripper = np.asarray(gripper_actions, dtype=np.float64).reshape(-1)
    if gripper.size < 2:
        return np.array([], dtype=np.int64)

    boundaries = np.nonzero(gripper[1:] != gripper[:-1])[0] + 1
    return boundaries.astype(np.int64, copy=False)


def compute_future_distance(num_timesteps: int, hard_boundaries: Iterable[int]) -> np.ndarray:
    """Compute D(t) for every timestep using the next hard boundary in the future.

    D(t) is the number of timesteps from t to the next hard boundary in the future.
    If no future hard boundary exists, the end-of-trajectory index is used as the
    fallback boundary so the final timesteps decay to zero gracefully.
    """
    num_timesteps = int(num_timesteps)
    D = np.zeros(num_timesteps, dtype=np.int64)
    if num_timesteps <= 0:
        return D

    boundaries = np.asarray(list(hard_boundaries), dtype=np.int64).reshape(-1)
    if boundaries.size == 0:
        D[:] = np.arange(num_timesteps - 1, -1, -1, dtype=np.int64)
        return D

    for t in range(num_timesteps):
        future_boundaries = boundaries[boundaries > t]
        if future_boundaries.size == 0:
            D[t] = (num_timesteps - 1) - t
        else:
            D[t] = int(future_boundaries[0] - t)

    return D


def inspect_demo(demo_name: str, gripper_actions: np.ndarray, hard_boundaries: np.ndarray | None = None,
                 D: np.ndarray | None = None) -> Dict[str, Any]:
    """Return and print a compact summary for one demo."""
    gripper = np.asarray(gripper_actions, dtype=np.float64).reshape(-1)
    if hard_boundaries is None:
        hard_boundaries = find_hard_boundaries(gripper)
    if D is None:
        D = compute_future_distance(gripper.size, hard_boundaries)

    print(f"Demo: {demo_name}")
    print(f"Length: {gripper.size}")
    print(f"Gripper values: {gripper.tolist()}")
    print(f"Hard boundaries: {hard_boundaries.tolist()}")
    print(f"D(t): {D.tolist()}")

    return {
        "demo_name": demo_name,
        "length": int(gripper.size),
        "gripper_values": gripper.copy(),
        "hard_boundaries": hard_boundaries.copy(),
        "D": D.copy(),
    }


def preprocess_demo(actions: np.ndarray, demo_name: str | None = None) -> Dict[str, np.ndarray]:
    """Compute the hard-boundary and D(t) metadata for a single demo."""
    actions_array = np.asarray(actions)
    if actions_array.ndim == 1:
        actions_array = actions_array.reshape(1, -1)

    if actions_array.shape[1] <= 6:
        raise ValueError(
            f"Expected actions array with at least 7 dimensions (gripper at index 6), "
            f"but received shape {actions_array.shape}."
        )

    gripper_actions = actions_array[:, 6]
    hard_boundaries = find_hard_boundaries(gripper_actions)
    D = compute_future_distance(actions_array.shape[0], hard_boundaries)

    return {
        "gripper_values": gripper_actions.astype(np.float64, copy=False),
        "hard_boundaries": hard_boundaries,
        "D": D,
        "length": np.asarray(actions_array.shape[0], dtype=np.int64),
    }


def _sanitize_demo_name(demo_name: str) -> str:
    return demo_name.replace("/", "_").replace(" ", "_").replace("-", "_")


def save_task_preprocessing(task_name: str, demo_data: Dict[str, Dict[str, np.ndarray]], output_path: str | Path) -> Path:
    """Save a task-level preprocessing archive keyed by demo name.

    The archive stores one pair of arrays per demo:
      - <demo_name>_hard_boundaries
      - <demo_name>_D
      - <demo_name>_length
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {
        "demo_names": np.asarray(list(demo_data.keys()), dtype="U"),
    }

    for demo_name, payload in demo_data.items():
        safe_name = _sanitize_demo_name(demo_name)
        arrays[f"{safe_name}_hard_boundaries"] = np.asarray(payload["hard_boundaries"], dtype=np.int64)
        arrays[f"{safe_name}_D"] = np.asarray(payload["D"], dtype=np.int64)
        arrays[f"{safe_name}_length"] = np.asarray(payload["length"], dtype=np.int64)
        arrays[f"{safe_name}_gripper_values"] = np.asarray(payload["gripper_values"], dtype=np.float64)

    np.savez_compressed(output_file, **arrays)
    return output_file


def load_task_preprocessing(npz_path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load a task-level preprocessing archive and reconstruct the task -> demo mapping."""
    with np.load(npz_path, allow_pickle=False) as archive:
        demo_names = [str(item) for item in np.asarray(archive["demo_names"]).tolist()]
        result: Dict[str, Dict[str, np.ndarray]] = {}

        for demo_name in demo_names:
            safe_name = _sanitize_demo_name(demo_name)
            result[demo_name] = {
                "hard_boundaries": np.asarray(archive[f"{safe_name}_hard_boundaries"], dtype=np.int64),
                "D": np.asarray(archive[f"{safe_name}_D"], dtype=np.int64),
                "length": np.asarray(archive[f"{safe_name}_length"], dtype=np.int64),
                "gripper_values": np.asarray(archive[f"{safe_name}_gripper_values"], dtype=np.float64),
            }

    return result


def preprocess_task_file(task_file: str | Path, output_dir: str | Path | None = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Process a single LIBERO task HDF5 file and save per-demo metadata."""
    task_path = Path(task_file)
    with h5py.File(task_path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise KeyError(f"No 'data' group found in {task_path}")

        demo_data: Dict[str, Dict[str, np.ndarray]] = {}
        for demo_name in sorted(data_group.keys()):
            demo_group = data_group[demo_name]
            if "actions" not in demo_group:
                continue
            actions = np.asarray(demo_group["actions"][:])
            demo_output = preprocess_demo(actions, demo_name=demo_name)
            demo_data[demo_name] = demo_output

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{task_path.stem}_gripper_preprocessing.npz"
        save_task_preprocessing(task_path.stem, demo_data, out_file)

    return demo_data


def preprocess_dataset(dataset_dir: str | Path, output_dir: str | Path) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Process all task files in a LIBERO dataset directory.

    Returns a nested mapping of the form:
      {
          task_name: {
              demo_name: {
                  "hard_boundaries": ..., "D": ...
              }
          }
      }
    """
    dataset_root = Path(dataset_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_tasks: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for task_file in sorted(dataset_root.glob("*.h5")) + sorted(dataset_root.glob("*.hdf5")):
        task_name = task_file.stem
        demos = preprocess_task_file(task_file, output_root)
        all_tasks[task_name] = demos

    return all_tasks


__all__ = [
    "find_hard_boundaries",
    "compute_future_distance",
    "inspect_demo",
    "preprocess_demo",
    "save_task_preprocessing",
    "load_task_preprocessing",
    "preprocess_task_file",
    "preprocess_dataset",
]
