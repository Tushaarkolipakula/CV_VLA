"""Reusable augmentation helpers for curriculum dataset generation."""

from __future__ import annotations

from typing import Optional

import numpy as np


def set_reproducible_seed(seed: Optional[int] = None) -> None:
    """Seed NumPy's RNG for reproducible augmentation generation."""
    if seed is None:
        return
    np.random.seed(seed)


def generate_random_unit_vector(dim: int = 6, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Create a random unit vector of the requested dimension."""
    if rng is None:
        rng = np.random
    vector = rng.randn(dim)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    return vector / norm


def gaussian_perturbation(scale: float, dim: int = 6, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Create a Gaussian perturbation vector with the requested magnitude."""
    direction = generate_random_unit_vector(dim=dim, rng=rng)
    return direction * float(scale)


def build_perturbation_vector(
    scale: float,
    dim: int = 6,
    max_mag: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create a perturbation vector and optionally clip it to a maximum magnitude."""
    perturbation = gaussian_perturbation(scale=scale, dim=dim, rng=rng)
    if max_mag is None:
        return perturbation
    return clip_perturbation_magnitude(perturbation, max_mag=max_mag)


def build_position_rotation_perturbation(
    pos_mag: float,
    rot_mag: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create a 6D perturbation with INDEPENDENT position (m) and rotation (rad) magnitudes.

    Unlike `build_perturbation_vector`, which spreads a single scalar norm across all
    6 dims (so position and rotation compete for the same "budget"), this samples a
    random unit direction separately for the 3 position dims and the 3 rotation dims,
    so you can tune how visible the translational vs rotational displacement is
    independently.
    """
    pos_dir = generate_random_unit_vector(dim=3, rng=rng)
    rot_dir = generate_random_unit_vector(dim=3, rng=rng)
    pos_component = pos_dir * float(pos_mag)
    rot_component = rot_dir * float(rot_mag)
    return np.concatenate([pos_component, rot_component]).astype(np.float32)


def orientation_error(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    """Axis-angle error rotating current_quat -> target_quat (small-angle friendly).

    Uses robosuite's transform_utils so it matches the convention robosuite's OSC_POSE
    controller expects for the rotation part of its action.
    """
    import robosuite.utils.transform_utils as T

    quat_diff = T.quat_multiply(target_quat, T.quat_inverse(current_quat))
    return T.quat2axisangle(quat_diff)


def perturbation_to_action(perturbation: np.ndarray, gripper_value: float, action_dim: int = 7) -> np.ndarray:
    """Convert a perturbation vector into a robot action tuple."""
    perturbation = np.asarray(perturbation, dtype=np.float32).reshape(-1)
    action = np.zeros(action_dim, dtype=np.float32)
    copy_size = min(perturbation.size, action_dim - 1)
    action[:copy_size] = perturbation[:copy_size]
    action[action_dim - 1] = float(gripper_value)
    return action


def clip_perturbation_magnitude(perturbation: np.ndarray, max_mag: float) -> np.ndarray:
    """Clip a perturbation vector to a maximum norm."""
    perturbation = np.asarray(perturbation, dtype=np.float32)
    magnitude = float(np.linalg.norm(perturbation))
    if magnitude > max_mag and magnitude > 0.0:
        return perturbation * (max_mag / magnitude)
    return perturbation


def safe_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector safely when the norm is non-zero."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector)
    return vector / norm


def interpolate_value(start: float, end: float, progress: float) -> float:
    """Linearly interpolate between two scalar values."""
    return start + progress * (end - start)
