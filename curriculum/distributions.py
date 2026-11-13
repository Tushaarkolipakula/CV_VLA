"""Mathematical distribution helpers for curriculum scheduling."""

from __future__ import annotations

from typing import Optional

import numpy as np


def linear_progress(epoch: int, total_epochs: int) -> float:
    """Return normalized training progress in the closed interval [0, 1]."""
    if total_epochs <= 0:
        return 1.0
    return min(max(epoch / float(total_epochs), 0.0), 1.0)


def cosine_progress(epoch: int, total_epochs: int) -> float:
    """Return a cosine-based progress weight that increases smoothly."""
    progress = linear_progress(epoch, total_epochs)
    return 0.5 * (1.0 - np.cos(np.pi * progress))


def early_bias_distribution(num_samples: int) -> np.ndarray:
    """Create exponentially decaying early-bias probabilities."""
    if num_samples <= 0:
        return np.array([], dtype=np.float64)
    x = np.linspace(0.0, 1.0, num_samples)
    return np.exp(-5.0 * x)


def uniform_distribution(num_samples: int) -> np.ndarray:
    """Create a uniform distribution over the requested number of samples."""
    if num_samples <= 0:
        return np.array([], dtype=np.float64)
    return np.ones(num_samples, dtype=np.float64)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize a weight vector to sum to one."""
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return np.ones_like(weights, dtype=np.float64) / max(len(weights), 1)
    return weights / total


def interpolate_distributions(early_weights: np.ndarray, uniform_weights: np.ndarray, progress: float) -> np.ndarray:
    """Blend early-bias and uniform distributions according to curriculum progress."""
    if len(early_weights) != len(uniform_weights):
        raise ValueError("Distribution arrays must have the same length")
    weights = (1.0 - progress) * early_weights + progress * uniform_weights
    return _normalize_weights(weights)


def sample_curriculum_distribution(epoch: int, total_epochs: int, num_samples: int, schedule: str = "linear") -> np.ndarray:
    """Sample a curriculum-weighted probability distribution."""
    if schedule == "linear":
        progress = linear_progress(epoch, total_epochs)
    elif schedule == "cosine":
        progress = cosine_progress(epoch, total_epochs)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    early_weights = early_bias_distribution(num_samples)
    uniform_weights = uniform_distribution(num_samples)
    return interpolate_distributions(early_weights, uniform_weights, progress)


def sample_curriculum_timestep(
    epoch: int,
    t_min: int,
    t_max: int,
    total_epochs: int,
    schedule: str = "linear",
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample a curriculum-aware timestep from the requested range."""
    if t_min > t_max:
        raise ValueError("t_min must be less than or equal to t_max")
    timesteps = np.arange(t_min, t_max + 1)
    if len(timesteps) == 0:
        return t_min
    weights = sample_curriculum_distribution(epoch, total_epochs, len(timesteps), schedule=schedule)
    if rng is None:
        rng = np.random
    return int(rng.choice(timesteps, p=weights))
