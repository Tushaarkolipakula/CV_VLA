"""Visualize perturbation vectors in a simple quiver plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cv_vla.augmentation.utils import generate_random_unit_vector


def plot_noise_vectors(output_path: str | Path | None = None) -> None:
    """Show an original vector and a perturbed counterpart."""
    rng = np.random.default_rng(0)
    original = generate_random_unit_vector(dim=2, rng=rng)
    perturbation = np.array([0.35, -0.2], dtype=np.float32)
    perturbed = original + perturbation

    fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
    ax.quiver(0, 0, original[0], original[1], angles="xy", scale_units="xy", scale=1, color="#111827", label="Original")
    ax.quiver(0, 0, perturbed[0], perturbed[1], angles="xy", scale_units="xy", scale=1, color="#dc2626", label="Perturbed")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Perturbation Vector Injection")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if output_path is None:
        output_path = Path("results/figures/visualize_noise_vectors.png")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_noise_vectors()
