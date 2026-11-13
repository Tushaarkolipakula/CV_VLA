"""Visualize how curriculum timestep sampling shifts across training epochs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cv_vla.curriculum.scheduler import CurriculumScheduler


def plot_sampling_distribution(output_path: str | Path | None = None) -> None:
    """Plot the timestep sampling distribution for several curriculum epochs."""
    scheduler = CurriculumScheduler(total_epochs=50, initial_mag=0.1, final_mag=0.6, schedule="linear")
    epochs = [0, 10, 20, 30, 40, 50]
    timesteps = np.arange(5, 51)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=180, sharex=True, sharey=True)
    axes = axes.flatten()

    for axis, epoch in zip(axes, epochs):
        _, weights = scheduler.get_sampling_weights(epoch=epoch, t_min=5, t_max=50)
        axis.plot(timesteps, weights, color="#2563eb", linewidth=1.8)
        axis.set_title(f"Epoch {epoch}")
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Probability")
        axis.grid(alpha=0.25)

    fig.suptitle("Curriculum Timestep Sampling Distribution Across Epochs")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        output_path = Path("results/figures/visualize_sampling.png")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_sampling_distribution()
