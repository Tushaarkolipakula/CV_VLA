"""Visualize perturbation magnitude schedules for curriculum learning."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cv_vla.curriculum.scheduler import CurriculumScheduler


def plot_magnitude_schedule(output_path: str | Path | None = None) -> None:
    """Plot linear and cosine magnitude schedules for comparison."""
    scheduler_linear = CurriculumScheduler(total_epochs=50, initial_mag=0.1, final_mag=0.6, schedule="linear")
    scheduler_cosine = CurriculumScheduler(total_epochs=50, initial_mag=0.1, final_mag=0.6, schedule="cosine")

    epochs = np.arange(0, 51)
    linear_mag = np.array([scheduler_linear.get_avg_mag(epoch) for epoch in epochs])
    cosine_mag = np.array([scheduler_cosine.get_avg_mag(epoch) for epoch in epochs])

    plt.figure(figsize=(7, 4), dpi=180)
    plt.plot(epochs, linear_mag, label="Linear", linewidth=2.2)
    plt.plot(epochs, cosine_mag, label="Cosine", linewidth=2.2)
    plt.title("Perturbation Magnitude vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Perturbation Magnitude")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if output_path is None:
        output_path = Path("results/figures/visualize_magnitude.png")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_magnitude_schedule()
