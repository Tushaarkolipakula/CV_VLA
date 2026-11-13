"""Create a simple curriculum progress dashboard for presentations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cv_vla.curriculum.scheduler import CurriculumScheduler


def plot_curriculum_dashboard(output_path: str | Path | None = None) -> None:
    """Create a two-panel figure for magnitude and progress over time."""
    scheduler = CurriculumScheduler(total_epochs=50, initial_mag=0.1, final_mag=0.6, schedule="linear")
    epochs = np.arange(0, 51)
    magnitudes = np.array([scheduler.get_avg_mag(epoch) for epoch in epochs])
    progress = np.array([scheduler.progress(epoch) for epoch in epochs])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=180, sharex=True)

    axes[0].plot(epochs, magnitudes, color="#dc2626", linewidth=2.0)
    axes[0].set_title("Curriculum Perturbation Magnitude")
    axes[0].set_ylabel("Average Magnitude")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, progress, color="#2563eb", linewidth=2.0)
    axes[1].set_title("Training Progress")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Progress")
    axes[1].grid(alpha=0.25)

    stage = f"Stage: {scheduler.schedule.title()}"
    fig.suptitle(f"Curriculum Dashboard\n{stage}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        output_path = Path("results/figures/visualize_curriculum_progress.png")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_curriculum_dashboard()
