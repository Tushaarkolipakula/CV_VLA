"""Create a publication-quality flow diagram for the curriculum pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_pipeline(output_path: str | Path | None = None) -> None:
    """Render a simple flow chart describing the curriculum pipeline."""
    nodes = [
        "Expert Demonstrations",
        "Curriculum Scheduler",
        "Timestep Sampling",
        "Perturbation Generator",
        "Augmented Demonstrations",
        "Dataset Loader",
        "LeRobot Training",
        "Policy",
    ]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    y_positions = [8.5, 7.0, 5.5, 4.0, 2.5, 1.0, -0.5, -2.0]
    for index, node in enumerate(nodes):
        y = y_positions[index]
        if index == 0:
            ax.text(0.5, y, node, ha="center", va="center", fontsize=11, bbox={"boxstyle": "round", "facecolor": "#dbeafe", "edgecolor": "#1d4ed8"})
        elif index == len(nodes) - 1:
            ax.text(9.5, y, node, ha="center", va="center", fontsize=11, bbox={"boxstyle": "round", "facecolor": "#dcfce7", "edgecolor": "#16a34a"})
        else:
            ax.text(5.0, y, node, ha="center", va="center", fontsize=11, bbox={"boxstyle": "round", "facecolor": "#fef3c7", "edgecolor": "#d97706"})

        if index < len(nodes) - 1:
            next_y = y_positions[index + 1]
            ax.annotate(
                "",
                xy=(5.0, next_y + 0.2),
                xytext=(5.0, y - 0.2),
                arrowprops=dict(arrowstyle="->", color="#4b5563", lw=1.4),
            )

    ax.text(5.0, 9.4, "Curriculum Learning Pipeline", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path is None:
        output_path = Path("results/figures/visualize_pipeline.png")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_pipeline()
