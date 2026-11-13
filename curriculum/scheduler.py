"""Curriculum scheduling logic for perturbation magnitude and timestep selection."""

from __future__ import annotations

import numpy as np

from curriculum.distributions import (
    cosine_progress,
    linear_progress,
    sample_curriculum_distribution,
    sample_curriculum_timestep,
)


class CurriculumScheduler:
    def __init__(
        self,
        total_epochs: int,
        initial_mag: float = 0.1,
        final_mag: float = 0.678342,
        schedule: str = "linear",
    ):
        self.total_epochs = total_epochs
        self.initial_mag = initial_mag
        self.final_mag = final_mag
        self.schedule = schedule

    def progress(self, epoch: int) -> float:
        """Return normalized training progress in [0, 1]."""
        if self.schedule == "cosine":
            return cosine_progress(epoch, self.total_epochs)
        return linear_progress(epoch, self.total_epochs)

    def get_avg_mag(self, epoch: int) -> float:
        """Return the average perturbation magnitude for the requested epoch."""
        p = self.progress(epoch)
        if self.schedule == "cosine":
            return self.initial_mag + p * (self.final_mag - self.initial_mag)
        return self.initial_mag + p * (self.final_mag - self.initial_mag)

    def sample_timestep(self, epoch: int, t_min: int, t_max: int) -> int:
        """Sample a timestep according to the curriculum distribution."""
        return sample_curriculum_timestep(
            epoch=epoch,
            t_min=t_min,
            t_max=t_max,
            total_epochs=self.total_epochs,
            schedule=self.schedule,
            rng=np.random,
        )

    def get_curriculum_state(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "progress": self.progress(epoch),
            "avg_mag": self.get_avg_mag(epoch),
        }