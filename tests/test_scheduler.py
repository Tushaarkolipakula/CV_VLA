import numpy as np

from cv_vla.curriculum.scheduler import CurriculumScheduler


def test_scheduler_progress_and_magnitude() -> None:
    scheduler = CurriculumScheduler(total_epochs=50, initial_mag=0.1, final_mag=0.4, schedule="linear")

    assert scheduler.progress(0) == 0.0
    assert scheduler.progress(50) == 1.0
    assert np.isclose(scheduler.get_avg_mag(0), 0.1)
    assert np.isclose(scheduler.get_avg_mag(50), 0.4)


def test_scheduler_sampling_weights_are_normalized() -> None:
    scheduler = CurriculumScheduler(total_epochs=50, initial_mag=0.1, final_mag=0.4, schedule="linear")
    timesteps, weights = scheduler.get_sampling_weights(epoch=0, t_min=5, t_max=20)

    assert len(timesteps) == len(weights)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
