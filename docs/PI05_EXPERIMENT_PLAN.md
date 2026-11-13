# π0.5 initial-state robustness experiment

Prepared 13 November 2026. This is an experiment plan and implementation record, not a completed-results report. A 70–80% success rate is a target, not a guarantee.

## Question and scope

Does progressively introducing independently generated robot starting-pose perturbations improve π0.5 success on LIBERO+ **Robot Initial States**, compared with clean fine-tuning and a constant mixture of perturbation strengths?

This version changes the perturbation timing from the earlier contact-triggered SmolVLA experiment to **episode initialization**, matching the requested evaluation axis. It does not claim to evaluate robustness to disturbances at contact. Both are useful experiments, but they answer different questions.

## Data

- Preserve the original 2,000 LIBERO demonstrations across spatial, object, goal and long-horizon (`libero_10`) suites. Do not train on LIBERO+ test initial states.
- Before augmentation, replay each source demonstration in the simulator. Reject that source for augmentation if its clean replay does not succeed.
- Perturb the seven arm joints independently. At level L, each offset is drawn uniformly from −0.1L to +0.1L radians, for L = 1…5. These are ranges; the robot does not always receive the same offset or direction.
- Reject joint-limit violations and newly increased initial contact penetration exceeding 2 mm relative to the unperturbed state. This is an initial collision check, not a general proof of collision-free motion.
- Use an end-effector pose controller to return toward the original starting pose, then replay the expert actions. Accept only examples whose full simulated task succeeds. Keep rejection and acceptance reports, including difficulty and displacement.
- Store the corrective window plus up to 49 following frames, sufficient for 50-action chunk targets. Full task success is checked before truncating storage. Sample recovery observations only within the corrective window.
- Keep all descendants of a source demonstration in the same train/validation split. Split 90/10 by source demonstration within each task. Compute state/action quantile statistics from clean training demonstrations only.
- Before augmented training, require at least three accepted training recoveries per task per level. Audit coverage, splits, finite bounded actions and observation/action lengths. Insufficient coverage stops the queue for diagnosis.

The generated data lives at `/home/tushaar/cv_vla/data/initial_joint_v2_20261113`. Originals remain under `/home/tushaar/data/LIBERO-datasets`. The old generated contact HDF5 files were deleted after their reports were archived under `results/pi05_v2_audit/previous_dataset`; prior model checkpoints and evaluation results were preserved.

## Three independent training runs

All three runs initialize from the same installed `lerobot/pi05_base` snapshot, with the same source split and seed. They do not initialize from one another.

| Run | Sampling |
|---|---|
| Clean | 100% clean demonstration observations |
| Fixed | 90% clean, 10% recovery; all five levels available throughout |
| Curriculum | 90% clean, 10% recovery; unlock levels 1 → 5 over the first 70% of optimizer updates, retain all five thereafter |

Recovery selection balances tasks, then available levels, then recovery trajectories. Fixed means a constant sampling distribution, not a constant perturbation vector. The common 10% recovery probability is a conservative starting hypothesis intended to reduce the clean-task degradation seen in the previous experiment; its benefit is unproven. Curriculum necessarily changes cumulative exposure to each level, so a paper should include a matched-exposure ordering ablation before attributing improvements solely to order.

Planned configuration: 30,000 optimizer updates per run; microbatch 4 with four accumulated gradients (effective batch 16); 50-action prediction chunks; learning rate 2.5e−5 with warmup/cosine decay; rank-8 LoRA in attention and MLP projections of both the language transformer and action expert; frozen vision encoder. There are 13,271,040 trainable adapter parameters. Trainable adapters remain float32; the base uses mixed precision. Checkpoints save adapters and optimizer state, sharing one base model instead of saving three full copies.

These are optimizer updates, not 30,000 passes through the dataset. Each update uses 16 sampled observation/action-chunk examples. Sampling is task-balanced and augmented, so there is no exact conventional epoch count.

The first experiment uses training seed 0. It is an exploratory comparison, not yet a multi-seed paper result. Checkpoints retain the latest and best clean-validation-loss adapters. The queued benchmark evaluation uses the prespecified final 30,000-step checkpoint, not the checkpoint that happens to score highest on LIBERO+.

## Evaluation

- Standard LIBERO: 40 tasks × 50 official initial states = 2,000 episodes per model, to measure retained task competence.
- LIBERO+ Robot Initial States only: 350 spatial + 398 object + 409 goal + 393 long-horizon variants = **1,550 episodes per model**. One episode per official variant. Other Plus categories are excluded.
- Use task language from the actual BDDL task definition. Do not append filename perturbation suffixes to the language command.
- Use official initial states, ten settling actions and the existing suite horizons (220/280/300/520). Do not lengthen the horizon to improve reported success.
- Predict an action chunk, execute five actions, then query the model again. Apply the same preprocessing and action normalization as training.
- Success means the simulator's task-specific `check_success()` condition becomes true within the horizon. Action loss is not task success rate.
- Preserve every episode's task, seed, difficulty, success and step count, plus model/code/benchmark identifiers. Compare the same task variants across models.

Before a paper, add several training seeds, per-task and per-difficulty results, uncertainty estimates accounting for shared base tasks, failure videos, and ablations. Do not use official Plus outcomes repeatedly to tune the method; use a separately generated development set for that.

## Commands on the server

```bash
cd /home/tushaar/cv_vla
.venv-pi05/bin/python -m pi05_pipeline.status
tail -f results/pi05_initial_v2_20261113/clean_train.log
tail -f results/pi05_v2_audit/full_generation.log
tmux ls
```

Code is in `/home/tushaar/cv_vla/pi05_pipeline/`: `initial_data.py`, `generate_all.py`, `data.py`, `audit_data.py`, `model.py`, `train.py`, `sim_worker.py`, `evaluate.py`, `run_experiment.py` and `status.py`.

Long-run adapters are planned under `/home/tushaar/cv_vla/checkpoints/pi05_initial_v2_{clean,fixed,curriculum}_seed0/step_030000/adapter.safetensors`. These paths describe the intended completed outputs, not files already produced. Evaluation outputs go under `/home/tushaar/cv_vla/results/pi05_initial_v2_20261113/`.

## Relevant sources and novelty limits

- [Official LIBERO+ repository](https://github.com/sylvestf/LIBERO-plus): benchmark definitions and perturbation categories.
- [Official OpenPI training configurations](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py): reference π0.5 fine-tuning recipes. The present single-GPU LoRA experiment is not an exact reproduction of its large-batch recipe.
- [LeRobot π0.5 documentation](https://huggingface.co/docs/lerobot/pi05): model integration and fine-tuning guidance. The installed project version is pinned; current online examples may differ.
- [Pre-execution return project](https://github.com/distortionlab/pre-execution-return): related work already studies returning the robot toward a nominal pose. Recovery toward the start pose alone must not be presented as a new contribution. A distinct, supported contribution would require evidence about learned recovery, curriculum, generalization and the comparison with such inference-time controllers.
