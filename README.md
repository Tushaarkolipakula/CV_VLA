# CV_VLA

Experiments on fine-tuning pretrained vision-language-action models for robust manipulation with LIBERO and LIBERO+.

The current pipeline fine-tunes **π0.5 using LoRA**, comparing clean demonstrations, a fixed mixture of initial-pose recovery examples, and a progressive curriculum. LIBERO+ evaluation is restricted to **Robot Initial States**. Standard LIBERO evaluation measures retained task competence. The earlier SmolVLA contact-perturbation experiments and legacy code are retained for reference.

## Clone and set up

```bash
git clone https://github.com/Tushaarkolipakula/CV_VLA.git cv_vla
cd cv_vla
```

See [SETUP.md](SETUP.md) for pinned source revisions, Python environments, dataset/model downloads, and adapting paths for a new machine. Cloning provides the code; datasets, pretrained weights, trained adapters, virtual environments and local results are intentionally excluded. GPU training was tested on Linux with Python 3.10 and an NVIDIA RTX 3090 Ti. Other hardware needs a compatible PyTorch/CUDA installation.

## Project layout

| Directory/file | Purpose |
|---|---|
| `pi05_pipeline/` | Current dataset generation, loading, LoRA training, evaluation and status tools |
| `smolvla_pipeline/` | Earlier SmolVLA experiments and shared utilities |
| `augmentation/`, `training/`, `curriculum/` | Data/simulator helpers and legacy experiments |
| `configs/` | Experiment settings; local LIBERO paths are generated during setup |
| `experiments/`, `tools/`, `tests/`, `visualization/` | Scripts, checks and analysis tools |
| `requirements/` | Package-version snapshots from the working server |
| `docs/PI05_EXPERIMENT_PLAN.md` | Protocol, noise distribution, comparison and evaluation details |

## Current experiment

Each of the three runs starts independently from the same pretrained π0.5 base. Clean training uses original LIBERO demonstrations. Fixed training mixes in all five recovery levels throughout. Curriculum training progressively unlocks levels while keeping the recovery sampling probability at 10%.

Generated recoveries perturb the arm at episode initialization, return toward its original starting pose, and continue the expert demonstration. Only successful simulated tasks are accepted. Source demonstrations and their generated descendants stay in the same train/validation split. This differs from the earlier contact-triggered experiment.

The planned budget is 30,000 optimizer updates per model with effective batch size 16. These are updates, not epochs. Evaluation covers 2,000 standard LIBERO episodes and 1,550 LIBERO+ Robot Initial States variants per model. Final results are not yet established; no target success rate is guaranteed.

After setup, inspect a run with:

```bash
.venv-pi05/bin/python -m pi05_pipeline.status
```

Run generation and the training/evaluation queue in separate persistent terminals:

```bash
LIBERO_CONFIG_PATH="$PWD/configs/libero_standard" MUJOCO_GL=egl \
  robo_env/bin/python -u -m pi05_pipeline.generate_all \
  --output data/initial_joint_v2_20261113 --demos 0

# In a separate terminal; clean training can run during generation.
# Augmented training waits for generation and its quality checks.
.venv-pi05/bin/python -u -m pi05_pipeline.run_experiment
```

The current runner uses fixed experiment directory names. Use a fresh clone/output layout or edit the run configuration deliberately for a new experiment; do not start duplicate runners against the same outputs. See [SMOLVLA_README.md](SMOLVLA_README.md) for the earlier pipeline, whose historical server paths and results describe that earlier experiment.

## Related projects

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO+](https://github.com/sylvestf/LIBERO-plus)
- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)

Third-party code, models and datasets retain their respective licenses and access terms.
