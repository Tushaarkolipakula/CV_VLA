# SmolVLA contact curriculum experiment

Project: `/home/tushaar/cv_vla` on `172.16.177.111`.

## Objective and protocol

Fine-tune `lerobot/smolvla_base` on the four standard LIBERO suites (40 tasks,
2,000 demonstrations). Compare clean, fixed-noise and progressive-curriculum
training under the same seed, 20,000 optimizer updates and effective batch 16.
Evaluate on standard LIBERO and LIBERO+ without training on LIBERO+.
This is an experiment; improved LIBERO+ success has not yet been established.

The model has 450,046,176 parameters. The pretrained vision/language backbone
is frozen; 99,880,992 action-expert/state-projection parameters are trainable.
Weights are loaded strictly from the complete pretrained policy. LeRobot is
installed and supplies the actual SmolVLA implementation. A custom HDF5 adapter
reads LIBERO directly, so conversion to LeRobotDataset is unnecessary.

## Environment

- Model: `.venv-smolvla`, LeRobot v0.4.4, PyTorch 2.6.0+cu118.
- Standard simulator: `robo_env`, robosuite 1.4.1, MuJoCo 2.3.2.
- Plus simulator: `/home/tushaar/robo_plus`, separate dependencies/assets.
- Sources: `/home/tushaar/data/LIBERO-datasets/{libero_spatial,libero_object,libero_goal,libero_10}`.
- Isolated simulator configs: `configs/libero_standard` and `configs/libero_plus`.
- Pinned model revision: `d9f33c94a60fb382c90dea2164c96845bd955e28`.
- Pinned LeRobot commit: `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`.

The original CNN and its checkpoints are preserved. Its best checkpoint scored
16/100 and its final checkpoint scored 6/100 on standard LIBERO Goal, using
the same 10 initial states per task. These are not LIBERO+ results.

## Data correctness

The source exporter stores post-action observations. In twelve simulator checks
across all four suites, source observation[t] matched state[t+1] about 24 times
more closely than state[t]. Clean training therefore pairs observation[t] with
action[t+1]. Generated recovery observations are recorded before their actions
and use offset zero. Images retain the measured source orientation.

Train/validation splitting is by source demonstration and stratified by task.
All recovery descendants stay in their source split. Normalization statistics
come only from clean training data. Padded action dimensions and padded chunk
timesteps are excluded from the flow-matching loss.

## Contact perturbations

The generator restores each expert demonstration's model XML and simulator
state, finds the first actual finger collision with the configured target,
and applies a short physical action perturbation. Five levels scale up to
3 cm and 0.15 radians. Actual achieved displacement is recorded.

The disturbance itself is not an expert training label. Training samples
contain corrective actions plus the subsequent expert continuation. They are
accepted only after pose recovery and simulator task success. Failed attempts
remain in diagnostic reports but are excluded from training. Existing old
servo-only augmentation is excluded too.

`configs/contact_targets.json` overrides ambiguous BDDL declarations. This
version perturbs the first contact with one selected object per task, not every
object interaction in a long-horizon task. Curriculum training unlocks noise
levels progressively and increases the recovery sampling probability to 30%.
Tasks are sampled uniformly; unavailable recoveries fall back to clean samples,
with actual recovery fraction logged.

## Run the complete experiment

```bash
cd /home/tushaar/cv_vla
.venv-smolvla/bin/python -m smolvla_pipeline.run_experiment \
  --name contact_v2_20261004 --plan-only
nohup .venv-smolvla/bin/python -u -m smolvla_pipeline.run_experiment \
  --name contact_v2_20261004 \
  > results/contact_v2_20261004/runner.log 2>&1 < /dev/null &
```

The runner holds a process lock, records stage state, generates recoveries with
four simulator processes, trains the three models sequentially on the GPU,
and evaluates each selected checkpoint. It stops on an error rather than
silently proceeding with a failed stage. Do not start a second copy manually
while the existing runner is active.

Training chooses the checkpoint with lowest fixed held-out clean loss over
32 batches. This is a proxy, not a success-rate measurement; LIBERO+ is never
used for checkpoint selection. Evaluation uses 50 official initial states per
standard task and one trial per LIBERO+ variant, with the same seed and action
execution settings across modes. Plus evaluation is split into blocks of 50
variants to retain completed work. Expect substantial runtime for complete
evaluation. Aggregate success is weighted by episode count; per-suite and
per-category results are included. The reported Wilson intervals are simple
binomial intervals and do not model correlations among benchmark variants.

## Individual stages

```bash
# Generate; use a fresh output version if changing the settings.
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/configs/libero_standard \
  robo_env/bin/python -m smolvla_pipeline.generate_all \
  --output data/contact_manual_v1 --targets configs/contact_targets.json \
  --levels 5 --repeats 1 --workers 4

# Train one mode; choose clean, fixed or curriculum.
RECOVERY_ROOT=$PWD/data/contact_manual_v1 \
  bash experiments/train_smolvla.sh curriculum manual_v1

# Evaluate a real step directory selected from best_validation.json.
.venv-smolvla/bin/python -m smolvla_pipeline.evaluate \
  --checkpoint checkpoints/RUN/step_XXXXXX \
  --benchmark plus --suite libero_goal --out results/UNIQUE_EVAL_NAME
```

For a clean baseline, recovery input is unnecessary. Checkpoints include model,
optimizer, scheduler, RNG state, source hashes, data manifests and preprocessing.
Use `--resume` with the same run directory and identical training settings.
The resume path must match the last recorded metric step. If interruption
happens between saves, preserve the trailing logs and reconcile them explicitly
before resuming; the runner intentionally refuses to overwrite such a run.
Interrupted evaluation blocks similarly need a fresh output path or a preserved
archive of the incomplete block. Completed stages/blocks can be retained.

## Progress and evidence

```bash
cat results/contact_v2_20261004/status.json
tail -n 20 results/contact_v2_20261004/runner.log
tail -n 5 checkpoints/smolvla_curriculum_contact_v2_20261004/metrics.jsonl
cat results/contact_v2_20261004/comparison.json
```

Evidence includes `results/full_run/dataset_inventory.json`,
`results/full_run/alignment.json`, `results/smolvla_setup/resume_equivalence.json`,
the CNN evaluation JSON files, and per-task generation reports.
Run results are incomplete until `status.json` says `complete`. A completed
plumbing smoke test is not evidence of a high-performing policy.
