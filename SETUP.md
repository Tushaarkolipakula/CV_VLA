# Using this checkout on another machine

The repository preserves the current project source files. Existing training scripts have **not** been rewritten during publication. Some still contain `/home/tushaar` paths. A new machine needs its own datasets, weights, environments and local path configuration. This guide records the working setup; a clean installation on another machine has not been tested end to end.

## 1. Layout and pinned dependencies

Use Linux, Python 3.10, Git, an NVIDIA GPU with an appropriate driver, and EGL/OpenGL support for headless MuJoCo rendering. Clone into a directory named `cv_vla` to simplify the existing layout:

```text
workspace/
  cv_vla/                  # this repository
    external/lerobot/
    .venv-pi05/
    robo_env/
  external/LIBERO/
  LIBERO-plus/
  robo_plus/
  data/LIBERO-datasets/
```

From the new `cv_vla` checkout, clone the external sources separately:

```bash
mkdir -p external ../external
git clone https://github.com/huggingface/lerobot.git external/lerobot
git -C external/lerobot checkout 8fff0fde7c79f23a93d845d1a50e985de01f8b8a
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../external/LIBERO
git -C ../external/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
git clone https://github.com/sylvestf/LIBERO-plus.git ../LIBERO-plus
git -C ../LIBERO-plus checkout 4976dc30028e805ff8094b55501d532c48fec182
```

These directories are excluded from this repository. Follow each benchmark's own instructions to obtain its assets and initial-state files; cloning source alone may not provide the complete asset downloads.

## 2. Model environment

The original server uses a `.venv-pi05` environment that shares some dependencies with `.venv-smolvla`. On another machine, build an independent environment rather than copying that machine-specific linkage.

```bash
python3.10 -m venv .venv-pi05
.venv-pi05/bin/python -m pip install --upgrade pip
.venv-pi05/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-pi05/bin/python -m pip install -e external/lerobot
.venv-pi05/bin/python -m pip install \
  'git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0' \
  tokenizers==0.21.4 peft==0.17.1 numpy==2.2.6 scipy==1.14.1 \
  sentencepiece==0.2.2 h5py==3.16.0 safetensors==0.8.0 torchcodec==0.2.1
```

The Transformers revision is the `fix/lerobot_openpi` implementation used by this π0.5 pipeline. Do not substitute an arbitrary current Transformers release. Package inventories from the actual server are under `requirements/`; they include transitive and some unrelated installed packages, so they are provenance snapshots rather than validated minimal lockfiles.

## 3. Simulator environments

Keep the legacy simulator stack isolated from the model's NumPy 2 environment. Create `robo_env` inside the project for standard LIBERO and `robo_plus` in its parent directory for LIBERO+. Follow the pinned repositories' installation instructions, then match the relevant `requirements/` inventory. The original repositories' requirements list robosuite 1.4.0, while this experiment uses **robosuite 1.4.1** and **MuJoCo 2.3.2**. Both use Python 3.10; the current standard environment uses NumPy 1.22.4 and Plus uses NumPy 1.26.4. Install each corresponding LIBERO checkout into its own environment, or ensure its source is importable; the simulator workers also insert their selected repository into `sys.path`.

The original model runs used PyTorch 2.6.0 with CUDA 11.8 wheels; simulator package versions are captured separately. Select compatible wheels for a different driver/GPU instead of copying compiled packages from the old virtual environments.

## 4. Data and pretrained weights

Download the original LIBERO HDF5 datasets separately into:

```text
../data/LIBERO-datasets/libero_spatial/
../data/LIBERO-datasets/libero_object/
../data/LIBERO-datasets/libero_goal/
../data/LIBERO-datasets/libero_10/
```

There should be ten task HDF5 files per suite. Do not substitute a differently aligned export without checking observation/action timing.

Accept the tokenizer terms at [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) and authenticate locally:

```bash
.venv-pi05/bin/hf auth login
```

Download the pinned base and write the local pointer expected by the scripts:

```bash
.venv-pi05/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
base = snapshot_download('lerobot/pi05_base',
    revision='22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0')
p = Path('results/pi05_setup/base_path.txt')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(base + '\n')
print('Base path recorded:', base)
PY
```

The repository does not contain trained adapters. To resume an existing experiment rather than start a new one, transfer its adapter/checkpoint, run manifest, preprocessing statistics, matching data and optimizer state separately. A Git clone alone cannot resume it.

## 5. Adapt paths in the NEW clone

Do this only before starting an experiment. Changing source paths after training starts changes the source hashes checked during resume.

Replace `/home/tushaar/cv_vla` with the new checkout's absolute path, then `/home/tushaar` with the checkout's parent path in the Python and shell scripts that reference them. Inspect the changes with `git diff`. Do not run that replacement on the original active server checkout. Some archived legacy scripts may have additional machine-specific assumptions.

Create the two ignored configuration files `configs/libero_standard/config.yaml` and `configs/libero_plus/config.yaml`. For each, set:

```yaml
benchmark_root: /absolute/path/to/benchmark/libero/libero
bddl_files: /absolute/path/to/benchmark/libero/libero/bddl_files
init_states: /absolute/path/to/benchmark/libero/libero/init_files
assets: /absolute/path/to/benchmark/libero/libero/assets
datasets: /absolute/path/to/workspace/data/LIBERO-datasets
```

Use `workspace/external/LIBERO` for the standard benchmark and `workspace/LIBERO-plus` for Plus. Do not commit tokens or these machine-specific files.

## 6. Validate before a long run

First run a small dataset-generation pilot in a fresh output directory and a short `pi05_pipeline.train` run with `--steps 2 --batch-size 1 --accumulation 1 --save-every 2 --validation-batches 1`. Supply the original suite directory with `--data` and a fresh checkpoint directory with `--out`. Then verify checkpoint reload and simulator evaluation on one task before starting the complete queue. Success on a smoke test establishes functioning code, not benchmark accuracy.

The complete experiment protocol is in [docs/PI05_EXPERIMENT_PLAN.md](docs/PI05_EXPERIMENT_PLAN.md). Git ignores generated data, weights, environments and output directories so normal subsequent commits remain code-only.
