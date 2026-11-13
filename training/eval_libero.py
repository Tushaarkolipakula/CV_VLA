"""Evaluate a trained BC policy on standard LIBERO tasks via closed-loop rollouts.

This measures the thing training/val loss can't: actual task SUCCESS RATE,
using closed-loop rollouts and the environment success signal. For exact
benchmark comparability, use the official LIBERO evaluator and its task-specific
initial states; this script is primarily an end-to-end pipeline/debug evaluator.

Usage:
    python -m training.eval_libero \
        --target_dir ~/data/LIBERO-datasets/libero_goal \
        --checkpoint ~/cv_vla/checkpoints/run1/checkpoint_best.pt \
        --episodes_per_task 50 \
        --save_videos 3

Run from ~/cv_vla (same convention as generate_curriculum_dataset.py / train.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import robosuite.utils.transform_utils as T

from training.policy import VisionProprioPolicy

ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = ROOT / "external" / "LIBERO"
sys.path.insert(0, str(LIBERO_ROOT))

try:
    import libero.libero.envs  # noqa: F401  -- registers LIBERO envs with robosuite
except ImportError:
    pass

from robosuite import make


def resolve_bddl_path(dataset_type: str, task_name: str) -> str:
    candidates = [
        LIBERO_ROOT / "libero" / "libero" / "bddl_files" / dataset_type / f"{task_name}.bddl",
        LIBERO_ROOT / "libero" / "bddl_files" / dataset_type / f"{task_name}.bddl",
        LIBERO_ROOT / "bddl_files" / dataset_type / f"{task_name}.bddl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def build_env(task_file: str):
    with h5py.File(task_file, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])

    task_name = Path(task_file).stem.replace("_demo", "")
    dataset_type = Path(task_file).parent.name
    bddl_path = resolve_bddl_path(dataset_type, task_name)

    env_name = env_args.get("env_name", "Libero_Tabletop_Manipulation")
    env_kwargs = dict(env_args.get("env_kwargs", {}))
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["camera_names"] = ["agentview"]
    env_kwargs["use_camera_obs"] = True
    env_kwargs.pop("bddl_file_name", None)
    env_kwargs.pop("env_name", None)

    env = make(env_name=env_name, bddl_file_name=bddl_path, **env_kwargs)
    return env, task_name


def build_proprio(obs_dict) -> np.ndarray:
    """Must exactly match CurriculumDataset._load_proprio's field order/semantics."""
    gripper = np.asarray(obs_dict["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    joints = np.asarray(obs_dict["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    ee = np.hstack([
        np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32),
        T.quat2axisangle(obs_dict["robot0_eef_quat"]).astype(np.float32),
    ])
    return np.concatenate([gripper, joints, ee])


def build_image(obs_dict) -> np.ndarray:
    """Convert a live LIBERO camera image to the canonical training orientation.

    LIBERO/robosuite camera observations are vertically inverted and the
    standard LIBERO preprocessing used by many evaluation implementations
    rotates them by 180 degrees.  The augmented HDF5 generator stores images
    after this same transformation, so this function must be identical to the
    generator's live-image preprocessing.
    """
    img = np.asarray(obs_dict["agentview_image"])
    img = img[::-1, ::-1]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))


def load_demo_initial_state(task_file: str) -> np.ndarray:
    """Use the recorded initial simulator state for deterministic debugging.

    This is not the full official LIBERO randomized-initial-state protocol;
    it is a reliable local sanity-check starting point.
    """
    with h5py.File(task_file, "r") as f:
        demo_keys = sorted(k for k in f["data"].keys() if k.startswith("demo_"))
        if not demo_keys:
            raise RuntimeError(f"No demos found in {task_file}")
        demo = f[f"data/{demo_keys[0]}"]
        if "init_state" in demo.attrs:
            return np.asarray(demo.attrs["init_state"], dtype=np.float64)
        return np.asarray(demo["states"][0], dtype=np.float64)


def rollout_episode(env, model, device, max_steps: int, record: bool, init_state=None, wait_steps: int = 10):
    obs_dict = env.reset()

    if init_state is not None:
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()
        obs_dict = env._get_observation() if hasattr(env, "_get_observation") else env._get_observation()

    frames = []
    success = False

    # Let the simulator settle before the first policy action. This mirrors
    # the dummy-wait used by common LIBERO evaluation implementations.
    dummy_action = np.zeros(7, dtype=np.float32)
    dummy_action[-1] = -1.0
    for _ in range(wait_steps):
        obs_dict, _, done, _ = env.step(dummy_action)
        if done:
            return False, frames

    for _step in range(max_steps):
        image = torch.from_numpy(build_image(obs_dict)).unsqueeze(0).to(device)
        proprio = torch.from_numpy(build_proprio(obs_dict)).unsqueeze(0).to(device)

        with torch.no_grad():
            action = model(image, proprio).squeeze(0).cpu().numpy()

        # LIBERO OSC-style actions are conventionally bounded to [-1, 1].
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs_dict, reward, done, info = env.step(action)

        if record:
            frame = env.sim.render(width=640, height=480, camera_name="agentview")
            frames.append(np.ascontiguousarray(frame[::-1, ::-1]))

        env_success = bool(hasattr(env, "_check_success") and env._check_success())
        info_success = bool(isinstance(info, dict) and info.get("success", False))
        if env_success or info_success or float(reward) >= 1.0:
            success = True
            break
        if done:
            break

    return success, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, help="Directory of ORIGINAL (non-augmented) LIBERO .hdf5 task files -- used only to read task/env metadata, not their demo data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes_per_task", type=int, default=50, help="LIBERO's standard protocol uses 50")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--use_demo_initial_state", action="store_true", help="Start each local episode from demo_0 init_state; useful for deterministic pipeline debugging.")
    parser.add_argument("--save_videos", type=int, default=0, help="Number of episodes per task to save as .mp4 (0 = none)")
    parser.add_argument("--video_dir", type=str, default="./eval_videos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    task_files = sorted(glob.glob(str(Path(args.target_dir) / "*.hdf5")))
    if not task_files:
        raise RuntimeError(f"No .hdf5 files found under {args.target_dir}")

    # Infer proprio_dim from the first task's env before building the model.
    probe_env, _ = build_env(task_files[0])
    probe_obs = probe_env.reset()
    proprio_dim = build_proprio(probe_obs).shape[0]
    probe_env.close()

    model = VisionProprioPolicy(proprio_dim=proprio_dim, action_dim=7).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.save_videos > 0:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for task_file in task_files:
        env, task_name = build_env(task_file)
        successes = 0
        for ep in range(args.episodes_per_task):
            record = ep < args.save_videos
            init_state = load_demo_initial_state(task_file) if args.use_demo_initial_state else None
            success, frames = rollout_episode(env, model, device, args.max_steps, record, init_state=init_state, wait_steps=args.wait_steps)
            successes += int(success)
            if record:
                video_path = Path(args.video_dir) / f"{task_name}_ep{ep}_{'SUCCESS' if success else 'FAIL'}.mp4"
                imageio.mimsave(video_path, frames, fps=20)

        success_rate = successes / args.episodes_per_task
        results[task_name] = success_rate
        print(f"{task_name}: {successes}/{args.episodes_per_task} = {success_rate*100:.1f}%")
        env.close()

    overall = float(np.mean(list(results.values())))
    print("=" * 50)
    print(f"Overall average success rate across {len(results)} tasks: {overall*100:.1f}%")
    print("Reference: recent published LIBERO SOTA averages are roughly 95-97%;")
    print("original LIBERO baselines were roughly 50-90% depending on suite/method.")


if __name__ == "__main__":
    main()
