"""Evaluate a checkpoint in LIBERO using the same image/proprio pipeline as training.

Run from ~/cv_vla:
    python -m training.evaluate --checkpoint ~/cv_vla/checkpoints/run1/checkpoint_best.pt \
        --suite libero_goal --episodes 50

Only evaluate on the suite you trained on: task ids come from the training files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for cand in (ROOT / "external" / "LIBERO", ROOT.parent / "external" / "LIBERO"):
    if cand.exists():
        sys.path.insert(0, str(cand))

from training.common import CAMERAS, env_obs_to_images, proprio_from_env_obs  # noqa: E402
from training.policy import VisionProprioPolicy  # noqa: E402

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
NUM_WAIT = 10  # no-op steps while objects settle, as in standard LIBERO eval


def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ck["meta"]
    model = VisionProprioPolicy(m["proprio_dim"], m["action_dim"], num_tasks=len(m["task_names"])).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, m


@torch.no_grad()
def act(model, meta, obs, task_id, device):
    imgs = env_obs_to_images(obs, meta["image_flips"])
    image = np.concatenate([imgs[k] for k, _ in CAMERAS], axis=-1).transpose(2, 0, 1)
    pro = (proprio_from_env_obs(obs) - np.array(meta["proprio_mean"], np.float32)) / np.array(meta["proprio_std"], np.float32)
    a = model(
        torch.from_numpy(np.ascontiguousarray(image))[None].to(device),
        torch.from_numpy(pro.astype(np.float32))[None].to(device),
        torch.tensor([task_id], device=device),
    )[0].cpu().numpy()
    a = np.clip(a, -1, 1)
    a[6] = 1.0 if a[6] > 0 else -1.0
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--save_videos", action="store_true")
    p.add_argument("--out", default=None, help="JSON results path")
    a = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_policy(Path(a.checkpoint).expanduser(), device)
    print(f"image_flips from checkpoint: {meta['image_flips']}")

    suite = benchmark.get_benchmark_dict()[a.suite]()
    results = {}
    for ti in range(suite.n_tasks):
        task = suite.get_task(ti)
        if task.name not in meta["task_names"]:
            print(f"skip {task.name}: not in training data")
            continue
        task_id = meta["task_names"].index(task.name)
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=a.image_size, camera_widths=a.image_size,
                                 camera_names=["agentview", "robot0_eye_in_hand"])
        init_states = suite.get_task_init_states(ti)
        succ = 0
        for ep in range(a.episodes):
            env.seed(ep)
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            frames, done = [], False
            for step in range(MAX_STEPS.get(a.suite, 400) + NUM_WAIT):
                action = np.array([0, 0, 0, 0, 0, 0, -1.0]) if step < NUM_WAIT else act(model, meta, obs, task_id, device)
                obs, _, done, _ = env.step(action)
                if a.save_videos and ep < 2:
                    frames.append(np.flip(obs["agentview_image"], 0))
                if done:
                    break
            succ += int(done)
            if frames:
                import imageio.v2 as imageio

                vdir = Path(a.checkpoint).expanduser().parent / "eval_videos"
                vdir.mkdir(exist_ok=True)
                imageio.mimsave(vdir / f"{task.name}_ep{ep}_{'ok' if done else 'fail'}.mp4", frames, fps=20)
        env.close()
        results[task.name] = succ / a.episodes
        print(f"{task.name}: {succ}/{a.episodes}", flush=True)

    mean = float(np.mean(list(results.values()))) if results else 0.0
    print(f"mean success: {mean:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps({"per_task": results, "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
