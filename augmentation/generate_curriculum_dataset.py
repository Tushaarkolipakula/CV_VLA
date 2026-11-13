"""Generate pre-grasp perturbation + recovery data for curriculum training.

For every expert demo:
  * the clean demo is copied unchanged (same obs layout as the source file);
  * at a timestep t sampled in the window just before the gripper first closes
    (or, for tasks with no grasp such as pushing, during the early approach
    phase), the sim is set to the exact demo state, the end-effector is pushed
    off by a random offset of a known size (meters / radians), and a
    closed-loop controller drives it back to the demo pose. Only the recovery
    frames and their corrective actions are saved, as a short "recovery" demo.
    The perturbation frames themselves are never saved as training targets.

Noise is generated at several levels (level 1 = smallest, level N = largest).
Training unlocks levels progressively, which is the curriculum.

Images are saved in the dataset's own orientation. The flip between raw env
frames and dataset frames is measured per file and stored in
data.attrs["image_flips"]; training and evaluation read it from there.

This legacy pre-grasp generator verifies pose recovery only, not task success.
For first-contact, task-success-verified recoveries use smolvla_pipeline.generate.

Run from ~/cv_vla:
    python -m augmentation.generate_curriculum_dataset \
        --target_dir ~/data/LIBERO-datasets/libero_goal \
        --output_dir ~/cv_vla/data/LIBERO-datasets-augmented/libero_goal
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
import zlib
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # ~/cv_vla
sys.path.insert(0, str(ROOT))
for cand in (ROOT / "external" / "LIBERO", ROOT.parent / "external" / "LIBERO"):
    if cand.exists():
        sys.path.insert(0, str(cand))

from training.common import (  # noqa: E402
    CAMERAS,
    apply_flip,
    detect_flip,
    dumps_flips,
    task_name_from_file,
)

# ---------------------------------------------------------------------------
# Noise settings. Level i of N uses magnitude lo + (hi - lo) * i / N.
# ---------------------------------------------------------------------------
POS_MAG_RANGE = (0.01, 0.08)    # meters
ROT_MAG_RANGE = (0.03, 0.35)    # radians (~2 deg -> ~20 deg)

# Perturb somewhere in [t_grasp - PRE_GRASP_WINDOW, t_grasp - PRE_GRASP_MARGIN]
PRE_GRASP_WINDOW = 25
PRE_GRASP_MARGIN = 3

# Fallback for demos with no real grasp (gripper never closes, or closes
# almost immediately): perturb during this fraction range of the demo length.
MIN_VALID_GRASP_STEP = 10
NO_GRASP_FRAC = (0.10, 0.40)

PERTURB_STEPS = 15
RECOVERY_MAX_STEPS = 80
RECOVERY_POS_TOL = 0.008        # meters
RECOVERY_ROT_TOL = 0.05         # radians
OBJECT_MOVE_TOL = 0.01          # discard sample if any object moved more than this

# robosuite OSC_POSE default output_max: 1.0 action unit = 0.05 m / 0.5 rad
OSC_POS_SCALE = 0.05
OSC_ROT_SCALE = 0.5
CTRL_GAIN = 1.0


def make_env(bddl_path: str, height: int, width: int):
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=height,
        camera_widths=width,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )


def find_bddl(task_name: str, suite: str) -> str:
    try:
        from libero.libero import get_libero_path

        p = Path(get_libero_path("bddl_files")) / suite / f"{task_name}.bddl"
        if p.exists():
            return str(p)
    except Exception:
        pass
    hits = glob.glob(str(ROOT / "**" / "bddl_files" / suite / f"{task_name}.bddl"), recursive=True)
    hits += glob.glob(str(ROOT.parent / "external" / "LIBERO" / "**" / "bddl_files" / suite / f"{task_name}.bddl"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"BDDL for {suite}/{task_name} not found")
    return hits[0]


def rot_error(target_quat, cur_quat):
    """World-frame axis-angle that rotates cur into target (matches OSC delta convention)."""
    import robosuite.utils.transform_utils as T

    r = T.quat2mat(target_quat) @ T.quat2mat(cur_quat).T
    return T.quat2axisangle(T.mat2quat(r))


def servo_action(obs, target_pos, target_quat, gripper):
    pos_err = target_pos - obs["robot0_eef_pos"]
    rot_err = rot_error(target_quat, obs["robot0_eef_quat"])
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip(CTRL_GAIN * pos_err / OSC_POS_SCALE, -1.0, 1.0)
    a[3:6] = np.clip(CTRL_GAIN * rot_err / OSC_ROT_SCALE, -1.0, 1.0)
    a[6] = gripper
    return a, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))


def random_unit(rng, n):
    v = rng.normal(size=n)
    return v / (np.linalg.norm(v) + 1e-9)


def robot_qpos_indices(env):
    try:
        robot = env.env.robots[0]
        return set(robot._ref_joint_pos_indexes) | set(robot._ref_gripper_joint_pos_indexes)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("Cannot identify robot joints; object-motion check unavailable") from exc


def objects_moved(env, ref_qpos, robot_idx) -> bool:
    if not robot_idx:
        raise ValueError("Missing robot joint indices")
    qpos = env.sim.data.qpos
    other = [i for i in range(len(qpos)) if i not in robot_idx]
    if not other:
        return False
    return float(np.max(np.abs(qpos[other] - ref_qpos[other]))) > OBJECT_MOVE_TOL


def perturb_window(actions: np.ndarray):
    """Return (t_grasp, t_lo, t_hi): the range of timesteps eligible for perturbation.

    Normal case: the last PRE_GRASP_WINDOW steps before the gripper first closes.
    No-grasp case (gripper never closes, or closes within the first
    MIN_VALID_GRASP_STEP steps): a fraction of the demo, so the perturbation
    lands during the approach rather than while touching objects.
    """
    n = len(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or n < 2:
        raise ValueError("Expected at least two 7-dimensional actions")
    closed = np.where(actions[:, 6] > 0)[0]
    t_grasp = int(closed[0]) if len(closed) else n
    if len(closed) and t_grasp >= MIN_VALID_GRASP_STEP:
        t_lo = max(1, t_grasp - PRE_GRASP_WINDOW)
        t_hi = max(t_lo, t_grasp - PRE_GRASP_MARGIN)
    else:
        t_lo = max(1, int(NO_GRASP_FRAC[0] * n))
        t_hi = max(t_lo, int(NO_GRASP_FRAC[1] * n))
    return t_grasp, min(t_lo, n-1), min(t_hi, n-1)


def copy_demo(src, dst_grp):
    for k in ("actions", "states", "rewards", "dones", "robot_states"):
        if k in src:
            dst_grp.create_dataset(k, data=src[k][:])
    o = dst_grp.create_group("obs")
    for k in src["obs"].keys():
        o.create_dataset(k, data=src["obs"][k][:], compression="gzip", compression_opts=4)
    for k, v in src.attrs.items():
        dst_grp.attrs[k] = v


def process_task(task_file, output_dir, num_levels, augs_per_level, seed, save_videos):
    import robosuite.utils.transform_utils as T

    task_file = str(task_file)
    task_name = task_name_from_file(task_file)
    suite = Path(task_file).parent.name
    out_file = Path(output_dir) / Path(task_file).name
    rng = np.random.default_rng(seed + zlib.crc32(task_name.encode()))
    print(f"[{task_name}] -> {out_file}", flush=True)

    if num_levels < 1 or augs_per_level < 1 or seed < 0:
        raise ValueError("Expected positive levels/repeats and non-negative seed")
    if out_file.exists() or out_file.resolve() == Path(task_file).resolve():
        raise FileExistsError(f"Refusing to overwrite {out_file}")
    from smolvla_pipeline.generate import restore_demo
    env = None
    partial = None
    try:
        with h5py.File(task_file, "r") as f_in:
            demo_keys = sorted(f_in["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            h, w = f_in[f"data/{demo_keys[0]}/obs/agentview_rgb"].shape[1:3]
            env = make_env(find_bddl(task_name, suite), h, w)

            # The source exporter records obs[0] after actions[0].
            obs = restore_demo(env, f_in["data"][demo_keys[0]], 1)
            flips = {}
            for ds_key, env_key in CAMERAS:
                mode, errs = detect_flip(obs[env_key], f_in[f"data/{demo_keys[0]}/obs/{ds_key}"][0])
                flips[ds_key] = mode
                print(f"[{task_name}] {ds_key}: flip={mode} errors={ {k: round(v, 1) for k, v in errs.items()} }", flush=True)
                if errs[mode] > 30:
                    print(f"[{task_name}] WARNING: best flip error {errs[mode]:.1f} is high; check camera settings", flush=True)

            stats = {"ok": 0, "timeout": 0, "object_moved": 0}
            fd, name = tempfile.mkstemp(prefix="." + out_file.name, suffix=".partial", dir=out_file.parent)
            os.close(fd)
            partial = Path(name)
            with h5py.File(partial, "w") as f_out:
                data_out = f_out.create_group("data")
                for k, v in f_in["data"].attrs.items():
                    data_out.attrs[k] = v
                data_out.attrs["image_flips"] = dumps_flips(flips)
                data_out.attrs["task_name"] = task_name
                data_out.attrs["num_levels"] = num_levels

                # 1) clean demos, unchanged
                for k in demo_keys:
                    f_in.copy(f_in["data"][k], data_out, name=k)
                    g = data_out[k]
                    g.attrs["kind"] = "clean"
                    g.attrs["noise_level"] = 0

                # 2) recovery demos
                rec_idx = 0
                for di, k in enumerate(demo_keys):
                    group = f_in["data"][k]
                    actions = f_in[f"data/{k}/actions"][:]
                    t_grasp, t_lo, t_hi = perturb_window(actions)

                    for level in range(1, num_levels + 1):
                        frac = level / num_levels
                        pos_mag = POS_MAG_RANGE[0] + (POS_MAG_RANGE[1] - POS_MAG_RANGE[0]) * frac
                        rot_mag = ROT_MAG_RANGE[0] + (ROT_MAG_RANGE[1] - ROT_MAG_RANGE[0]) * frac

                        for a_i in range(augs_per_level):
                            t = int(rng.integers(t_lo, t_hi + 1))
                            gripper = float(actions[t, 6])

                            obs = restore_demo(env, group, t)
                            robot_idx = robot_qpos_indices(env)
                            ref_qpos = env.sim.data.qpos.copy()
                            home_pos = obs["robot0_eef_pos"].copy()
                            home_quat = obs["robot0_eef_quat"].copy()

                            d = random_unit(rng, 3)
                            d[2] = abs(d[2])  # never push down into the table/objects
                            tgt_pos = home_pos + d * pos_mag
                            dq = T.axisangle2quat(random_unit(rng, 3) * rot_mag)
                            tgt_quat = T.mat2quat(T.quat2mat(dq) @ T.quat2mat(home_quat))

                            video = []
                            for _ in range(PERTURB_STEPS):
                                a, _, _ = servo_action(obs, tgt_pos, tgt_quat, gripper)
                                obs, _, _, _ = env.step(a)
                                if save_videos:
                                    video.append(apply_flip(obs["agentview_image"], "ud"))
                            achieved_pos = float(np.linalg.norm(obs["robot0_eef_pos"] - home_pos))
                            achieved_rot = float(np.linalg.norm(rot_error(home_quat, obs["robot0_eef_quat"])))

                            rec = {ds: [] for ds, _ in CAMERAS}
                            rec_grip, rec_pos, rec_ori, rec_joint, rec_act = [], [], [], [], []
                            recovered = False
                            for _ in range(RECOVERY_MAX_STEPS):
                                a, dp, dr = servo_action(obs, home_pos, home_quat, gripper)
                                if dp < RECOVERY_POS_TOL and dr < RECOVERY_ROT_TOL:
                                    recovered = True
                                    break
                                for ds_key, env_key in CAMERAS:
                                    rec[ds_key].append(apply_flip(obs[env_key], flips[ds_key]))
                                rec_grip.append(obs["robot0_gripper_qpos"].copy())
                                rec_pos.append(obs["robot0_eef_pos"].copy())
                                rec_ori.append(T.quat2axisangle(obs["robot0_eef_quat"]))
                                rec_joint.append(obs["robot0_joint_pos"].copy())
                                rec_act.append(a)
                                obs, _, _, _ = env.step(a)
                                if save_videos:
                                    video.append(apply_flip(obs["agentview_image"], "ud"))

                            if not recovered:
                                _, dp, dr = servo_action(obs, home_pos, home_quat, gripper)
                                recovered = dp < RECOVERY_POS_TOL and dr < RECOVERY_ROT_TOL
                            if not recovered:
                                stats["timeout"] += 1
                                continue
                            if objects_moved(env, ref_qpos, robot_idx):
                                stats["object_moved"] += 1
                                continue
                            if len(rec_act) == 0:
                                continue

                            g = data_out.create_group(f"recovery_{rec_idx}")
                            o = g.create_group("obs")
                            for ds_key, _ in CAMERAS:
                                o.create_dataset(ds_key, data=np.stack(rec[ds_key]).astype(np.uint8), compression="gzip", compression_opts=4)
                            o.create_dataset("gripper_states", data=np.array(rec_grip, dtype=np.float64))
                            o.create_dataset("ee_pos", data=np.array(rec_pos, dtype=np.float64))
                            o.create_dataset("ee_ori", data=np.array(rec_ori, dtype=np.float64))
                            o.create_dataset("joint_states", data=np.array(rec_joint, dtype=np.float64))
                            g.create_dataset("actions", data=np.array(rec_act, dtype=np.float64))
                            g.attrs["kind"] = "recovery"
                            g.attrs["trigger_type"] = "pregrasp_or_approach"
                            g.attrs["task_success_verified"] = False
                            g.attrs["noise_level"] = level
                            g.attrs["source_demo"] = k
                            g.attrs["t_perturb"] = t
                            g.attrs["t_grasp"] = t_grasp
                            g.attrs["pos_mag_m"] = achieved_pos
                            g.attrs["rot_mag_rad"] = achieved_rot
                            g.attrs["num_samples"] = len(rec_act)
                            rec_idx += 1
                            stats["ok"] += 1

                            if save_videos and di < 2 and a_i == 0:
                                import imageio.v2 as imageio

                                vdir = Path(output_dir) / "videos"
                                vdir.mkdir(exist_ok=True)
                                imageio.mimsave(vdir / f"{task_name}_{k}_level{level}.mp4", video, fps=20)

                if not rec_idx:
                    raise RuntimeError(f"No usable recoveries for {task_name}: {stats}")
                data_out.attrs["generation_complete"] = True
        os.link(partial, out_file)  # atomic publication; never overwrite another run
        print(f"[{task_name}] done: {stats}", flush=True)
        return stats
    except Exception as e:
        print(f"[{task_name}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return None
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
        if env is not None:
            env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_levels", type=int, default=5, help="Number of curriculum noise levels")
    p.add_argument("--augs_per_level", type=int, default=2, help="Recovery samples per demo per level")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="0 = one per task, capped at CPU count")
    p.add_argument("--save_videos", action="store_true", help="Save a few perturbation/recovery videos for inspection")
    p.add_argument("--only", type=str, default=None, help="Substring filter on task file names (for testing)")
    args = p.parse_args()

    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if Path(args.target_dir).expanduser().resolve() == Path(args.output_dir):
        p.error("Source and output directories must differ")
    if min(args.num_levels, args.augs_per_level) < 1 or min(args.seed,args.workers) < 0:
        p.error("Levels and repeats must be positive; seed and workers non-negative")
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.target_dir), "*.hdf5")))
    if args.only:
        files = [f for f in files if args.only in f]
    if not files:
        raise SystemExit(f"No .hdf5 files in {args.target_dir}")
    print(f"{len(files)} task files", flush=True)

    jobs = [(f, args.output_dir, args.num_levels, args.augs_per_level, args.seed, args.save_videos) for f in files]
    n = args.workers or min(mp.cpu_count(), len(files))
    if n > 1 and len(files) > 1:
        with mp.get_context("spawn").Pool(n) as pool:
            results = pool.starmap(process_task, jobs)
    else:
        results = [process_task(*j) for j in jobs]

    failed = [f for f, r in zip(files, results) if r is None]
    if failed:
        raise SystemExit(f"{len(failed)} task(s) failed: {failed}")


if __name__ == "__main__":
    main()
