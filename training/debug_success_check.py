"""Diagnostic: replay one ORIGINAL expert demo (not your trained policy) through
the same env-building/success-checking code eval_libero.py uses, to confirm
the success detection itself works before blaming the trained policy.

If this script also reports "success=False" for a real expert demonstration,
the bug is in the success-checking mechanism, not your model.

Usage:
    python -m training.debug_success_check \
        --task_file ~/data/LIBERO-datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np

from training.eval_libero import build_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=str, required=True)
    parser.add_argument("--demo_key", type=str, default="demo_0")
    args = parser.parse_args()

    with h5py.File(args.task_file, "r") as f:
        orig_states = f[f"data/{args.demo_key}/states"][:]
        orig_actions = f[f"data/{args.demo_key}/actions"][:]

    env, task_name = build_env(args.task_file)
    env.reset()
    env.sim.set_state_from_flattened(orig_states[0])
    env.sim.forward()

    has_check = hasattr(env, "_check_success")
    print(f"Task: {task_name}")
    print(f"env has _check_success(): {has_check}")

    success_ever = False
    success_step = None
    for step, action in enumerate(orig_actions):
        obs, reward, done, info = env.step(action)

        step_success = False
        if has_check:
            step_success = env._check_success()
        elif isinstance(info, dict) and "success" in info:
            step_success = bool(info["success"])

        if step_success and not success_ever:
            success_ever = True
            success_step = step

        if step % 20 == 0:
            print(f"  step={step} reward={reward} done={done} check_success={step_success}")

    print("=" * 50)
    print(f"Expert demo replay result: success_ever={success_ever} at step={success_step}")
    if not success_ever:
        print(
            "WARNING: even the expert's own recorded actions never registered as a "
            "success. This means the success-detection mechanism itself needs a fix, "
            "NOT that your trained policy is failing. Check what `info` actually "
            "contains, or whether this LIBERO version exposes success differently "
            "(e.g. reward==1.0 signaling success for sparse-reward tasks)."
        )
    env.close()


if __name__ == "__main__":
    main()
