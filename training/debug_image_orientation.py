"""Dump three images to PNG so you can visually compare orientation:

1. `orig_raw_*.png`      -- a frame straight from LIBERO's ORIGINAL demo file,
                            completely unprocessed. This is presumably the
                            "ground truth correct" orientation.
2. `dataset_sample_*.png` -- the same frame as it's actually stored in your
                            AUGMENTED dataset (what the model trains on).
3. `eval_live_*.png`      -- a frame straight from a live env reset, processed
                            exactly the way eval_libero.py's build_image() does.

If (1) and (2) don't visually match (e.g. one looks upside-down or mirrored
relative to the other), that confirms the flip bug described in chat. If (2)
and (3) don't match each other, that's the train/eval mismatch actually
causing the bad rollouts.

Usage:
    python -m training.debug_image_orientation \
        --orig_task_file ~/data/LIBERO-datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5 \
        --augmented_task_file ~/data/LIBERO-datasets-augmented/libero_goal/put_the_bowl_on_the_plate_demo.hdf5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
from PIL import Image

from training.eval_libero import build_env, build_image


def save(arr: np.ndarray, path: str):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"Saved {path}  shape={arr.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_task_file", type=str, required=True)
    parser.add_argument("--augmented_task_file", type=str, required=True)
    parser.add_argument("--demo_key", type=str, default="demo_1")
    parser.add_argument("--frame_idx", type=int, default=10)
    args = parser.parse_args()

    with h5py.File(args.orig_task_file, "r") as f:
        raw_frame = np.asarray(f["data/demo_0/obs/agentview_rgb"][args.frame_idx])
        state = np.asarray(f["data/demo_0/states"][args.frame_idx])

    with h5py.File(args.augmented_task_file, "r") as f:
        stored_frame = np.asarray(f[f"data/{args.demo_key}/obs/agentview_rgb"][0])

    env, _ = build_env(args.orig_task_file)
    env.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    obs = env._get_observations()
    live_processed = build_image(obs)
    env.close()

    save(raw_frame, "orig_raw_frame.png")
    save(np.transpose(live_processed, (1, 2, 0)), "eval_live_same_state_frame.png")
    save(stored_frame, "augmented_demo0_frame.png")

    print()
    print("Expected:")
    print("  orig_raw_frame.png            = raw LIBERO HDF5 image (typically inverted)")
    print("  eval_live_same_state_frame.png = same simulator state after canonical 180-degree preprocessing")
    print("  augmented_demo0_frame.png      = canonical image stored by the augmentation pipeline")
    print()
    print("The second and third images should have the same orientation.")
    print("The raw image should look like the 180-degree inverse of them.")


if __name__ == "__main__":
    main()
