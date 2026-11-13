"""
Visualize curriculum-augmented rollouts to sanity-check that the
perturb -> recover stitching in generate_curriculum_dataset.py is actually
bringing the trajectory back to the original expert's state/behavior.

Usage:
    python -m visualization.visualize_rollout \
        --orig_file data/LIBERO-datasets/libero_goal/put_the_wine_bottle_on_the_rack_demo.hdf5 \
        --aug_file data/LIBERO-datasets-augmented/libero_goal/put_the_wine_bottle_on_the_rack_demo.hdf5 \
        --demo demo_0 \
        --out_dir results/rollouts

Run this from the repo root (so `cv_vla` / `visualization` import correctly).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def load_demo(h5_path, demo_key):
    with h5py.File(h5_path, "r") as f:
        grp = f[f"data/{demo_key}"]
        frames = grp["obs"]["agentview_rgb"][:]
        ee_pos = grp["obs"]["ee_pos"][:]
        states = grp["states"][:]
    return frames, ee_pos, states


def save_video(frames, out_path, fps=20):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # frames are uint8 HxWx3 already (agentview_rgb straight from the sim)
    imageio.mimsave(out_path, [f.astype(np.uint8) for f in frames], fps=fps)
    print(f"Saved rollout video -> {out_path}")


def compare_trajectories(orig_ee_pos, aug_ee_pos, out_path):
    """Overlay the augmented (perturbed+recovered) ee_pos path against the
    original expert path, and report how far apart the *final* positions are.
    If curriculum recovery is working, the augmented trajectory should
    reconverge onto the original path after the perturbed timestep, and the
    final ee_pos values should match closely (task still completes the same
    way)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(orig_ee_pos[:, i], label="original (unperturbed)", color="#2563eb")
        ax.plot(aug_ee_pos[:, i], label="augmented (perturb+recover)", color="#dc2626", linestyle="--")
        ax.set_title(f"ee_pos[{labels[i]}]")
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory comparison plot -> {out_path}")

    final_orig = orig_ee_pos[-1]
    final_aug = aug_ee_pos[-1]
    delta = np.linalg.norm(final_orig - final_aug)
    print(f"Final ee_pos (original):  {final_orig}")
    print(f"Final ee_pos (augmented): {final_aug}")
    print(f"Final position delta:     {delta:.5f} m")
    if delta > 0.03:
        print("WARNING: final position differs by >3cm — the augmented rollout "
              "may not actually be recovering back to the original task outcome. "
              "Check t_thresh / avg_mag for this epoch.")
    else:
        print("Looks consistent: augmented rollout reconverges near the original final position.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_file", required=True, help="Path to the *original* (un-augmented) demo HDF5")
    ap.add_argument("--aug_file", required=True, help="Path to the curriculum-augmented HDF5 (output of generate_curriculum_dataset.py)")
    ap.add_argument("--demo", default="demo_0", help="Demo key inside the augmented file, e.g. demo_0")
    ap.add_argument("--orig_demo", default="demo_0", help="Demo key inside the original file to compare against")
    ap.add_argument("--out_dir", default="results/rollouts")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    aug_frames, aug_ee_pos, aug_states = load_demo(args.aug_file, args.demo)
    save_video(aug_frames, out_dir / f"{args.demo}_rollout.mp4", fps=args.fps)

    orig_frames, orig_ee_pos, orig_states = load_demo(args.orig_file, args.orig_demo)
    save_video(orig_frames, out_dir / f"{args.orig_demo}_original.mp4", fps=args.fps)

    n = min(len(orig_ee_pos), len(aug_ee_pos))
    compare_trajectories(orig_ee_pos[:n], aug_ee_pos[:n], out_dir / f"{args.demo}_trajectory_compare.png")


if __name__ == "__main__":
    main()
