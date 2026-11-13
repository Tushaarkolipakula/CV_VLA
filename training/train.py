"""Train a BC policy on the generated curriculum dataset.

Run from ~/cv_vla:
    python -m training.train \
        --dataset_dir ~/cv_vla/data/LIBERO-datasets-augmented/libero_goal \
        --checkpoint_dir ~/cv_vla/checkpoints/run1 --epochs 100
"""
from __future__ import annotations

import argparse

from training.train_curriculum import train_with_curriculum


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--recovery_ratio", type=float, default=0.3, help="Fraction of each epoch drawn from recovery frames")
    p.add_argument("--unlock_fraction", type=float, default=0.6, help="All noise levels unlocked by this fraction of training")
    p.add_argument("--samples_per_epoch", type=int, default=0, help="0 = size of currently unlocked train set")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean_action_offset", type=int, choices=(0, 1), default=0,
                   help="Use 1 for verified post-action clean observations; recovery pairs stay unchanged")
    a = p.parse_args()

    train_with_curriculum(
        dataset_dir=a.dataset_dir,
        checkpoint_dir=a.checkpoint_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        val_fraction=a.val_fraction,
        recovery_ratio=a.recovery_ratio,
        unlock_fraction=a.unlock_fraction,
        samples_per_epoch=a.samples_per_epoch,
        num_workers=a.num_workers,
        save_every=a.save_every,
        seed=a.seed,
        clean_action_offset=a.clean_action_offset,
    )


if __name__ == "__main__":
    main()
