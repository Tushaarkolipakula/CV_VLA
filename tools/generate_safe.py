"""Generate curriculum files without overwriting completed datasets.

Each task is staged separately, validated, and published only after success.
Existing files require --skip-existing; no command in this wrapper overwrites them.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_levels", type=int, default=5)
    parser.add_argument("--augs_per_level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.target_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if source == output:
        parser.error("Source and output directories must differ")
    if args.num_levels < 1 or args.augs_per_level < 1:
        parser.error("Levels and augmentations must be positive")
    files = [p for p in sorted(source.glob("*.hdf5")) if not args.only or args.only in p.name]
    if not files:
        parser.error("No matching source files")
    collisions = [p.name for p in files if (output / p.name).exists()]
    if collisions and not args.skip_existing:
        parser.error(f"Output files already exist: {collisions}. Choose a new version directory or use --skip-existing.")
    jobs = [p for p in files if not (output / p.name).exists()]
    print(json.dumps({"generate": [p.name for p in jobs], "skip": collisions,
                      "output": str(output)}, indent=2), flush=True)
    if args.dry_run:
        return
    # Import simulation dependencies only after the plan and collision checks.
    import h5py
    from augmentation.generate_curriculum_dataset import process_task
    output.mkdir(parents=True, exist_ok=True)
    for path in jobs:
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output))
        try:
            stats = process_task(str(path), str(staging), args.num_levels,
                                 args.augs_per_level, args.seed, args.save_videos)
            if stats is None:
                raise RuntimeError(f"Generation failed for {path}; staging retained at {staging}")
            artifact = staging / path.name
            manifest = {"source": str(path), "source_bytes": path.stat().st_size,
                        "source_mtime_ns": path.stat().st_mtime_ns,
                        "num_levels": args.num_levels, "augs_per_level": args.augs_per_level,
                        "seed": args.seed, "stats": stats,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "generator_sha256": hashlib.sha256((ROOT / "augmentation/generate_curriculum_dataset.py").read_bytes()).hexdigest()}
            with h5py.File(path, "r") as original, h5py.File(artifact, "r+") as generated:
                data = generated["data"]
                for key in original["data"]:
                    if key not in data or data[key]["actions"].shape != original["data"][key]["actions"].shape:
                        raise RuntimeError(f"Missing or incomplete clean demo: {key}")
                for key, demo in data.items():
                    n = len(demo["actions"])
                    for obs_key in ("agentview_rgb", "eye_in_hand_rgb", "gripper_states", "ee_pos", "ee_ori"):
                        if len(demo["obs"][obs_key]) != n:
                            raise RuntimeError(f"Observation/action length mismatch: {key}/{obs_key}")
                data.attrs["generation_manifest"] = json.dumps(manifest, sort_keys=True)
                data.attrs["generation_complete"] = True
            # Hard-link publication is atomic and refuses a concurrent existing file.
            os.link(artifact, output / path.name)
            if (staging / "videos").exists():
                shutil.move(str(staging / "videos"), str(output / (path.stem + "_videos_" + staging.name)))
            shutil.rmtree(staging)
            print(f"Published {output / path.name}: {stats}", flush=True)
        except Exception:
            print(f"Failed task staging retained at {staging}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
