"""Refresh a compact inventory of datasets and training runs (no GPU needed)."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py


def inspect(root):
    report = {"updated_utc": datetime.now(timezone.utc).isoformat(),
              "project": str(root), "datasets": [], "runs": [], "warnings": []}
    for path in sorted((root / "data").rglob("*.hdf5")):
        entry = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
        try:
            with h5py.File(path, "r") as handle:
                data = handle["data"]
                counts, frames = Counter(), Counter()
                for name, demo in data.items():
                    kind = str(demo.attrs.get("kind", "clean"))
                    level = int(demo.attrs.get("noise_level", 0))
                    label = f"{kind}/level{level}"
                    counts[label] += 1
                    frames[label] += len(demo["actions"])
                entry.update(demos=dict(counts), frames=dict(frames),
                             image_flips=str(data.attrs.get("image_flips", "missing")))
        except Exception as exc:
            entry["error"] = str(exc)
        report["datasets"].append(entry)
    for path in sorted((root / "checkpoints").glob("*/history.json")):
        entry = {"run": path.parent.name, "history": str(path.relative_to(root))}
        try:
            history = json.loads(path.read_text())
            scored = [r for r in history if r.get("val_loss") is not None]
            entry.update(epochs_recorded=len(history), latest=history[-1] if history else None,
                         best_validation=min(scored, key=lambda r: r["val_loss"]) if scored else None,
                         checkpoints=[p.name for p in sorted(path.parent.glob("*.pt"))])
        except Exception as exc:
            entry["error"] = str(exc)
        report["runs"].append(entry)
    plus = root.parent / "LIBERO-plus"
    report["libero_plus"] = {"checkout_exists": plus.exists(),
                             "assets_exist": (plus / "libero/libero/assets").is_dir()}
    report["evaluation_results"] = [str(p.relative_to(root)) for p in sorted((root / "results").rglob("*.json"))
                                    if p.name != "inventory.json"]
    report["warnings"].append("Validation loss is not rollout success rate. Compare frozen checkpoints on identical initial states.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report = inspect(root)
    out = args.out or root / "results/inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(out)
    print(f"Inventory: {out}")
    print(f"{len(report['datasets'])} dataset files; {len(report['runs'])} training runs")
    for run in report["runs"]:
        best = run.get("best_validation")
        print(run["run"], "epochs:", run.get("epochs_recorded"),
              "best epoch:", best.get("epoch") if best else None,
              "best val loss:", best.get("val_loss") if best else None)


if __name__ == "__main__":
    main()
