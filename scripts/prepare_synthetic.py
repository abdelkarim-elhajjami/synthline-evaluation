"""Prepare synthetic data for training.

Normalizes raw Synthline SDK output (capitalized columns, per-label directories)
into a single train.csv with lowercase text/label columns.

For T2 (FR/NFR binary), assembles T1 NFR subtypes into a single Non-Functional
class and concatenates with T2 Functional data.

Usage:
    python scripts/prepare_synthetic.py --task t3 --config c1 --run 1

Output:
    outputs/training/{task}/{config}/run{n}/train.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _shared import SYNTHETIC_DIR, TRAINING_DIR, normalize_synthetic_df


def _dir_name_to_label(dir_name: str, label_map: dict | None) -> str:
    """Convert directory name back to original label, using label map if available."""
    if label_map and dir_name in label_map:
        return label_map[dir_name]
    return dir_name.replace("_", " ").title()


def _load_raw(task: str, config: str, run: int) -> pd.DataFrame:
    """Load raw SDK output for a single task/config/run."""
    run_dir = SYNTHETIC_DIR / task / config / f"run{run}"
    if not run_dir.exists():
        raise FileNotFoundError(
            f"No synthetic data at {run_dir}. Run generate.py first."
        )

    label_map_path = run_dir / "label_map.json"
    label_map = None
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)

    dfs = []
    for label_dir in sorted(run_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        csv_path = label_dir / "data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = normalize_synthetic_df(df)
            label = _dir_name_to_label(label_dir.name, label_map)
            df["label"] = label
            dfs.append(df[["text", "label"]])

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {run_dir}")

    return pd.concat(dfs, ignore_index=True)


def prepare(task: str, config: str, run: int) -> None:
    print(f"\n{'='*60}")
    print(f"Preparing training data: {task.upper()}/{config.upper()}/run{run}")
    print(f"{'='*60}\n")

    df = _load_raw(task, config, run)

    if task == "t2":
        print("  Assembling T2: loading T1 NFR data as Non-Functional class...")
        t1_df = _load_raw("t1", config, run)
        t1_df["label"] = "Non-Functional"
        df = pd.concat([df, t1_df], ignore_index=True)

    out_dir = TRAINING_DIR / task / config / f"run{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.csv"
    df.to_csv(out_path, index=False)

    print(f"  Samples: {len(df)}")
    print(f"  Labels:  {df['label'].nunique()}")
    print(f"  Distribution:\n{df['label'].value_counts().to_string()}\n")
    print(f"  Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare synthetic training data")
    parser.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    parser.add_argument("--config", required=True,
                        help="Config name (c1, c2, c3, c4, or sensitivity variant e.g. c1_claude-haiku-4.5)")
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    prepare(args.task, args.config, args.run)


if __name__ == "__main__":
    main()
