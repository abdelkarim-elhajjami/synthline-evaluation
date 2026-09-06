"""Shared constants and utilities for experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
DOMAIN_DIR = ROOT / "domain"
DATA_DIR = ROOT / "data" / "real"
SYNTHETIC_DIR = ROOT / "outputs" / "synthetic"
TRAINING_DIR = ROOT / "outputs" / "training"

TASK_CONFIG_MAP = {
    "t1": "t1_nfr_multiclass.yaml",
    "t2": "t2_fr_nfr_binary.yaml",
    "t3": "t3_security.yaml",
    "t4": "t4_causality.yaml",
}

TASK_DATASET_MAP = {
    "t1": "promise_exp/t1",
    "t2": "promise_exp/t2",
    "t3": "secreq/t3",
    "t4": "cira/t4",
}

CLASSIFIER_MAP = {
    "modernbert": "answerdotai/ModernBERT-base",
    "bert": "bert-base-uncased",
}


def load_task_config(task: str) -> dict:
    with open(CONFIGS_DIR / TASK_CONFIG_MAP[task]) as f:
        return yaml.safe_load(f)


def load_experiment_config() -> dict:
    with open(CONFIGS_DIR / "experiment.yaml") as f:
        return yaml.safe_load(f)


def config_key(config: dict) -> str:
    filtered = {k: v for k, v in config.items() if k != "prompt"}
    return json.dumps(filtered, sort_keys=True)


def progress_bar(prefix: str = "", *, width: int = 30) -> Callable:
    tag = f"{prefix} " if prefix else ""

    async def _cb(progress: float, message: str) -> None:
        pct = max(0.0, min(100.0, progress))
        filled = int(width * pct / 100)
        bar = "\u2588" * filled + "\u2591" * (width - filled)
        print(f"\033[2K\r    [{bar}] {pct:3.0f}% {tag}{message}", end="", flush=True)

    return _cb


def verification_bar() -> Callable:
    state = {"last_accepted": 0}

    async def _cb(event) -> None:
        if event.accepted != state["last_accepted"]:
            state["last_accepted"] = event.accepted
            print(
                f"\033[2K\r    [Attempt {event.attempt}/{event.max_attempts}] "
                f"Accepted: {event.accepted}/{event.needed}",
                end="", flush=True,
            )

    return _cb


def load_test_data(task: str) -> pd.DataFrame:
    data_path = DATA_DIR / TASK_DATASET_MAP[task] / "test.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"No test data at {data_path}. Run prepare_real.py first."
        )
    return pd.read_csv(data_path)


def label_slug(name: str) -> str:
    """Convert a label name to a filesystem-safe slug."""
    return name.lower().replace(" ", "_")


def normalize_synthetic_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Synthline SDK column names to lowercase."""
    rename = {}
    if "Text" in df.columns and "text" not in df.columns:
        rename["Text"] = "text"
    if "Label" in df.columns and "label" not in df.columns:
        rename["Label"] = "label"
    return df.rename(columns=rename) if rename else df


def synthetic_source_tasks(task: str) -> tuple[str, ...]:
    """Raw pools used for training, in prepare_synthetic.py assembly order."""
    return ("t2", "t1") if task == "t2" else (task,)


def load_synthetic_frames(task: str, config: str, run: int) -> list[tuple[Path, pd.DataFrame]]:
    """Load the full training corpus while retaining generation attributes.

    T2 reuses all T1 NFR rows from the same config/run after its Functional
    rows. Keep duplicates and original labels here: binary training labels
    must not replace the generation attributes used for alignment evaluation.
    """
    frames = []
    for source_task in synthetic_source_tasks(task):
        run_dir = SYNTHETIC_DIR / source_task / config / f"run{run}"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing synthetic pool for {task}: {run_dir}")
        paths = [
            p / "data.csv" for p in sorted(run_dir.iterdir())
            if p.is_dir() and p.name != "__pycache__" and (p / "data.csv").is_file()
        ]
        if not paths:
            raise FileNotFoundError(f"No data.csv files in required pool: {run_dir}")
        count = 0
        for path in paths:
            df = normalize_synthetic_df(pd.read_csv(path))
            if "text" not in df.columns:
                raise ValueError(f"Missing text column: {path}")
            valid = df["text"].map(lambda text: isinstance(text, str) and bool(text.strip()))
            if not valid.all():
                raise ValueError(f"Missing, blank or non-string text in {path}")
            count += len(df)
            frames.append((path, df))
        if count == 0:
            raise ValueError(f"Required synthetic pool is empty: {run_dir}")
    return frames

