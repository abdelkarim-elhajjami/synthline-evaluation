"""Download real datasets and create stratified 80/20 train/test splits.

Usage:
    python scripts/prepare_real.py

Sources:
    PROMISE_exp: GitHub (AleksandarMitrevski/se-requirements-classification)
    SecReq:      Zenodo (record 4530183)
    CiRA:        GitHub (fischJan/CiRA)

    If downloads fail, place preprocessed files manually:
      data/real/promise_exp/full.csv   (columns: ProjectID, text, class)
      data/real/secreq/full.csv        (columns: text, label)
      data/real/cira/full.csv          (columns: text, label)

Outputs:
    data/real/{dataset}/t{n}/train.csv
    data/real/{dataset}/t{n}/test.csv
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import pandas as pd
from sklearn.model_selection import train_test_split

from _shared import DATA_DIR
SEED = 42
TEST_SIZE = 0.2

# PROMISE_exp class code → full label name
PROMISE_CLASS_MAP = {
    "F": "Functional",
    "SE": "Security",
    "US": "Usability",
    "O": "Operational",
    "PE": "Performance",
    "LF": "Look and Feel",
    "A": "Availability",
    "MN": "Maintainability",
    "SC": "Scalability",
    "FT": "Fault Tolerance",
    "L": "Legal",
    "PO": "Portability",
}

# ── Download helpers ──────────────────────────────────────────────────


def _download(url: str, encoding: str = "utf-8") -> str:
    """Download text content from a URL."""
    req = Request(url, headers={"User-Agent": "synthline-evaluation/1.0"})
    with urlopen(req, timeout=60) as resp:
        raw = resp.read()
    # Try requested encoding first, fallback to latin-1
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _parse_arff(content: str) -> pd.DataFrame:
    """Parse ARFF format into a DataFrame (handles PROMISE_exp format)."""
    lines = content.split("\n")
    data_started = False
    rows = []

    for line in lines:
        line = line.strip()
        if line.upper() == "@DATA":
            data_started = True
            continue
        if not data_started or not line or line.startswith("%"):
            continue

        # ARFF format: ProjectID,'RequirementText',_class_
        # Text is quoted with single quotes and may contain commas
        match = re.match(r"(\d+),\s*'(.+)',\s*(\w+)", line)
        if match:
            project_id = int(match.group(1))
            text = match.group(2).strip()
            cls = match.group(3).strip()
            rows.append({"ProjectID": project_id, "text": text, "class": cls})

    return pd.DataFrame(rows)


def _parse_secreq_csv(content: str) -> list[dict]:
    """Parse SecReq semicolon-separated format: text;label."""
    rows = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Last semicolon-separated token is the label
        parts = line.rsplit(";", 1)
        if len(parts) == 2:
            text = parts[0].strip().strip('"')
            label_raw = parts[1].strip().lower()
            if label_raw in ("sec", "nonsec"):
                label = "Security" if label_raw == "sec" else "Non-Security"
                rows.append({"text": text, "label": label})
    return rows


# ── PROMISE_exp ───────────────────────────────────────────────────────

PROMISE_URL = (
    "https://raw.githubusercontent.com/AleksandarMitrevski/"
    "se-requirements-classification/master/0-datasets/PROMISE_exp/PROMISE_exp.arff"
)


def prepare_promise_exp() -> None:
    """PROMISE_exp: 969 requirements (444 FR + 525 NFR in 11 subtypes)."""
    out = DATA_DIR / "promise_exp"
    out.mkdir(parents=True, exist_ok=True)

    raw_path = out / "full.csv"
    if raw_path.exists():
        print(f"  Loading from local file: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print(f"  Downloading PROMISE_exp from GitHub...")
        try:
            content = _download(PROMISE_URL)
            df = _parse_arff(content)
            # Save raw CSV for future runs
            df.to_csv(raw_path, index=False)
            print(f"  Downloaded {len(df)} requirements, saved to {raw_path}")
        except (URLError, Exception) as e:
            print(f"\n  ERROR: Failed to download PROMISE_exp: {e}")
            print(f"  Place a CSV manually at: {raw_path}")
            sys.exit(1)

    # Map class codes to full names (check all values, not just the first row)
    if df["class"].isin(PROMISE_CLASS_MAP.keys()).all():
        df["class"] = df["class"].map(PROMISE_CLASS_MAP)

    # T1 subset: NFR only (525 rows, 11 subtypes)
    df_nfr = df[df["class"] != "Functional"].copy()
    df_nfr["label"] = df_nfr["class"]
    train_nfr, test_nfr = train_test_split(
        df_nfr, test_size=TEST_SIZE, random_state=SEED, stratify=df_nfr["label"]
    )

    t1_dir = out / "t1"
    t1_dir.mkdir(exist_ok=True)
    train_nfr[["text", "label"]].to_csv(t1_dir / "train.csv", index=False)
    test_nfr[["text", "label"]].to_csv(t1_dir / "test.csv", index=False)
    print(f"  T1: {len(train_nfr)} train, {len(test_nfr)} test ({df_nfr['label'].nunique()} classes)")

    # T2: Full dataset (FR vs NFR binary)
    df["label"] = df["class"].apply(
        lambda x: "Functional" if x == "Functional" else "Non-Functional"
    )
    train_full, test_full = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )

    t2_dir = out / "t2"
    t2_dir.mkdir(exist_ok=True)
    train_full[["text", "label"]].to_csv(t2_dir / "train.csv", index=False)
    test_full[["text", "label"]].to_csv(t2_dir / "test.csv", index=False)
    print(f"  T2: {len(train_full)} train, {len(test_full)} test (binary)")


# ── SecReq ────────────────────────────────────────────────────────────

SECREQ_URLS = {
    "CPN": "https://zenodo.org/api/records/4530183/files/CPN.csv/content",
    "ePurse": "https://zenodo.org/api/records/4530183/files/ePurse-selective.csv/content",
    "GPS": "https://zenodo.org/api/records/4530183/files/GPS.csv/content",
}


def prepare_secreq() -> None:
    """SecReq: 510 requirements (Security vs Non-Security)."""
    out = DATA_DIR / "secreq"
    out.mkdir(parents=True, exist_ok=True)

    raw_path = out / "full.csv"
    if raw_path.exists():
        print(f"  Loading from local file: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print(f"  Downloading SecReq from Zenodo...")
        all_rows = []
        try:
            for spec_name, url in SECREQ_URLS.items():
                content = _download(url)
                rows = _parse_secreq_csv(content)
                print(f"    {spec_name}: {len(rows)} requirements")
                all_rows.extend(rows)
        except (URLError, Exception) as e:
            print(f"\n  ERROR: Failed to download SecReq: {e}")
            print(f"  Place a CSV manually at: {raw_path}")
            sys.exit(1)

        df = pd.DataFrame(all_rows)
        df.to_csv(raw_path, index=False)
        print(f"  Downloaded {len(df)} total requirements, saved to {raw_path}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )

    t3_dir = out / "t3"
    t3_dir.mkdir(exist_ok=True)
    train_df[["text", "label"]].to_csv(t3_dir / "train.csv", index=False)
    test_df[["text", "label"]].to_csv(t3_dir / "test.csv", index=False)
    print(f"  T3: {len(train_df)} train, {len(test_df)} test (binary)")


# ── CiRA ──────────────────────────────────────────────────────────────

CIRA_URL = (
    "https://raw.githubusercontent.com/fischJan/CiRA/main/"
    "annotated%20data%20sets/overall.csv"
)


def prepare_cira() -> None:
    """CiRA: Causal vs Non-Causal requirement sentences."""
    out = DATA_DIR / "cira"
    out.mkdir(parents=True, exist_ok=True)

    raw_path = out / "full.csv"
    if raw_path.exists():
        print(f"  Loading from local file: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        print(f"  Downloading CiRA from GitHub...")
        try:
            content = _download(CIRA_URL)
            df_raw = pd.read_csv(io.StringIO(content))

            # Normalize: rename 'Sentence' → 'text', map Causal 0/1 → labels
            df = pd.DataFrame({
                "text": df_raw["Sentence"],
                "label": df_raw["Causal"].map({1: "Causal", 0: "Non-Causal"}),
            })
            df = df.dropna(subset=["text", "label"])
            df.to_csv(raw_path, index=False)
            print(f"  Downloaded {len(df)} sentences, saved to {raw_path}")
        except (URLError, Exception) as e:
            print(f"\n  ERROR: Failed to download CiRA: {e}")
            print(f"  Place a CSV manually at: {raw_path}")
            sys.exit(1)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )

    t4_dir = out / "t4"
    t4_dir.mkdir(exist_ok=True)
    train_df[["text", "label"]].to_csv(t4_dir / "train.csv", index=False)
    test_df[["text", "label"]].to_csv(t4_dir / "test.csv", index=False)
    print(f"  T4: {len(train_df)} train, {len(test_df)} test (binary)")


# ── Main ──────────────────────────────────────────────────────────────


def _build_task_summary(name: str, dataset: str, task_type: str, task_dir: Path) -> dict:
    """Build a distribution summary for one task split."""
    train = pd.read_csv(task_dir / "train.csv")
    test = pd.read_csv(task_dir / "test.csv")
    return {
        "name": name,
        "dataset": dataset,
        "type": task_type,
        "total": len(train) + len(test),
        "train": len(train),
        "test": len(test),
        "num_classes": train["label"].nunique(),
        "distribution": {
            "train": {k: int(v) for k, v in train["label"].value_counts().sort_index().items()},
            "test": {k: int(v) for k, v in test["label"].value_counts().sort_index().items()},
        },
    }


def generate_summary() -> None:
    """Generate data/real/summary.json with per-task class distributions."""
    summary = {
        "T1": _build_task_summary("NFR Multi-class Classification", "PROMISE_exp", "multi-class", DATA_DIR / "promise_exp" / "t1"),
        "T2": _build_task_summary("FR/NFR Binary Classification", "PROMISE_exp", "binary", DATA_DIR / "promise_exp" / "t2"),
        "T3": _build_task_summary("Security Classification", "SecReq", "binary", DATA_DIR / "secreq" / "t3"),
        "T4": _build_task_summary("Causality Detection", "CiRA", "binary", DATA_DIR / "cira" / "t4"),
    }

    out_path = DATA_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


def main() -> None:
    print("Preparing datasets...\n")
    print("PROMISE_exp:")
    prepare_promise_exp()
    print("\nSecReq:")
    prepare_secreq()
    print("\nCiRA:")
    prepare_cira()
    generate_summary()
    print("\nDone. All splits saved to data/real/")


if __name__ == "__main__":
    main()
