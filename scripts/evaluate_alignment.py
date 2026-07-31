"""Evaluate text-attribute alignment using DeBERTa-v3-large NLI.

Uses a strictly larger model than the pipeline-internal scorer (DeBERTa-v3-base)
so that the optimization signal and evaluation measure do not coincide.

Usage:
    python scripts/evaluate_alignment.py --task t1 --config c4 --run 1

Output:
    outputs/metrics/{task}/{config}/run{n}/alignment.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

from _shared import ROOT, SYNTHETIC_DIR, normalize_synthetic_df

METRICS_DIR = ROOT / "outputs" / "metrics"

# External evaluation model (different from pipeline-internal DeBERTa-v3-base)
EVAL_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"


SKIP_COLUMNS = {"text"}


def load_generated_data(task: str, config: str, run: int) -> list[dict]:
    """Load all generated samples for a task/config/run.

    Reads FM constraint values directly from CSV columns (Label, Domain,
    StakeholderViewpoint, Context, etc.).  Columns are the union of all
    features used during generation; some rows may leave a subset empty.
    """
    run_dir = SYNTHETIC_DIR / task / config / f"run{run}"
    if not run_dir.exists():
        raise FileNotFoundError(f"No data found at {run_dir}")

    samples = []
    for label_dir in sorted(run_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name == "__pycache__":
            continue
        csv_path = label_dir / "data.csv"

        if csv_path.exists():
            df = normalize_synthetic_df(pd.read_csv(csv_path))
            feature_cols = [c for c in df.columns if c not in SKIP_COLUMNS]

            for _, row in df.iterrows():
                features = {}
                for col in feature_cols:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        features[col] = str(val).strip()

                samples.append({
                    "text": row.get("text", ""),
                    "features": features,
                    "label_dir": label_dir.name,
                })

    return samples


# Columns evaluated as constraints (label + FM constraint dimensions).
# Note: normalize_synthetic_df lowercases "Label" -> "label".
CONSTRAINT_COLUMNS = [
    "label", "RequirementType", "Domain", "StakeholderViewpoint", "Context",
    "DescriptionType", "Language", "AbstractionLevel",
]


def verbalize_features(features: dict) -> str:
    """Build an NLI hypothesis in the SDK's AlignScorer style.

    Example output:
        "The text satisfies all of the following constraints:
         Label is Security; Domain is Healthcare Information System;
         Context is Usage; Language is English; AbstractionLevel is HighLevel."
    """
    if not features:
        return ""

    statements = []
    for col in CONSTRAINT_COLUMNS:
        val = features.get(col, "")
        if val:
            statements.append(f"{col} is {val}")

    if not statements:
        return ""

    return (
        "The text satisfies all of the following constraints: "
        + "; ".join(statements)
        + "."
    )


def compute_alignment(samples: list[dict], nli_pipe) -> dict:
    """Compute mean NLI entailment score for all samples."""
    scores = []
    batch_size = 32

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        pairs = []
        for s in batch:
            hypothesis = verbalize_features(s["features"])
            if hypothesis:
                pairs.append({"text": s["text"], "text_pair": hypothesis})

        if not pairs:
            continue

        # NLI: pass premise/hypothesis pairs to text-classification pipeline
        results = nli_pipe(
            [{"text": p["text"], "text_pair": p["text_pair"]} for p in pairs],
            top_k=None,  # Return scores for all labels
        )

        for result in results:
            # DeBERTa NLI returns labels: entailment, neutral, contradiction
            entailment_score = 0.0
            if isinstance(result, list):
                for label_score in result:
                    if label_score["label"].lower() == "entailment":
                        entailment_score = label_score["score"]
                        break
            elif isinstance(result, dict):
                if result["label"].lower() == "entailment":
                    entailment_score = result["score"]
            scores.append(entailment_score)

        if (i // batch_size) % 10 == 0:
            print(f"    Processed {min(i + batch_size, len(samples))}/{len(samples)} samples")

    if not scores:
        return {"mean": 0.0, "count": 0}

    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "std": pd.Series(scores).std(),
        "count": len(scores),
        "scores": scores,
    }


def evaluate(task: str, config: str, run: int) -> None:
    print(f"\nEvaluating alignment: {task}/{config}/run{run}")

    samples = load_generated_data(task, config, run)
    print(f"  Loaded {len(samples)} samples")

    if not samples:
        print("  No samples found, skipping.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading {EVAL_MODEL} on {device}...")
    nli_pipe = pipeline(
        "text-classification",
        model=EVAL_MODEL,
        device=device,
    )

    result = compute_alignment(samples, nli_pipe)

    # Save (without individual scores to keep file manageable)
    out_dir = METRICS_DIR / task / config / f"run{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "scores"}
    with open(out_dir / "alignment.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Alignment: {result['mean']:.4f} (n={result['count']})")
    print(f"  Saved to {out_dir / 'alignment.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate alignment (NLI)")
    parser.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    evaluate(args.task, args.config, args.run)


if __name__ == "__main__":
    main()
