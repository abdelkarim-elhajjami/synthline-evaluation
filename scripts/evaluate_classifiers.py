"""Evaluate fine-tuned classifiers on the held-out test set.

Loads fine-tuned encoder model (ModernBERT, BERT) via HuggingFace.
Computes weighted F1/precision/recall.

Usage:
    python scripts/evaluate_classifiers.py --task t1 --config c4 --classifier modernbert --run 1 --seed 42

Output:
    outputs/predictions/{task}/{config}/{classifier}/run{n}/seed{s}/metrics.json
    outputs/predictions/{task}/{config}/{classifier}/run{n}/seed{s}/predictions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from _shared import (
    CLASSIFIER_MAP,
    ROOT,
    load_experiment_config,
    load_test_data,
)

MODELS_DIR = ROOT / "outputs" / "models"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"


def classify_with_encoder(
    texts: list[str],
    model_path: str,
    max_seq_length: int = 512,
) -> list[str]:
    """Run classification inference using a fine-tuned encoder model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    # Load label mapping (required — created by train.py)
    label_map_path = Path(model_path).parent / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"No label_map.json at {label_map_path}. Training may not have completed."
        )
    with open(label_map_path) as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}

    predictions = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_seq_length, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        for pid in pred_ids:
            predictions.append(id2label.get(pid, str(pid)))

        if (i + batch_size) % 100 < batch_size:
            print(f"    Classified {min(i + batch_size, len(texts))}/{len(texts)}")

    return predictions


def evaluate(task: str, config: str, classifier: str, run: int, seed_override: int | None = None) -> None:
    model_name = CLASSIFIER_MAP[classifier]
    exp_cfg = load_experiment_config()
    seed = seed_override if seed_override is not None else exp_cfg["seed"]

    run_dir = MODELS_DIR / task / config / classifier / f"run{run}" / f"seed{seed}"
    out_dir = PREDICTIONS_DIR / task / config / classifier / f"run{run}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating: {task.upper()}/{config.upper()}/{classifier}/run{run}/seed{seed}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    test_df = load_test_data(task)
    print(f"  Test set: {len(test_df)} samples")

    # Read max_seq_length: prefer training metadata, fall back to experiment config
    max_seq_length = exp_cfg.get("max_seq_length", 512)
    metadata_path = run_dir / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        max_seq_length = metadata.get("training_params", {}).get("max_seq_length", max_seq_length)

    print("  Running inference...")
    model_path = run_dir / "model"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Run train.py first.")
    predictions = classify_with_encoder(
        texts=test_df["text"].tolist(),
        model_path=str(model_path),
        max_seq_length=max_seq_length,
    )

    # Compute metrics
    y_true = test_df["label"].tolist()
    y_pred = predictions

    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    metrics = {
        "task": task,
        "config": config,
        "classifier": classifier,
        "model": model_name,
        "run": run,
        "seed": seed,
        "f1_weighted": f1_w,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "classification_report": report,
        "n_test": len(test_df),
    }

    # Save
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({"text": test_df["text"], "true": y_true, "predicted": y_pred})
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n  Results:")
    print(f"    F1 (weighted):        {f1_w:.4f}")
    print(f"    Precision (weighted): {precision_w:.4f}")
    print(f"    Recall (weighted):    {recall_w:.4f}")
    print(f"  Saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifiers")
    parser.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    parser.add_argument("--config", required=True,
                        help="Config name (b1, c1-c4, or sensitivity variant e.g. c1_claude-haiku-4.5)")
    parser.add_argument("--classifier", required=True, choices=["modernbert", "bert"])
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Override seed from experiment.yaml")
    args = parser.parse_args()

    evaluate(args.task, args.config, args.classifier, args.run, seed_override=args.seed)


if __name__ == "__main__":
    main()
