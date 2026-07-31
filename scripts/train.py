"""Fine-tuning classifiers on synthetic data (C1–C4) or real data (B1).

Encoder classifiers (ModernBERT, BERT) are trained via HuggingFace Trainer with full fine-tuning.

Usage:
    python scripts/train.py --task t1 --config c4 --classifier modernbert --run 1 --seed 42
    python scripts/train.py --task t1 --config b1 --classifier bert --run 1 --seed 43

Output:
    outputs/models/{task}/{config}/{classifier}/run{n}/seed{s}/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _shared import (
    CLASSIFIER_MAP,
    DATA_DIR,
    ROOT,
    TASK_DATASET_MAP,
    TRAINING_DIR,
    load_experiment_config,
)

MODELS_DIR = ROOT / "outputs" / "models"


def _train_encoder(
    df: pd.DataFrame,
    experiment_cfg: dict,
    model_name: str,
    seed: int,
    out_dir: Path,
) -> None:
    """Full fine-tuning for encoder models via HuggingFace Trainer."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    labels = sorted(df["label"].unique().tolist())
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    ds = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "label"}))

    max_seq_length = experiment_cfg.get("max_seq_length", 512)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_seq_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    batch_size = experiment_cfg["per_device_batch_size"]
    grad_accum = experiment_cfg["gradient_accumulation_steps"]
    epochs = experiment_cfg["epochs"]
    steps_per_epoch = len(ds) // (batch_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_ratio = experiment_cfg.get("warmup_ratio", 0.06)
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=experiment_cfg["learning_rate"],
        weight_decay=experiment_cfg.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        seed=seed,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"\n  Starting full fine-tuning with HuggingFace Trainer...")
    trainer.train()
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))

    # Save label mapping
    with open(out_dir / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print("  Training complete!")


def train(task: str, config: str, classifier: str, run: int, seed_override: int | None = None) -> None:
    experiment_cfg = load_experiment_config()
    model_name = CLASSIFIER_MAP[classifier]
    seed = seed_override if seed_override is not None else experiment_cfg["seed"]

    out_dir = MODELS_DIR / task / config / classifier / f"run{run}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {task.upper()} | Config: {config.upper()} | Classifier: {classifier}")
    print(f"Model: {model_name} | Run: {run} | Seed: {seed}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    if config == "b1":
        print("  Loading real training data (B1 baseline)...")
        data_path = DATA_DIR / TASK_DATASET_MAP[task] / "train.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                f"No real data at {data_path}. Run prepare_real.py first."
            )
        df = pd.read_csv(data_path)
    else:
        print(f"  Loading prepared training data ({config}/run{run})...")
        data_path = TRAINING_DIR / task / config / f"run{run}" / "train.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                f"No training data at {data_path}. Run prepare_synthetic.py first."
            )
        df = pd.read_csv(data_path)

    print(f"  Dataset: {len(df)} samples, {df['label'].nunique()} classes")
    print(f"  Distribution:\n{df['label'].value_counts().to_string()}\n")

    _train_encoder(df, experiment_cfg, model_name, seed, out_dir)

    metadata = {
        "task": task,
        "config": config,
        "classifier": classifier,
        "model": model_name,
        "run": run,
        "seed": seed,
        "n_samples": len(df),
        "training_params": {
            "learning_rate": experiment_cfg["learning_rate"],
            "epochs": experiment_cfg["epochs"],
            "batch_size": experiment_cfg["per_device_batch_size"],
            "max_seq_length": experiment_cfg.get("max_seq_length", 512),
            "weight_decay": experiment_cfg.get("weight_decay", 0.01),
            "warmup_ratio": experiment_cfg.get("warmup_ratio", 0.06),
        },
    }
    with open(out_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune classifier")
    parser.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    parser.add_argument("--config", required=True,
                        help="Config name (b1, c1-c4, or sensitivity variant e.g. c1_claude-haiku-4.5)")
    parser.add_argument("--classifier", required=True, choices=["modernbert", "bert"])
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--seed", type=int, default=None, help="Override seed from experiment.yaml")
    args = parser.parse_args()

    train(args.task, args.config, args.classifier, args.run, seed_override=args.seed)


if __name__ == "__main__":
    main()
