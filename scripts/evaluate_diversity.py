"""Evaluate diversity of generated synthetic data.

Metrics:
  - INGF: Inter-sample N-gram Frequency (Eq. 4) — lexical diversity
  - APS:  Average Pairwise Similarity (Eq. 5)  — semantic diversity
Lower values = higher diversity for both metrics.

Usage:
    python scripts/evaluate_diversity.py --task t1 --config c4 --run 1

Output:
    outputs/metrics/{task}/{config}/run{n}/diversity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from _shared import ROOT, SYNTHETIC_DIR, normalize_synthetic_df
METRICS_DIR = ROOT / "outputs" / "metrics"

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_texts(task: str, config: str, run: int) -> list[str]:
    """Load all generated texts for a task/config/run."""
    run_dir = SYNTHETIC_DIR / task / config / f"run{run}"
    if not run_dir.exists():
        raise FileNotFoundError(f"No data found at {run_dir}")

    texts = []
    for label_dir in sorted(run_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        csv_path = label_dir / "data.csv"
        if csv_path.exists():
            df = normalize_synthetic_df(pd.read_csv(csv_path))
            texts.extend(df["text"].dropna().tolist())

    return texts


def compute_trigrams(text: str) -> set[str]:
    """Extract trigrams from text."""
    words = text.lower().split()
    if len(words) < 3:
        return set()
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def compute_ingf(texts: list[str]) -> float:
    """Inter-sample N-gram Frequency (Eq. 4).

    For each unique trigram, compute the fraction of samples containing it.
    INGF = mean of these fractions. Lower = more diverse.
    """
    all_trigrams: set[str] = set()
    sample_trigrams: list[set[str]] = []

    for text in texts:
        tg = compute_trigrams(text)
        all_trigrams.update(tg)
        sample_trigrams.append(tg)

    if not all_trigrams:
        return 0.0

    n_samples = len(texts)
    freq_sum = 0.0
    for trigram in all_trigrams:
        count = sum(1 for st in sample_trigrams if trigram in st)
        freq_sum += count / n_samples

    return freq_sum / len(all_trigrams)


def compute_aps(texts: list[str], model: SentenceTransformer, batch_size: int = 256) -> float:
    """Average Pairwise Similarity (Eq. 5).

    Mean cosine similarity over all sample pairs. Lower = more diverse.
    For large datasets, uses a random sample of pairs to keep computation tractable.
    """
    print(f"    Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms

    n = len(embeddings)
    if n < 2:
        return 0.0

    # For large n, computing all pairs is O(n^2). Use matrix multiplication.
    # sim_matrix[i,j] = cosine similarity between i and j
    if n <= 10000:
        sim_matrix = embeddings @ embeddings.T
        # Extract upper triangle (excluding diagonal)
        upper_indices = np.triu_indices(n, k=1)
        similarities = sim_matrix[upper_indices]
        return float(np.mean(similarities))
    else:
        # For very large datasets, sample pairs in chunks to avoid OOM
        max_pairs = 5_000_000  # 5M pairs (sufficient for stable estimate)
        n_pairs = n * (n - 1) // 2
        print(f"    Sampling {max_pairs:,} of {n_pairs:,} pairs...")
        rng = np.random.default_rng(42)
        chunk_size = 500_000  # Process 500K pairs at a time
        sim_sum = 0.0
        sim_count = 0
        for _ in range(0, max_pairs, chunk_size):
            k = min(chunk_size, max_pairs - sim_count)
            idx_i = rng.integers(0, n, size=k)
            idx_j = rng.integers(0, n, size=k)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            sim_sum += float(np.sum(
                np.sum(embeddings[idx_i] * embeddings[idx_j], axis=1)
            ))
            sim_count += len(idx_i)
        return sim_sum / sim_count if sim_count else 0.0


def evaluate(task: str, config: str, run: int) -> None:
    print(f"\nEvaluating diversity: {task}/{config}/run{run}")

    texts = load_texts(task, config, run)
    print(f"  Loaded {len(texts)} texts")

    if len(texts) < 2:
        print("  Too few samples, skipping.")
        return

    # INGF
    print("  Computing INGF...")
    ingf = compute_ingf(texts)

    # APS
    print("  Computing APS...")
    model = SentenceTransformer(SBERT_MODEL)
    aps = compute_aps(texts, model)

    result = {
        "ingf": ingf,
        "aps": aps,
        "n_samples": len(texts),
    }

    out_dir = METRICS_DIR / task / config / f"run{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diversity.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  INGF: {ingf:.6f} (lower = more diverse)")
    print(f"  APS:  {aps:.4f} (lower = more diverse)")
    print(f"  Saved to {out_dir / 'diversity.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate diversity (INGF + APS)")
    parser.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    evaluate(args.task, args.config, args.run)


if __name__ == "__main__":
    main()
