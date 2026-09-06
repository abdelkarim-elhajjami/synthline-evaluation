"""Recompute T2 intrinsic metrics on the exact FR/NFR training corpus.

Read-only inputs: the original synthline-experiments repository and local model.
Outputs: an auditable JSON result and resumable embedding cache outside that repo.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

CONFIGS = ("c1", "c2", "c3", "c4")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ingf(texts):
    counts = Counter()
    for text in texts:
        words = text.lower().split()
        counts.update({tuple(words[i:i + 3]) for i in range(len(words) - 2)})
    total = sum(counts.values())
    score = total / (len(texts) * len(counts)) if counts else 0.0
    return {"value": score, "unique_trigrams": len(counts), "document_ngram_incidences": total}


def exact_aps(embeddings):
    """Exact off-diagonal mean, via the sum-of-vectors identity in float64."""
    x = embeddings.astype(np.float64)
    n = len(x)
    if n < 2:
        return 0.0
    total = x.sum(axis=0)
    return float((total @ total - np.einsum("ij,ij->", x, x)) / (n * (n - 1)))


def sampled_aps(embeddings):
    """Same draws/reductions as evaluate_diversity.py; smaller dot-product blocks."""
    n = len(embeddings)
    rng = np.random.default_rng(42)
    score_sum, pair_count = 0.0, 0
    for _ in range(0, 5_000_000, 500_000):
        k = min(500_000, 5_000_000 - pair_count)
        left, right = rng.integers(0, n, size=k), rng.integers(0, n, size=k)
        keep = left != right
        left, right = left[keep], right[keep]
        scores = np.empty(len(left), dtype=np.float32)
        for start in range(0, len(left), 8192):
            end = min(start + 8192, len(left))
            scores[start:end] = np.sum(embeddings[left[start:end]] * embeddings[right[start:end]], axis=1)
        score_sum += float(np.sum(scores))
        pair_count += len(scores)
    return {"value": score_sum / pair_count, "draws": 5_000_000, "nonself_pairs": pair_count, "seed": 42}


def aps(embeddings):
    exact = exact_aps(embeddings)
    if len(embeddings) <= 10_000:
        return {"value": exact, "method": "all_distinct_pairs", "exact_check": exact}
    result = sampled_aps(embeddings)
    return {**result, "method": "sampled_pairs", "exact_check": exact, "sampling_error": result["value"] - exact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local all-mpnet-base-v2 snapshot")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    root = args.experiments.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "scripts"))
    import prepare_synthetic as prepare
    import evaluate_alignment as alignment

    provenance = {}

    def record(path):
        provenance[str(path.relative_to(root))] = file_sha(path)

    for name in ("prepare_synthetic.py", "evaluate_alignment.py", "evaluate_diversity.py", "_shared.py"):
        record(root / "scripts" / name)

    corpora, checks = {}, {}
    for config in CONFIGS:
        fr = prepare._load_raw("t2", config, 1)
        nfr = prepare._load_raw("t1", config, 1)
        nfr["label"] = "Non-Functional"
        assembled = pd.concat([fr, nfr], ignore_index=True)
        training_path = root / "outputs" / "training" / "t2" / config / "run1" / "train.csv"
        training = pd.read_csv(training_path)
        pd.testing.assert_frame_equal(assembled, training[["text", "label"]], check_dtype=False)
        assert assembled["text"].notna().all() and assembled["text"].str.strip().ne("").all()
        record(training_path)
        pieces = {}
        for task, frame in (("t2", fr), ("t1", nfr)):
            samples = alignment.load_generated_data(task, config, 1)
            assert [s["text"] for s in samples] == frame["text"].tolist()
            hypotheses = [alignment.verbalize_features(s["features"]) for s in samples]
            assert all(hypotheses)
            metric_dir = root / "outputs" / "metrics" / task / config / "run1"
            align_path, div_path = metric_dir / "alignment.json", metric_dir / "diversity.json"
            align = json.loads(align_path.read_text())
            div = json.loads(div_path.read_text())
            assert align["count"] == div["n_samples"] == len(frame)
            record(align_path)
            record(div_path)
            raw_dir = root / "outputs" / "synthetic" / task / config / "run1"
            for path in sorted(raw_dir.rglob("*.csv")):
                record(path)
            label_map = raw_dir / "label_map.json"
            if label_map.exists():
                record(label_map)
            pieces[task] = {
                "n": len(frame), "original_alignment": align, "original_diversity": div,
                "ordered_texts_sha256": digest(frame["text"].tolist()),
                "ordered_text_hypothesis_sha256": digest(list(zip(frame["text"], hypotheses))),
            }
        n = len(assembled)
        weighted = sum(p["n"] * p["original_alignment"]["mean"] for p in pieces.values()) / n
        checks[config] = {
            "n": n, "n_fr": len(fr), "n_nfr": len(nfr),
            "ordered_text_label_sha256": digest(assembled.values.tolist()),
            "training_equals_raw_assembly_in_order": True,
            "original_components": pieces,
            "alignment": {"value": weighted, "count": n, "method": "count_weighted_original_means"},
        }
        corpora[config] = assembled["text"].tolist()
        print(f"{config}: exact training match, FR={len(fr)}, NFR={len(nfr)}, alignment={weighted:.12f}", flush=True)

    # Encode each distinct string once; restore all repetitions and original order below.
    unique_texts = list(dict.fromkeys(text for config in CONFIGS for text in corpora[config]))
    model_files = {str(p.relative_to(args.model)): file_sha(p) for p in sorted(args.model.rglob("*")) if p.is_file()}
    fingerprint = digest({"texts": unique_texts, "model_files": model_files, "device": args.device, "batch": args.batch_size})
    vector_path = args.cache / f"{fingerprint}.npy"
    completed_path = args.cache / f"{fingerprint}.progress.json"
    print(f"Embedding {len(unique_texts)} distinct strings on {args.device} (multiplicity preserved in scores)", flush=True)
    model = SentenceTransformer(str(args.model), device=args.device, local_files_only=True)
    model.eval()
    dimension = model.get_sentence_embedding_dimension()
    if vector_path.exists() and completed_path.exists():
        vectors = np.lib.format.open_memmap(vector_path, mode="r+")
        done = json.loads(completed_path.read_text())["completed"]
        assert vectors.shape == (len(unique_texts), dimension)
    else:
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float32, shape=(len(unique_texts), dimension))
        done = 0
    for start in range(done, len(unique_texts), 2048):
        end = min(start + 2048, len(unique_texts))
        vectors[start:end] = model.encode(unique_texts[start:end], batch_size=args.batch_size, show_progress_bar=False)
        vectors.flush()
        completed_path.write_text(json.dumps({"completed": end, "total": len(unique_texts), "fingerprint": fingerprint}))
        print(f"Encoded {end}/{len(unique_texts)}", flush=True)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    assert np.isfinite(vectors).all() and (norms > 0).all()
    normalized = np.asarray(vectors) / norms
    # Verify the algebraic all-pairs identity against a direct off-diagonal mean.
    small = normalized[:64].astype(np.float64)
    direct = (small @ small.T)[np.triu_indices(len(small), k=1)].mean()
    assert abs(exact_aps(small) - direct) < 1e-12
    lookup = {text: i for i, text in enumerate(unique_texts)}
    for config in CONFIGS:
        row = checks[config]
        texts = corpora[config]
        embed = normalized[[lookup[text] for text in texts]]
        row["ingf"] = ingf(texts)
        row["aps"] = aps(embed)
        row["component_reproduction"] = {}
        for task, part, emb in (("t2", texts[:row["n_fr"]], embed[:row["n_fr"]]), ("t1", texts[row["n_fr"]:], embed[row["n_fr"]:])):
            original = row["original_components"][task]["original_diversity"]
            new_ingf, new_aps = ingf(part)["value"], aps(emb)["value"]
            assert abs(new_ingf - original["ingf"]) < 1e-12
            row["component_reproduction"][task] = {
                "ingf": new_ingf, "aps": new_aps,
                "aps_difference_from_original": new_aps - original["aps"],
                "aps_matches_original_at_three_decimals": f"{new_aps:.3f}" == f"{original['aps']:.3f}",
            }
        fr_mean = embed[:row["n_fr"]].astype(np.float64).mean(axis=0)
        nfr_mean = embed[row["n_fr"]:].astype(np.float64).mean(axis=0)
        row["aps"]["exact_cross_fr_nfr"] = float(fr_mean @ nfr_mean)
        print(f"{config}: INGF={row['ingf']['value']:.12f}, APS={row['aps']['value']:.12f}, exact APS={row['aps']['exact_check']:.12f}", flush=True)

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "experiment_status": subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True),
        "recalculation_script_sha256": file_sha(Path(__file__).resolve()),
        "model": {"name": "sentence-transformers/all-mpnet-base-v2", "revision": args.model.name, "files_sha256": model_files},
        "environment": {"platform": platform.platform(), "device": args.device, "batch_size": args.batch_size,
                        "packages": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "sentence-transformers", "numpy", "pandas")}},
        "embedding_cache_fingerprint": fingerprint,
        "embedding_cache_sha256": file_sha(vector_path),
        "source_files_sha256": provenance,
        "checks": {"exact_aps_identity_matches_direct_64_texts": True},
        "scope": "T2 run1 C1-C4; Functional outputs plus all T1 NFR outputs of the same condition; original generation hypotheses; training multiplicities retained",
        "alignment_provenance_limit": "Original aggregate JSON files have no embedded dataset hash or checkpoint revision; reuse relies on the unchanged repository, common evaluator, complete matching counts and exact raw-to-training correspondence. No new NLI inference is claimed.",
        "conditions": checks,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
