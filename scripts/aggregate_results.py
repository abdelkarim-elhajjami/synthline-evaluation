"""Aggregate all experiment results into a summary CSV.

Reads metrics.json (classifier performance), alignment.json, and diversity.json
for all experiments across seeds and classifiers. Computes mean ± std over seeds
and bootstrap 95% CIs on paired mean differences for key comparisons.

Usage:
    python scripts/aggregate_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import ROOT

PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
METRICS_DIR = ROOT / "outputs" / "metrics"

TASKS = ["t1", "t2", "t3", "t4"]
CONFIGS = ["c1", "c2", "c3", "c4"]
CLASSIFIERS = ["modernbert", "bert"]

# Sensitivity configs
SENSITIVITY_CONFIGS = {
    "t3": [f"{c}_claude-haiku-4.5" for c in CONFIGS],
    "t1": [f"{c}_claude-haiku-4.5" for c in CONFIGS],
}
ALL_SENSITIVITY_CONFIGS = {cfg for cfgs in SENSITIVITY_CONFIGS.values() for cfg in cfgs}


def _read_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _discover_seeds(task: str, config: str, classifier: str) -> list[int]:
    """Discover available seeds by globbing the predictions directory."""
    run_dir = PREDICTIONS_DIR / task / config / classifier / "run1"
    if not run_dir.exists():
        return []
    seeds = []
    for seed_dir in sorted(run_dir.glob("seed*")):
        if (seed_dir / "metrics.json").exists():
            try:
                seeds.append(int(seed_dir.name.removeprefix("seed")))
            except ValueError:
                continue
    return seeds


def collect_seed_metrics() -> pd.DataFrame:
    """Collect per-seed metrics for all (task, config, classifier, run, seed) combos."""
    rows = []

    all_experiments = []

    # B1 baselines
    for task in TASKS:
        all_experiments.append((task, "b1"))

    # Main ablation
    for task in TASKS:
        for config in CONFIGS:
            all_experiments.append((task, config))

    # Sensitivity
    for task, sens_configs in SENSITIVITY_CONFIGS.items():
        for config in sens_configs:
            all_experiments.append((task, config))

    for task, config in all_experiments:
        for classifier in CLASSIFIERS:
            seeds = _discover_seeds(task, config, classifier)
            for seed in seeds:
                metrics_path = (
                    PREDICTIONS_DIR / task / config / classifier
                    / "run1" / f"seed{seed}" / "metrics.json"
                )
                metrics = _read_json(metrics_path)
                if metrics:
                    rows.append({
                        "task": task,
                        "config": config,
                        "classifier": classifier,
                        "seed": seed,
                        "f1_weighted": metrics["f1_weighted"],
                        "precision_weighted": metrics["precision_weighted"],
                        "recall_weighted": metrics["recall_weighted"],
                    })

    return pd.DataFrame(rows)


def compute_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std over seeds for each (task, config, classifier)."""
    if seed_df.empty:
        return pd.DataFrame()

    grouped = seed_df.groupby(["task", "config", "classifier"]).agg(
        f1_mean=("f1_weighted", "mean"),
        f1_std=("f1_weighted", "std"),
        precision_mean=("precision_weighted", "mean"),
        precision_std=("precision_weighted", "std"),
        recall_mean=("recall_weighted", "mean"),
        recall_std=("recall_weighted", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()

    for col in ["f1_mean", "f1_std", "precision_mean", "precision_std", "recall_mean", "recall_std"]:
        grouped[col] = grouped[col].round(4)

    return grouped


def compute_bootstrap_ci(
    seed_df: pd.DataFrame,
    classifier: str,
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap 95% CI on the paired mean difference Δ̄.

    For each pairwise comparison, pivots seeds to paired vectors a, b,
    computes per-seed deltas δᵢ = bᵢ − aᵢ, then resamples those deltas
    with replacement to obtain a percentile CI on the mean delta.
    """
    comparisons = [
        ("c1", "c3", "AV effect (baseline)"),
        ("c2", "c4", "AV effect (PACE)"),
        ("c1", "c2", "PACE effect"),
    ]
    rng = np.random.default_rng(rng_seed)
    rows = []

    clf_df = seed_df[seed_df["classifier"] == classifier]

    def _bootstrap(deltas: np.ndarray) -> tuple[float, float]:
        n = len(deltas)
        idx = rng.integers(0, n, size=(n_resamples, n))
        boot_means = deltas[idx].mean(axis=1)
        return tuple(np.percentile(boot_means, [2.5, 97.5]))

    def _add_row(task, comparison, label, a, b):
        deltas = b - a
        ci_low, ci_high = _bootstrap(deltas)
        rows.append({
            "task": task,
            "classifier": classifier,
            "comparison": comparison,
            "label": label,
            "mean_a": round(float(np.mean(a)), 4),
            "mean_b": round(float(np.mean(b)), 4),
            "delta_mean": round(float(np.mean(deltas)), 4),
            "delta_std": round(float(np.std(deltas, ddof=1)), 4),
            "ci_low": round(float(ci_low), 4),
            "ci_high": round(float(ci_high), 4),
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        })

    for task in TASKS:
        task_df = clf_df[clf_df["task"] == task]

        for config_a, config_b, label in comparisons:
            paired = task_df[task_df["config"].isin([config_a, config_b])].pivot(
                index="seed", columns="config", values="f1_weighted"
            ).dropna()
            if len(paired) < 5:
                continue
            _add_row(
                task,
                f"{config_a} vs {config_b}",
                label,
                paired[config_a].values,
                paired[config_b].values,
            )

        # Best C vs B1, where best C is selected by mean F1 for this task/classifier.
        best_config = None
        best_mean = -1.0
        for config in CONFIGS:
            vals = task_df[task_df["config"] == config]["f1_weighted"].values
            if len(vals) > 0 and float(np.mean(vals)) > best_mean:
                best_mean = float(np.mean(vals))
                best_config = config

        if best_config is None:
            continue

        paired = task_df[task_df["config"].isin([best_config, "b1"])].pivot(
            index="seed", columns="config", values="f1_weighted"
        ).dropna()
        if len(paired) < 5:
            continue
        _add_row(
            task,
            f"b1 vs {best_config}",
            "Best synthetic vs real",
            paired["b1"].values,
            paired[best_config].values,
        )

    return pd.DataFrame(rows)


def compute_sensitivity_bootstrap(
    seed_df: pd.DataFrame,
    classifier: str,
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap 95% CI on the paired mean difference Δ̄ for sensitivity contrasts.

    For each sensitivity task, computes Best Claude C_i vs B1 (a paired contrast
    where B1 is run on the same classifier and shares no generator with the Claude
    cells, but pairing is by seed). Mirrors compute_bootstrap_ci.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []

    clf_df = seed_df[seed_df["classifier"] == classifier]

    def _bootstrap(deltas: np.ndarray) -> tuple[float, float]:
        n = len(deltas)
        idx = rng.integers(0, n, size=(n_resamples, n))
        boot_means = deltas[idx].mean(axis=1)
        return tuple(np.percentile(boot_means, [2.5, 97.5]))

    for task, sens_configs in SENSITIVITY_CONFIGS.items():
        task_df = clf_df[clf_df["task"] == task]

        # Best Claude C_i vs B1
        best_config = None
        best_mean = -1.0
        for config in sens_configs:
            vals = task_df[task_df["config"] == config]["f1_weighted"].values
            if len(vals) > 0 and float(np.mean(vals)) > best_mean:
                best_mean = float(np.mean(vals))
                best_config = config
        if best_config is None:
            continue

        paired = task_df[task_df["config"].isin([best_config, "b1"])].pivot(
            index="seed", columns="config", values="f1_weighted"
        ).dropna()
        if len(paired) < 5:
            continue

        a = paired["b1"].values
        b = paired[best_config].values
        deltas = b - a
        ci_low, ci_high = _bootstrap(deltas)
        rows.append({
            "task": task,
            "classifier": classifier,
            "comparison": f"b1 vs {best_config}",
            "label": "Best sensitivity (Claude) vs real",
            "mean_a": round(float(np.mean(a)), 4),
            "mean_b": round(float(np.mean(b)), 4),
            "delta_mean": round(float(np.mean(deltas)), 4),
            "delta_std": round(float(np.std(deltas, ddof=1)), 4),
            "ci_low": round(float(ci_low), 4),
            "ci_high": round(float(ci_high), 4),
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        })

    return pd.DataFrame(rows)


def compute_sensitivity_contrasts(
    seed_df: pd.DataFrame,
    classifier: str,
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap 95% CI on Δ̄ for the three ablation contrasts with Claude.

    Applies the procedure of compute_bootstrap_ci to the Claude cells of the
    sensitivity tasks. It uses its own generator, so the rows computed by the
    other functions are unchanged.
    """
    comparisons = [
        ("c1", "c3", "AV effect (baseline)"),
        ("c2", "c4", "AV effect (PACE)"),
        ("c1", "c2", "PACE effect"),
    ]
    suffix = "_claude-haiku-4.5"
    rng = np.random.default_rng(rng_seed)
    rows = []

    clf_df = seed_df[seed_df["classifier"] == classifier]

    for task in sorted(SENSITIVITY_CONFIGS):
        task_df = clf_df[clf_df["task"] == task]
        for base_a, base_b, label in comparisons:
            config_a, config_b = base_a + suffix, base_b + suffix
            paired = task_df[task_df["config"].isin([config_a, config_b])].pivot(
                index="seed", columns="config", values="f1_weighted"
            ).dropna()
            if len(paired) < 5:
                continue
            a = paired[config_a].values
            b = paired[config_b].values
            deltas = b - a
            idx = rng.integers(0, len(deltas), size=(n_resamples, len(deltas)))
            ci_low, ci_high = np.percentile(deltas[idx].mean(axis=1), [2.5, 97.5])
            rows.append({
                "task": task,
                "classifier": classifier,
                "comparison": f"{config_a} vs {config_b}",
                "label": f"{label} (Claude)",
                "mean_a": round(float(np.mean(a)), 4),
                "mean_b": round(float(np.mean(b)), 4),
                "delta_mean": round(float(np.mean(deltas)), 4),
                "delta_std": round(float(np.std(deltas, ddof=1)), 4),
                "ci_low": round(float(ci_low), 4),
                "ci_high": round(float(ci_high), 4),
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            })

    return pd.DataFrame(rows)


def add_data_quality(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add alignment and diversity metrics (not seed-dependent)."""
    rows = []
    for _, row in summary_df.iterrows():
        task, config = row["task"], row["config"]
        entry = row.to_dict()

        if config not in ("b1",):
            alignment = _read_json(METRICS_DIR / task / config / "run1" / "alignment.json")
            if alignment:
                entry["alignment_mean"] = round(alignment["mean"], 4)

            diversity = _read_json(METRICS_DIR / task / config / "run1" / "diversity.json")
            if diversity:
                entry["ingf"] = round(diversity["ingf"], 6)
                entry["aps"] = round(diversity["aps"], 4)

        rows.append(entry)

    return pd.DataFrame(rows)


def print_summary(summary_df: pd.DataFrame, bootstrap_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (mean ± std over seeds)")
    print("=" * 80)

    for classifier in CLASSIFIERS:
        clf_df = summary_df[summary_df["classifier"] == classifier]
        if clf_df.empty:
            continue

        print(f"\n--- {classifier.upper()} — F1 weighted (mean ± std) ---")
        main = clf_df[~clf_df["config"].isin(ALL_SENSITIVITY_CONFIGS)].copy()
        main["f1_display"] = main.apply(
            lambda r: f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}" if pd.notna(r["f1_std"]) else f"{r['f1_mean']:.4f}",
            axis=1,
        )
        pivot = main.pivot(index="config", columns="task", values="f1_display")
        pivot = pivot.reindex(["b1", "c1", "c2", "c3", "c4"])
        if all(t in pivot.columns for t in TASKS):
            pivot = pivot[TASKS]
        print(pivot.to_string())

    if not bootstrap_df.empty:
        print("\n--- Bootstrap 95% CI on paired mean difference Δ̄ ---")
        for classifier in CLASSIFIERS:
            clf_rows = bootstrap_df[bootstrap_df["classifier"] == classifier]
            if clf_rows.empty:
                continue
            print(f"\n  {classifier.upper()}:")
            for _, r in clf_rows.iterrows():
                mark = "*" if r["ci_excludes_zero"] else ""
                print(
                    f"    {r['task'].upper()} | {r['label']:32s} | "
                    f"Δ̄={r['delta_mean']:+.4f} "
                    f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]{mark}"
                )

    print()


def main() -> None:
    seed_df = collect_seed_metrics()

    if seed_df.empty:
        print("No metrics found. Run train.py and evaluate_classifiers.py first.")
        return

    summary_df = compute_summary(seed_df)
    summary_df = add_data_quality(summary_df)

    boot_dfs = []
    sens_boot_dfs = []
    for classifier in CLASSIFIERS:
        b = compute_bootstrap_ci(seed_df, classifier)
        if not b.empty:
            boot_dfs.append(b)
        sb = compute_sensitivity_bootstrap(seed_df, classifier)
        if not sb.empty:
            sens_boot_dfs.append(sb)
    for classifier in CLASSIFIERS:
        sc = compute_sensitivity_contrasts(seed_df, classifier)
        if not sc.empty:
            sens_boot_dfs.append(sc)
    bootstrap_df = pd.concat(boot_dfs, ignore_index=True) if boot_dfs else pd.DataFrame()
    if sens_boot_dfs:
        bootstrap_df = pd.concat([bootstrap_df, *sens_boot_dfs], ignore_index=True)

    out_dir = ROOT / "outputs"

    seed_df.to_csv(out_dir / "results_per_seed.csv", index=False)
    summary_df.to_csv(out_dir / "results_summary.csv", index=False)
    if not bootstrap_df.empty:
        bootstrap_df.to_csv(out_dir / "results_bootstrap.csv", index=False)

    print(f"Saved to {out_dir / 'results_summary.csv'}")
    print(f"Saved to {out_dir / 'results_per_seed.csv'}")
    if not bootstrap_df.empty:
        print(f"Saved to {out_dir / 'results_bootstrap.csv'}")

    print_summary(summary_df, bootstrap_df)


if __name__ == "__main__":
    main()
