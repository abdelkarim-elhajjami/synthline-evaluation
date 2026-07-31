"""Generate forest plots of pairwise effect sizes for the main ablation.

Reads per-seed F1 weighted scores from results_per_seed.csv, computes paired
mean differences and 95% percentile bootstrap CIs (10,000 resamples) for four
contrasts, and renders each contrast as a separate PNG.

Usage:
    python scripts/make_forest_plot.py
    python scripts/make_forest_plot.py --output-dir results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _shared import ROOT

# Serif typography to match Springer/Times-based paper body text
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

TASKS = ["t1", "t2", "t3", "t4"]
CLASSIFIERS = ["modernbert", "bert"]
CONFIGS = ["c1", "c2", "c3", "c4"]

# contrast_key, config_a, config_b, shared-x-range group
CONTRASTS = [
    ("PACE effect  (C1$\\rightarrow$C2)", "c1", "c2", "pace", "ablation"),
    ("AV effect on baseline  (C1$\\rightarrow$C3)", "c1", "c3", "av_baseline", "ablation"),
    ("AV effect on PACE branch  (C2$\\rightarrow$C4)", "c2", "c4", "av_pace", "ablation"),
    ("Best synthetic vs.\\ real  (B1$\\rightarrow$best $C_i$)", "b1", "best", "best_vs_real", "best"),
]

TASK_LABEL = {"t1": "T1", "t2": "T2", "t3": "T3", "t4": "T4"}

CLASSIFIER_COLOR = {
    "modernbert": "#0072B2",   # Okabe-Ito blue (CVD-safe, B&W-safe)
    "bert": "#D55E00",          # Okabe-Ito vermilion (CVD-safe, B&W-safe)
}
CLASSIFIER_MARKER = {"modernbert": "o", "bert": "s"}
CLASSIFIER_LABEL = {"modernbert": "ModernBERT", "bert": "BERT-base"}


def load_per_seed_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    main_configs = {"b1"} | set(CONFIGS)
    return df[df["config"].isin(main_configs)].copy()


def bootstrap_ci(deltas: np.ndarray, n_resamples: int = 10_000, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = deltas[idx].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def paired_deltas(df, task, classifier, config_a, config_b):
    cell = df[
        (df["task"] == task)
        & (df["classifier"] == classifier)
        & (df["config"].isin([config_a, config_b]))
    ]
    paired = cell.pivot(index="seed", columns="config", values="f1_weighted").dropna()
    if len(paired) < 5 or config_a not in paired.columns or config_b not in paired.columns:
        return None
    return (paired[config_b].values - paired[config_a].values).astype(float)


def best_c_for(df, task, classifier):
    cell = df[
        (df["task"] == task)
        & (df["classifier"] == classifier)
        & (df["config"].isin(CONFIGS))
    ]
    return cell.groupby("config")["f1_weighted"].mean().idxmax()


def compute_effects(df):
    rows = []
    for contrast_label, config_a, config_b_raw, contrast_key, _group in CONTRASTS:
        for task in TASKS:
            for classifier in CLASSIFIERS:
                config_b = best_c_for(df, task, classifier) if config_b_raw == "best" else config_b_raw
                deltas = paired_deltas(df, task, classifier, config_a, config_b)
                if deltas is None:
                    continue
                deltas_pp = deltas * 100.0
                delta_mean = float(np.mean(deltas_pp))
                ci_low, ci_high = bootstrap_ci(deltas_pp)
                rows.append({
                    "contrast_key": contrast_key,
                    "contrast_label": contrast_label,
                    "task": task,
                    "classifier": classifier,
                    "config_a": config_a,
                    "config_b": config_b,
                    "delta_mean": delta_mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })
    return pd.DataFrame(rows)


def _shared_xrange(effects, group_keys, pad_frac=0.08):
    sub = effects[effects["contrast_key"].isin(group_keys)]
    xlo = float(sub["ci_low"].min())
    xhi = float(sub["ci_high"].max())
    span = xhi - xlo
    pad = max(span * pad_frac, 0.4)
    return xlo - pad, xhi + pad


def plot_single_contrast(effects, contrast_key, output_path, xlim=None, dpi=300):
    sub = effects[effects["contrast_key"] == contrast_key].copy()

    rows = []
    for task in TASKS:
        for classifier in CLASSIFIERS:
            r = sub[(sub["task"] == task) & (sub["classifier"] == classifier)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append({
                "task": task,
                "classifier": classifier,
                "delta": r["delta_mean"],
                "low": r["ci_low"],
                "high": r["ci_high"],
            })

    n_rows = len(rows)
    # y-positions with a gap between task groups: within-task rows are 1 apart,
    # successive task groups are separated by an additional GROUP_GAP.
    GROUP_GAP = 0.5
    ROW_SPACING = 0.85
    y_positions = []
    y_cursor = 0.0  # topmost row (high y, drawn at top)
    for i in range(n_rows):
        if i > 0:
            y_cursor -= ROW_SPACING + (GROUP_GAP if i % 2 == 0 else 0.0)
        y_positions.append(y_cursor)
    y_positions = np.array(y_positions)

    # Figure height scales with number of rows + gaps for breathable spacing
    fig_h = 0.38 * (abs(y_positions[-1] - y_positions[0]) + 1.1) + 0.8
    fig, ax = plt.subplots(figsize=(6.6, fig_h))

    # Zero reference: solid hairline
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.7, zorder=1)

    # Task-zone boundaries: midpoints of the inter-task gaps so that shaded
    # (T2, T4) and unshaded (T1, T3) zones all have identical height.
    half_gap = ROW_SPACING / 2.0 + GROUP_GAP / 2.0
    zone_top = y_positions[0] + half_gap
    zone_bot = y_positions[-1] - half_gap
    boundaries = [zone_top]
    for t_idx in range(len(TASKS) - 1):
        last_row_of_group = 2 * t_idx + 1
        first_row_next = last_row_of_group + 1
        boundaries.append((y_positions[last_row_of_group] + y_positions[first_row_next]) / 2.0)
    boundaries.append(zone_bot)

    # Shade T2 (idx 1) and T4 (idx 3) zones
    for task_idx in (1, 3):
        y_top = boundaries[task_idx]
        y_bot = boundaries[task_idx + 1]
        ax.axhspan(y_bot, y_top, color="#f5f5f5", zorder=0)

    # Resolve x-range now (needed to decide whether CIs fit under the marker)
    if xlim is None:
        xlo_auto = min(r["low"] for r in rows)
        xhi_auto = max(r["high"] for r in rows)
        span_auto = xhi_auto - xlo_auto
        pad = max(span_auto * 0.08, 0.4)
        xlim = (xlo_auto - pad, xhi_auto + pad)

    # CI line + endpoint caps + marker (fine, elegant strokes).
    # When the CI is narrower than the marker's visual footprint, the bar/caps
    # would be hidden under the marker and only yield antenna-like stubs.
    # Skip them in that case and let the marker alone convey the estimate.
    cap_half = 0.08  # vertical half-height of endpoint caps in y units
    xspan = xlim[1] - xlim[0]
    ci_visibility_threshold = 0.012 * xspan  # ~marker diameter in data units
    for y, row in zip(y_positions, rows):
        clf = row["classifier"]
        color = CLASSIFIER_COLOR[clf]
        marker = CLASSIFIER_MARKER[clf]
        ci_width = row["high"] - row["low"]
        if ci_width >= ci_visibility_threshold:
            ax.plot([row["low"], row["high"]], [y, y],
                    color=color, linewidth=0.9, solid_capstyle="butt",
                    alpha=0.9, zorder=2)
            for x_end in (row["low"], row["high"]):
                ax.plot([x_end, x_end], [y - cap_half, y + cap_half],
                        color=color, linewidth=0.9, alpha=0.9, zorder=2)
        ax.plot(row["delta"], y,
                marker=marker, color=color,
                markersize=5.0,
                markerfacecolor=color,
                markeredgecolor=color,
                zorder=3, linestyle="none")

    # Hierarchical y-labels: one task label centered between its two rows
    task_ticks = []
    task_labels = []
    for t_idx, task in enumerate(TASKS):
        top_row = 2 * t_idx
        bot_row = top_row + 1
        y_center = (y_positions[top_row] + y_positions[bot_row]) / 2.0
        task_ticks.append(y_center)
        task_labels.append(TASK_LABEL[task])

    ax.set_yticks(task_ticks)
    ax.set_yticklabels(task_labels, fontsize=10.5)
    ax.set_ylim(zone_bot, zone_top)

    # X-axis (xlim resolved above)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"$\overline{\Delta}$ in F1$_{\mathrm{weighted}}$ (percentage points)",
                  fontsize=10, labelpad=8)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0, pad=4)

    # Spines: fully minimal — drop all four
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Legend (shape-only, small, top-right outside data)
    handles = [
        Line2D([0], [0], marker="o", color=CLASSIFIER_COLOR["modernbert"],
               markerfacecolor=CLASSIFIER_COLOR["modernbert"],
               markeredgecolor=CLASSIFIER_COLOR["modernbert"],
               linestyle="none", markersize=6.5, label="ModernBERT"),
        Line2D([0], [0], marker="s", color=CLASSIFIER_COLOR["bert"],
               markerfacecolor=CLASSIFIER_COLOR["bert"],
               markeredgecolor=CLASSIFIER_COLOR["bert"],
               linestyle="none", markersize=6.5, label="BERT-base"),
    ]
    leg = ax.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.30), ncol=2,
                    fontsize=9, frameon=False,
                    handletextpad=0.6, columnspacing=2.6)
    leg.set_zorder(5)

    ax.grid(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "results_per_seed.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--dump-csv", type=Path, default=None)
    args = parser.parse_args()

    df = load_per_seed_results(args.input)
    effects = compute_effects(df)

    if args.dump_csv is not None:
        args.dump_csv.parent.mkdir(parents=True, exist_ok=True)
        effects.to_csv(args.dump_csv, index=False)
        print(f"Wrote effects CSV to {args.dump_csv}")

    # Shared x-range across the three ablation contrasts for direct comparability
    ablation_keys = [k for *_, k, g in CONTRASTS if g == "ablation"]
    ablation_xlim = _shared_xrange(effects, ablation_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _label, _ca, _cb, contrast_key, group in CONTRASTS:
        out_path = args.output_dir / f"forest_{contrast_key}.png"
        xlim = ablation_xlim if group == "ablation" else None
        plot_single_contrast(effects, contrast_key, out_path, xlim=xlim, dpi=args.dpi)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
