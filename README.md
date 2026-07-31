# Synthline Evaluation

Synthline evaluation: a 2x2 ablation of PACE and Alignment Verification (AV)
across four Requirements Engineering classification tasks.

## What This Repository Contains

- Experiment configuration and domain constraints used by the evaluation.
- Scripts to prepare public datasets, generate synthetic data, train classifiers,
  evaluate utility, and aggregate results.
- Generated synthetic datasets, prepared training CSVs, prediction files, quality
  metrics, summary CSVs, and figures.
- No trained model weights. `outputs/models/` is intentionally ignored because
  local fine-tuned checkpoints are large and can be reproduced with `train.py`.

## Ablation Design

| Config | PACE | AV | Description |
| --- | --- | --- | --- |
| C1 | Off | Off | FM-constrained generation only |
| C2 | On | Off | Adds prompt optimization |
| C3 | Off | On | Verifies C1 outputs with AV |
| C4 | On | On | Verifies C2 outputs with AV |

C3 and C4 use a shared-base design: they apply AV to the already generated C1
and C2 datasets, respectively, rather than generating from scratch.

## Tasks

| Task | Type | Dataset |
| --- | --- | --- |
| T1 | NFR 11-class classification | PROMISE_exp |
| T2 | FR/NFR binary classification | PROMISE_exp, with T1 NFR data reused |
| T3 | Security binary classification | SecReq |
| T4 | Causality binary classification | CiRA |

## Data Sources and Licenses

This repository includes prepared splits of the public datasets used for the
evaluation. Third-party source datasets retain their original licenses and terms;
the repository license applies to the code and original evaluation artifacts.

| Task | Source | Upstream location | License/terms |
| --- | --- | --- | --- |
| T1/T2 | PROMISE_exp | [AleksandarMitrevski/se-requirements-classification](https://github.com/AleksandarMitrevski/se-requirements-classification) | No license declared in the upstream GitHub repository; cite Lima et al. (2019). |
| T3 | SecReq | [Zenodo 10.5281/zenodo.4530183](https://doi.org/10.5281/zenodo.4530183) | CC-BY-4.0. |
| T4 | CiRA | [fischJan/CiRA](https://github.com/fischJan/CiRA) | No standard SPDX license exposed by GitHub metadata; cite the CiRA publications. |

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Generation uses the Synthline SDK and an LLM provider. For the default
OpenRouter-backed model, set the provider key expected by your Synthline
installation, for example:

```bash
export OPENROUTER_API_KEY="..."
```

## Pipeline

Run stages in order:

```bash
# 1. Prepare real data for the B1 baseline
python scripts/prepare_real.py

# 2. Optimize prompts for PACE-enabled configs
python scripts/optimize.py --task t1 --config c2 --run 1
# repeat for t2, t3, t4

# 3. Generate synthetic data
python scripts/generate.py --task t1 --config c1 --run 1
python scripts/generate.py --task t1 --config c2 --run 1
python scripts/generate.py --task t1 --config c3 --run 1  # requires c1
python scripts/generate.py --task t1 --config c4 --run 1  # requires c2
# repeat for t2, t3, t4

# 4. Prepare synthetic training data
python scripts/prepare_synthetic.py --task t1 --config c1 --run 1
# repeat for all tasks and configs

# 5. Train classifiers with full HuggingFace fine-tuning
python scripts/train.py --task t1 --config c1 --classifier modernbert --run 1 --seed 42
# repeat for b1/c1-c4, modernbert/bert, seeds 42-46

# 6. Evaluate classifiers
python scripts/evaluate_classifiers.py --task t1 --config c1 --classifier modernbert --run 1 --seed 42
# repeat for all trained combinations

# 7. Evaluate synthetic data quality
python scripts/evaluate_alignment.py --task t1 --config c1 --run 1
python scripts/evaluate_diversity.py --task t1 --config c1 --run 1
# repeat for all synthetic configs

# 8. Aggregate results and bootstrap confidence intervals
python scripts/aggregate_results.py

# 9. Regenerate forest plots
python scripts/make_forest_plot.py
```

## Generator Sensitivity

T1 and T3 are also re-run with Claude Haiku 4.5 as the generator:

```bash
python scripts/optimize.py --task t1 --config c2 --run 1 --generator "openrouter/anthropic/claude-haiku-4.5"
python scripts/generate.py --task t1 --config c1 --run 1 --generator "openrouter/anthropic/claude-haiku-4.5"
# repeat for c2-c4, then prepare_synthetic, train, evaluate, and aggregate as above
```

## Configuration

- `configs/experiment.yaml`: shared generation settings, full fine-tuning
  hyperparameters, seeds, and ablation parameters.
- `configs/t{1-4}_*.yaml`: task-specific datasets, labels, feature selections,
  definitions, and target sample counts.
- `domain/fm.xml`: Feature Model used to constrain valid configurations.
- `domain/glossary.yaml`: feature glossary used during prompt construction.

## Outputs

```text
outputs/
  synthetic/              # generated data per task/config/label
  training/               # training CSVs consumed by train.py
  predictions/            # per-seed predictions and metrics
  metrics/                # alignment and diversity scores
  results_summary.csv     # mean and std across seeds
  results_per_seed.csv    # per-seed weighted F1/precision/recall
  results_bootstrap.csv   # paired bootstrap 95% CIs
  models/                 # local only; ignored by git

results/figures/          # forest plots
```

## License

Repository code and original evaluation artifacts are released under the MIT
License. Third-party source datasets retain their original licenses and terms as
listed above.
