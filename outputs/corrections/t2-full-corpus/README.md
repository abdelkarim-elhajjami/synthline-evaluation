# T2 quality-metric correction

The original evaluators loaded only the generated Functional pool for T2.
Classifier training also includes all T1 NFR rows from the same configuration
and run. The corrected evaluators use this complete corpus and retain duplicates,
row order and original generation attributes.

This record covers C1–C4, run 1, at the original experiment commit
`5132a3c0461378e8cbd94c963a79572cb69eb8e7`.
`results.json` and `recompute.py` are unchanged copies of the independently
verified recalculation used to correct the thesis and Applied Intelligence paper.
The JSON records source hashes, the local embedding-model revision, software
versions, corpus identity checks and component-score reproduction.
The old Functional-only T2 summaries, including their counts and dispersion,
are retained in `conditions.<config>.original_components.t2`.
Their exact original files remain accessible at the baseline Git commit.

| Config | Functional | NFR | Total | Alignment | INGF | APS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 11,880 | 11,880 | 23,760 | 0.669975565 | 0.000116362911 | 0.302902220 |
| C2 | 11,880 | 11,880 | 23,760 | 0.622600692 | 0.000117983556 | 0.323500464 |
| C3 | 10,519 | 9,111 | 19,630 | 0.737427500 | 0.000137387742 | 0.324114762 |
| C4 | 11,274 | 10,232 | 21,506 | 0.684980479 | 0.000131043679 | 0.340269662 |

Alignment is reconstructed from recorded NLI summaries, not a new inference run.
For component counts `n_i`, means `m_i` and sample standard deviations `s_i`,
the full mean is `m = sum(n_i * m_i) / N` and the pooled sample variance is
`sum((n_i - 1) * s_i**2 + n_i * (m_i - m)**2) / (N - 1)`.
The full minimum and maximum are the component extrema.
All summaries use the declared common external NLI evaluator and each text's
original generation hypothesis. Historical files do not record checkpoint
revisions or embedded dataset hashes; this provenance limit is retained in
`results.json` and cannot be resolved retrospectively.

INGF and APS are recomputed on the full corpus, including cross-class pairs.
APS uses the original five million index-pair draws with replacement, seed 42,
and excludes self-pairs. The exact all-pairs check remains a validation value,
not a substitute for the reported sampled estimate.

The eight current metric JSON files carry their corpus scope and link to this
record. Only the three quality columns of the eight corresponding summary CSV
rows changed. Training CSVs, classifier predictions, F1 scores, per-seed results,
bootstrap intervals and figures are unaffected.

## Reproduce the historical recalculation

The archived script intentionally imports the original loaders and reads the
original component summaries. Run it against a separate checkout of the baseline
commit, not against the corrected working tree. Otherwise, T2 would no longer
refer to the original Functional-only component.

From the current repository, after activating its installed environment:

```bash
BASELINE_DIR=/tmp/synthline-t2-original-5132a3c
git worktree add --detach "$BASELINE_DIR" 5132a3c0461378e8cbd94c963a79572cb69eb8e7
# Set MODEL_SNAPSHOT to a local all-mpnet-base-v2 snapshot. The recorded revision is
# e8c3b32edf5434bc2275fc9bab85f82640a19130.
PYTHONDONTWRITEBYTECODE=1 python outputs/corrections/t2-full-corpus/recompute.py \
  --experiments "$BASELINE_DIR" --model "$MODEL_SNAPSHOT" \
  --cache /tmp/synthline-t2-embeddings --output /tmp/synthline-t2-recomputed.json \
  --device cpu
```

Use `--device mps` on a compatible Mac to match the recorded inference device.
Small floating-point differences can occur across devices and library versions.
The script loads the supplied local model in offline mode and does not modify
the experiment inputs. It stores a resumable embedding cache outside the repo.

For new experiments, use the normal corrected `evaluate_alignment.py` and
`evaluate_diversity.py` commands from the main README. They evaluate the full
corpus directly and do not use these historical summaries.
