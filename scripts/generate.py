"""Generate synthetic data for a task and ablation configuration.

Usage:
    python scripts/generate.py --task t1 --config c1 --run 1
    python scripts/generate.py --task t1 --config c2 --run 1
    python scripts/generate.py --task t1 --config c3 --run 1
    python scripts/generate.py --task t1 --config c4 --run 1

C1/C2: standard generation (C2 auto-loads PACE-optimized prompts from optimize.py).
C3/C4: shared-base verification — loads the base dataset (C1/C2) and runs verify().
Auto-resumes: detects completed configs via checkpoint.json and generates only missing ones.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from synthline import Dataset, PromptSet, Synthline

from _shared import (
    DOMAIN_DIR,
    ROOT,
    config_key,
    label_slug,
    load_experiment_config,
    load_task_config,
    progress_bar,
    verification_bar,
)

load_dotenv()

OUTPUTS_DIR = ROOT / "outputs" / "synthetic"

ABLATION = {
    "c1": {"pace": False, "av": False},
    "c2": {"pace": True, "av": False},
    "c3": {"pace": False, "av": True},
    "c4": {"pace": True, "av": True},
}

# C4 reuses C2's PACE-optimized prompts (non-deterministic, can't rebuild)
PROMPTS_SOURCE = {"c4": "c2"}

# Shared-base: C3 verifies C1's output, C4 verifies C2's output
VERIFY_SOURCE = {"c3": "c1", "c4": "c2"}


def _append_to_csv(path: Path, dataset) -> None:
    existing = pd.read_csv(path)
    new = pd.DataFrame(dataset.samples)
    merged = pd.concat([existing, new], ignore_index=True)
    merged.to_csv(path, index=False)


def _load_prompts(
    label_dir: Path,
    source_dir: Path | None,
    sl: Synthline,
    label_name: str,
    label_def: str,
    samples_per_prompt: int,
    features: dict,
) -> PromptSet:
    """Load prompts from source, own dir, or build on the fly."""
    # 1. Load from source config (e.g. C4 → C2)
    if source_dir:
        source_path = source_dir / "prompts.json"
        if source_path.exists():
            prompts = PromptSet.load(str(source_path))
            print(f"    Loaded {len(prompts)} prompts from {source_dir.parent.parent.name} (optimized={prompts.optimized})")
            return prompts

    # 2. Load from own label dir (from prior optimize.py run)
    own_path = label_dir / "prompts.json"
    if own_path.exists():
        prompts = PromptSet.load(str(own_path))
        print(f"    Loaded {len(prompts)} prompts (optimized={prompts.optimized})")
        return prompts

    # 3. Build on the fly (C1, C3 — deterministic)
    prompts = sl.build_prompts(
        label=label_name,
        label_definition=label_def,
        samples_per_prompt=samples_per_prompt,
        features=features,
    )
    label_dir.mkdir(exist_ok=True)
    prompts.save(str(label_dir / "prompts.json"))
    print(f"    Built {len(prompts)} atomic prompts")
    return prompts


async def run_verification(
    task: str, config: str, run: int,
    generator: str | None,
    task_cfg: dict, experiment_cfg: dict,
) -> None:
    """Shared-base workflow: load base dataset (C1/C2) → averify() → save."""
    gen_cfg = experiment_cfg["generation"]
    base_config = VERIFY_SOURCE[config]
    ablation_cfg = ABLATION[config]
    llm = generator or gen_cfg["llm"]
    gen_suffix = f"_{llm.split('/')[-1]}" if generator else ""

    base_dir = OUTPUTS_DIR / task / f"{base_config}{gen_suffix}" / f"run{run}"
    out_dir = OUTPUTS_DIR / task / f"{config}{gen_suffix}" / f"run{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    av_cfg = experiment_cfg["ablation"]["av"]
    threshold = av_cfg["threshold"]

    print(f"\n{'='*60}")
    print(f"Task: {task_cfg['task']} | Config: {config.upper()} | Run: {run}")
    print(f"Shared-base: verifying {base_config.upper()}'s output")
    print(f"Generator: {llm} | AV threshold: {threshold}")
    print(f"Base: {base_dir}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    sl = Synthline(
        fm=str(DOMAIN_DIR / "fm.xml"),
        llm=llm,
        glossary=str(DOMAIN_DIR / "glossary.yaml"),
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
    )

    for label in task_cfg["labels"]:
        slug = label_slug(label["name"])
        base_label_dir = base_dir / slug
        out_label_dir = out_dir / slug

        if (out_label_dir / "data.csv").exists():
            print(f"  Label: {label['name']} — already verified, skipping")
            continue

        if not (base_label_dir / "data.csv").exists():
            print(f"  Label: {label['name']} — base data not found at {base_label_dir}, skipping")
            continue

        base_dataset = Dataset.load(str(base_label_dir))
        print(f"  Label: {label['name']} — loaded {len(base_dataset)} samples from {base_config.upper()}")

        print(f"    Verifying (threshold={threshold})...")
        dataset = await sl.verify(
            base_dataset,
            threshold=threshold,
            samples_per_prompt=gen_cfg["samples_per_prompt"],
            on_verification=verification_bar(),
        )
        print()
        print(f"    Verified: {len(dataset)} samples")

        av_info = dataset.metadata.get("alignment_verification", {})
        if isinstance(av_info, dict):
            reason = av_info.get("termination_reason", "?")
            deficit = av_info.get("alignment_deficit", 0)
            attempts = av_info.get("attempts_used", 0)
            print(f"    Termination: {reason} | Deficit: {deficit} | Attempts: {attempts}")

        out_label_dir.mkdir(parents=True, exist_ok=True)
        dataset.save(str(out_label_dir))
        print(f"    Saved to {out_label_dir}\n")

    # Guard: if no labels were verified at all, something went wrong upstream
    verified_labels = [
        label_slug(label["name"]) for label in task_cfg["labels"]
        if (out_dir / label_slug(label["name"]) / "data.csv").exists()
    ]
    if not verified_labels:
        raise RuntimeError(
            f"Verification for {task}/{config}/run{run} produced no data. "
            f"Ensure the base config ({VERIFY_SOURCE[config]}) has been generated first."
        )

    # Write label map and run metadata
    label_map = {}
    for label in task_cfg["labels"]:
        label_map[label_slug(label["name"])] = label["name"]
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    metadata = {
        "task": task,
        "config": config,
        "run": run,
        "generator": llm,
        "ablation": ablation_cfg,
        "verify_source": base_config,
        "av_params": experiment_cfg["ablation"]["av"],
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Verification complete for {task}/{config}/run{run}")


async def run_generation(
    task: str, config: str, run: int,
    generator: str | None, prompts_from: str | None,
) -> None:
    task_cfg = load_task_config(task)
    experiment_cfg = load_experiment_config()

    # Shared-base: C3/C4 verify base datasets instead of generating from scratch
    if config in VERIFY_SOURCE:
        await run_verification(task, config, run, generator, task_cfg, experiment_cfg)
        return

    gen_cfg = experiment_cfg["generation"]
    ablation_cfg = ABLATION[config]
    llm = generator or gen_cfg["llm"]
    gen_suffix = f"_{llm.split('/')[-1]}" if generator else ""

    out_dir = OUTPUTS_DIR / task / f"{config}{gen_suffix}" / f"run{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve prompt source
    source_config = prompts_from or PROMPTS_SOURCE.get(config)
    source_base = OUTPUTS_DIR / task / f"{source_config}{gen_suffix}" / f"run{run}" if source_config else None

    print(f"\n{'='*60}")
    print(f"Task: {task_cfg['task']} | Config: {config.upper()} | Run: {run}")
    print(f"Generator: {llm}")
    print(f"PACE: {'On' if ablation_cfg['pace'] else 'Off'} | AV: Off")
    if source_config:
        print(f"Prompts from: {source_config.upper()}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    sl = Synthline(
        fm=str(DOMAIN_DIR / "fm.xml"),
        llm=llm,
        glossary=str(DOMAIN_DIR / "glossary.yaml"),
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
    )

    target = task_cfg["target_samples_per_label"]
    samples_per_prompt = gen_cfg["samples_per_prompt"]

    for label in task_cfg["labels"]:
        label_dir = out_dir / label_slug(label["name"])
        csv_path = label_dir / "data.csv"
        checkpoint_path = label_dir / "checkpoint.json"

        source_dir = source_base / label_slug(label["name"]) if source_base else None

        full_prompts = _load_prompts(
            label_dir=label_dir,
            source_dir=source_dir,
            sl=sl,
            label_name=label["name"],
            label_def=label["definition"],
            samples_per_prompt=samples_per_prompt,
            features=task_cfg["features"],
        )

        if ablation_cfg["pace"] and not full_prompts.optimized:
            raise RuntimeError(
                f"Config {config} requires PACE-optimized prompts but none were found "
                f"for label '{label['name']}'. Run optimize.py first."
            )

        # Detect missing configs via checkpoint
        if checkpoint_path.exists():
            completed_keys = set(json.loads(checkpoint_path.read_text()))
            missing_entries = [e for e in full_prompts if config_key(e.config) not in completed_keys]

            if not missing_entries:
                print(f"  Label: {label['name']} — complete ({len(full_prompts)} configs), skipping")
                continue

            print(f"  Label: {label['name']} — resuming ({len(full_prompts) - len(missing_entries)}/{len(full_prompts)} configs done, {len(missing_entries)} remaining)")
            to_generate = PromptSet(
                entries=missing_entries,
                label=full_prompts.label,
                label_definition=full_prompts.label_definition,
                samples_per_prompt=full_prompts.samples_per_prompt,
                optimized=full_prompts.optimized,
                base_features=full_prompts.base_features,
            )
            remaining = len(missing_entries) * samples_per_prompt
        else:
            print(f"  Label: {label['name']}")
            completed_keys = set()
            to_generate = full_prompts
            remaining = target

        print(f"    Generating {remaining} samples...")
        dataset = await sl.generate(
            to_generate,
            samples=remaining,
            verify=False,
            on_progress=progress_bar(),
        )
        print()
        if len(dataset) < remaining:
            print(f"    Generated {len(dataset)}/{remaining} samples (some configs failed)")
        else:
            print(f"    Generated {len(dataset)} samples")

        # Save: append to existing CSV or create new
        label_dir.mkdir(exist_ok=True)
        if csv_path.exists():
            _append_to_csv(csv_path, dataset)
        else:
            dataset.save(str(label_dir))

        # Update checkpoint
        new_keys = {config_key(e.config) for e in to_generate}
        checkpoint_path.write_text(json.dumps(list(completed_keys | new_keys)))
        print(f"    Saved to {label_dir}\n")

    label_map = {}
    for label in task_cfg["labels"]:
        label_map[label_slug(label["name"])] = label["name"]
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    metadata = {
        "task": task,
        "config": config,
        "run": run,
        "generator": llm,
        "ablation": ablation_cfg,
        "prompts_from": source_config,
        "pace_params": experiment_cfg["ablation"]["pace"] if ablation_cfg["pace"] else None,
        "av_params": None,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generation complete for {task}/{config}/run{run}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    parser.add_argument("--config", required=True, choices=["c1", "c2", "c3", "c4"])
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--generator", type=str, default=None,
                        help="Override generator LLM")
    parser.add_argument("--prompts-from", type=str, default=None,
                        help="Load prompts from another config (e.g. c2)")
    args = parser.parse_args()
    asyncio.run(run_generation(args.task, args.config, args.run, args.generator, args.prompts_from))


if __name__ == "__main__":
    main()
