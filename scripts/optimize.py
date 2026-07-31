"""Build and optimize prompts for a task and ablation configuration.

Usage:
    python scripts/optimize.py --task t1 --config c2 --run 1

Only needed for PACE-enabled configs (C2, C4). Saves prompts.json per label.
Auto-resumes: detects missing configs and optimizes only those.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from synthline import PromptSet, Synthline

from _shared import (
    DOMAIN_DIR,
    ROOT,
    config_key,
    label_slug,
    load_experiment_config,
    load_task_config,
    progress_bar,
)

load_dotenv()

OUTPUTS_DIR = ROOT / "outputs" / "synthetic"


async def run_optimization(task: str, config: str, run: int, generator: str | None = None) -> None:
    task_cfg = load_task_config(task)
    experiment_cfg = load_experiment_config()

    gen_cfg = experiment_cfg["generation"]
    pace_cfg = experiment_cfg["ablation"]["pace"]
    llm = generator or gen_cfg["llm"]
    gen_suffix = f"_{llm.split('/')[-1]}" if generator else ""
    out_dir = OUTPUTS_DIR / task / f"{config}{gen_suffix}" / f"run{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Optimize: {task_cfg['task']} | Config: {config.upper()} | Run: {run}")
    print(f"LLM: {llm}")
    print(f"PACE: alpha={pace_cfg['alpha']}, iterations={pace_cfg['iterations']}, "
          f"actors={pace_cfg['actors']}, candidates={pace_cfg['candidates']}")
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
        label_dir = out_dir / label_slug(label["name"])
        prompts_path = label_dir / "prompts.json"

        full_prompts = sl.build_prompts(
            label=label["name"],
            label_definition=label["definition"],
            samples_per_prompt=gen_cfg["samples_per_prompt"],
            features=task_cfg["features"],
        )

        # Detect missing configs
        if prompts_path.exists():
            existing = PromptSet.load(str(prompts_path))
            existing_keys = {config_key(e.config) for e in existing}
            missing_entries = [e for e in full_prompts if config_key(e.config) not in existing_keys]

            if not missing_entries:
                print(f"  Label: {label['name']} — complete ({len(existing)} prompts), skipping")
                continue

            print(f"  Label: {label['name']} — resuming ({len(existing)}/{len(full_prompts)}, {len(missing_entries)} remaining)")
            to_optimize = PromptSet(
                entries=missing_entries,
                label=full_prompts.label,
                label_definition=full_prompts.label_definition,
                samples_per_prompt=full_prompts.samples_per_prompt,
                base_features=full_prompts.base_features,
            )
        else:
            print(f"  Label: {label['name']}")
            print(f"    Built {len(full_prompts)} atomic prompts")
            existing = None
            to_optimize = full_prompts

        print(f"    Running PACE on {len(to_optimize)} configs (alpha={pace_cfg['alpha']}, iterations={pace_cfg['iterations']})...")
        optimized = await sl.optimize(
            to_optimize,
            alpha=pace_cfg["alpha"],
            iterations=pace_cfg["iterations"],
            actors=pace_cfg["actors"],
            candidates=pace_cfg["candidates"],
            on_progress=progress_bar("PACE"),
        )
        print()

        # Merge with existing if resuming
        if existing is not None:
            merged = PromptSet(
                entries=existing.entries + list(optimized.entries),
                label=full_prompts.label,
                label_definition=full_prompts.label_definition,
                samples_per_prompt=full_prompts.samples_per_prompt,
                optimized=True,
                base_features=full_prompts.base_features,
            )
        else:
            merged = optimized

        label_dir.mkdir(exist_ok=True)
        merged.save(str(prompts_path))
        print(f"    Saved {len(merged)} prompts to {prompts_path}\n")

    print(f"Optimization complete for {task}/{config}/run{run}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and optimize prompts")
    parser.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    parser.add_argument("--config", required=True, choices=["c2"],
                        help="C4 reuses C2's optimized prompts — only C2 needs optimization.")
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--generator", type=str, default=None,
                        help="Override generator LLM")
    args = parser.parse_args()
    asyncio.run(run_optimization(args.task, args.config, args.run, args.generator))


if __name__ == "__main__":
    main()
