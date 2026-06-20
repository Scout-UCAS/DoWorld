import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


BENCHMARK_CONFIGS = {
    "causalworld": "train_do_world_causalworld",
    "maniskill2": "train_do_world_maniskill2",
    "procthor": "train_do_world_procthor",
}

BENCHMARK_ENV_OVERRIDES = {
    "causalworld": [
        "model.env.suite=causalworld",
        "model.env.name=pushing",
        "model.env.image_size=[128,128]",
        "model.env.max_episode_steps=100",
        "model.env.action_repeat=1",
    ],
    "maniskill2": [
        "model.env.suite=maniskill2",
        "model.env.name=PickCube-v0",
        "model.env.image_size=[128,128]",
        "model.env.max_episode_steps=100",
        "model.env.action_repeat=1",
    ],
    "procthor": [
        "model.env.suite=procthor",
        "model.env.name=FloorPlan1",
        "model.env.image_size=[128,128]",
        "model.env.max_episode_steps=200",
        "model.env.action_repeat=1",
    ],
}

ABLATION_CONFIGS = {
    "no_mechanism_library": "ablations/no_mechanism_library",
    "no_counterfactual_mpc": "ablations/no_counterfactual_mpc",
    "no_intervention_loss": "ablations/no_intervention_loss",
    "no_language_alignment": "ablations/no_language_alignment",
}


@dataclass
class ExperimentCommand:
    name: str
    command: List[str]
    output_dir: Path

    def to_record(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "command": self.command,
            "output_dir": str(self.output_dir),
        }


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _do_world_command(config_name: str, seed: int, output_dir: Path) -> List[str]:
    return [
        "python",
        "sold/train_sold.py",
        "--config-name",
        config_name,
        f"seed={seed}",
        f"hydra.run.dir={output_dir}",
    ]


def _load_baseline_manifest(name: str) -> Dict[str, object]:
    path = Path("configs") / "baselines" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown baseline `{name}`. Expected {path}.")
    with path.open() as file:
        return yaml.safe_load(file)


def _baseline_command(name: str, task: str, seed: int, output_dir: Path) -> List[str]:
    manifest = _load_baseline_manifest(name)
    return [
        str(part).format(task=task, seed=seed, output_dir=output_dir)
        for part in manifest["command_template"]
    ]


def build_commands(
    benchmarks: Iterable[str],
    seeds: Iterable[int],
    include_ablations: bool,
    baselines: Iterable[str],
    output_root: Path,
) -> List[ExperimentCommand]:
    commands: List[ExperimentCommand] = []
    for benchmark in benchmarks:
        if benchmark not in BENCHMARK_CONFIGS:
            raise ValueError(f"Unknown benchmark `{benchmark}`. Valid: {sorted(BENCHMARK_CONFIGS)}")

        for seed in seeds:
            output_dir = output_root / benchmark / "do_world" / f"seed_{seed}"
            commands.append(ExperimentCommand(
                name=f"{benchmark}/do_world/seed_{seed}",
                command=_do_world_command(BENCHMARK_CONFIGS[benchmark], seed, output_dir),
                output_dir=output_dir,
            ))

            if include_ablations:
                for ablation, config_name in ABLATION_CONFIGS.items():
                    ablation_output = output_root / benchmark / ablation / f"seed_{seed}"
                    commands.append(ExperimentCommand(
                        name=f"{benchmark}/{ablation}/seed_{seed}",
                        command=_do_world_command(config_name, seed, ablation_output)
                        + BENCHMARK_ENV_OVERRIDES[benchmark],
                        output_dir=ablation_output,
                    ))

            for baseline in baselines:
                baseline_output = output_root / benchmark / baseline / f"seed_{seed}"
                commands.append(ExperimentCommand(
                    name=f"{benchmark}/{baseline}/seed_{seed}",
                    command=_baseline_command(baseline, benchmark, seed, baseline_output),
                    output_dir=baseline_output,
                ))

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or run Do-World benchmark commands.")
    parser.add_argument("--benchmarks", default="causalworld,maniskill2,procthor")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--baselines", default="")
    parser.add_argument("--output-root", default="experiments/do_world_benchmarks")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", default="experiments/do_world_benchmarks/commands.jsonl")
    args = parser.parse_args()

    benchmarks = _split_csv(args.benchmarks)
    seeds = [int(seed) for seed in _split_csv(args.seeds)]
    baselines = _split_csv(args.baselines)
    output_root = Path(args.output_root)
    commands = build_commands(benchmarks, seeds, args.include_ablations, baselines, output_root)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as file:
        for command in commands:
            file.write(json.dumps(command.to_record()) + "\n")

    for command in commands:
        printable = " ".join(command.command)
        print(f"[{command.name}] {printable}")
        if args.execute:
            command.output_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = "sold" + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run(command.command, check=True, env=env)

    if not args.execute:
        print(f"Wrote command manifest to {manifest_path}. Add --execute to run commands.")


if __name__ == "__main__":
    main()
