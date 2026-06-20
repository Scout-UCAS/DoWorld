import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List


DEFAULT_INTERVENTIONS = {
    "object_rearrangement": {
        "description": "Evaluate on environment variants with changed object layouts.",
        "overrides": ["env.name={env_name}"],
    },
    "dynamics_shift": {
        "description": "Evaluate on environment variants with altered mass/friction/contact dynamics.",
        "overrides": ["env.name={env_name}"],
    },
    "visual_ood": {
        "description": "Evaluate on held-out visual/object instances.",
        "overrides": ["env.name={env_name}"],
    },
}


def load_interventions(path: str | None) -> Dict[str, Dict[str, object]]:
    if path is None:
        return DEFAULT_INTERVENTIONS
    with open(path) as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Do-World OOD/intervention evaluation commands.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--interventions-json", default=None)
    parser.add_argument("--output-root", default="experiments/do_world_ood")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    interventions = load_interventions(args.interventions_json)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "ood_commands.jsonl"

    with manifest_path.open("w") as manifest:
        for name, spec in interventions.items():
            overrides: List[str] = [
                override.format(env_name=args.env_name)
                for override in spec.get("overrides", [])
            ]
            output_dir = output_root / name
            command = [
                "python",
                "sold/evaluate_do_world.py",
                f"checkpoint_path={args.checkpoint_path}",
                f"hydra.run.dir={output_dir}",
                *overrides,
            ]
            record = {
                "name": name,
                "description": spec.get("description", ""),
                "command": command,
                "output_dir": str(output_dir),
            }
            manifest.write(json.dumps(record) + "\n")
            print(f"[{name}] {' '.join(command)}")
            if args.execute:
                env = os.environ.copy()
                env["PYTHONPATH"] = "sold" + os.pathsep + env.get("PYTHONPATH", "")
                output_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True, env=env)

    if not args.execute:
        print(f"Wrote OOD evaluation manifest to {manifest_path}. Add --execute to run commands.")


if __name__ == "__main__":
    main()
