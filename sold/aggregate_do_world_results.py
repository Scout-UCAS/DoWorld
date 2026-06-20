import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional


DEFAULT_METRICS = [
    "episode_return",
    "success_rate",
    "prediction_error",
    "multi_step_error",
    "intervention_error",
    "counterfactual_drop",
    "factual_counterfactual_gap",
    "relation_sparsity",
    "mechanism_diversity",
]


def _records(root: Path) -> Iterable[Dict[str, object]]:
    for path in root.rglob("*.jsonl"):
        with path.open() as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source"] = str(path)
                parts = path.relative_to(root).parts
                if len(parts) >= 4:
                    record.setdefault("benchmark", parts[0])
                    record.setdefault("method", parts[1])
                    record.setdefault("seed", parts[2].replace("seed_", ""))
                yield record


def _float(record: Dict[str, object], key: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate(root: Path, metrics: List[str]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for record in _records(root):
        grouped[(record.get("benchmark", "unknown"), record.get("method", "unknown"))].append(record)

    rows: List[Dict[str, object]] = []
    for (benchmark, method), records in sorted(grouped.items()):
        row: Dict[str, object] = {"benchmark": benchmark, "method": method, "num_records": len(records)}
        for metric in metrics:
            values = [value for record in records if (value := _float(record, metric)) is not None]
            if not values:
                continue
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No result records found.\n")
        return
    fieldnames = ["benchmark", "method", "num_records"] + [
        key for key in sorted({key for row in rows for key in row.keys()})
        if key not in {"benchmark", "method", "num_records"}
    ]
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = []
        for key in fieldnames:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Do-World benchmark JSONL metrics.")
    parser.add_argument("--root", default="experiments/do_world_benchmarks")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--csv", default="experiments/do_world_benchmarks/summary.csv")
    parser.add_argument("--markdown", default="experiments/do_world_benchmarks/summary.md")
    args = parser.parse_args()

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    rows = aggregate(Path(args.root), metrics)
    write_csv(rows, Path(args.csv))
    write_markdown(rows, Path(args.markdown))
    print(f"Wrote {len(rows)} aggregate rows to {args.csv} and {args.markdown}.")


if __name__ == "__main__":
    main()
