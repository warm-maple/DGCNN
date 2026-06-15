from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def balanced_score(record: dict[str, Any]) -> float:
    return float(record["instance_acc"]) + float(record["class_acc"])


def rank_top_checkpoints(records: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (balanced_score(record), float(record["instance_acc"]), -int(record["epoch"])),
        reverse=True,
    )[:limit]


def retain_top_checkpoint(
    run_dir: str | Path,
    checkpoint_path: str | Path,
    record: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "top_checkpoints.json"
    records: list[dict[str, Any]] = []
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))

    entry = {
        "epoch": int(record["epoch"]),
        "instance_acc": float(record["instance_acc"]),
        "class_acc": float(record["class_acc"]),
        "balanced_score": balanced_score(record),
        "path": str(Path(checkpoint_path).name),
    }
    records = [existing for existing in records if int(existing["epoch"]) != entry["epoch"]]
    records.append(entry)
    kept = rank_top_checkpoints(records, limit=limit)
    kept_paths = {str(item["path"]) for item in kept}

    for existing in records:
        path = run_dir / str(existing["path"])
        if str(existing["path"]) not in kept_paths and path.exists():
            path.unlink()

    manifest_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    return kept
