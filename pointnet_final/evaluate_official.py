from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .data import build_cache, list_modelnet_split, read_class_names
from .predict import load_model
from .train import evaluate


OFFICIAL_TEST_CACHE_NAME = "official_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen checkpoints once on the official ModelNet40 test split.")
    parser.add_argument("--data-root", default=r"F:\Python Project\pointnet\modelnet40_normal_resampled")
    parser.add_argument("--cache-dir", default="cache/modelnet40")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--votes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--points-per-shape", type=int, default=10000)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    class_names = read_class_names(args.data_root)
    official_samples = list_modelnet_split(args.data_root, "test", class_names)
    build_cache(
        official_samples,
        args.cache_dir,
        OFFICIAL_TEST_CACHE_NAME,
        args.points_per_shape,
        force=args.force_cache,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    summary: list[dict[str, object]] = []
    for checkpoint_path in args.checkpoints:
        model, model_args, checkpoint_classes = load_model(checkpoint_path, device)
        if checkpoint_classes != class_names:
            raise ValueError(f"Class names do not match: {checkpoint_path}")
        eval_args = SimpleNamespace(
            cache_dir=args.cache_dir,
            eval_batch_size=args.batch_size,
            num_points=int(model_args.get("num_points", 1024)),
            no_normals=bool(model_args.get("no_normals", False)),
            workers=args.workers,
            seed=int(model_args.get("seed", 1)),
            batch_size=args.batch_size,
        )
        result = evaluate(
            model,
            eval_args,
            class_names,
            votes=args.votes,
            split_name=OFFICIAL_TEST_CACHE_NAME,
        )
        record = {
            "checkpoint": str(checkpoint_path),
            "votes": args.votes,
            "instance_acc": result["instance_acc"],
            "class_acc": result["class_acc"],
            "num_samples": len(result["labels"]),
            "confusion": result["confusion"],
        }
        output_path = output_dir / f"{Path(checkpoint_path).stem}_official_test_eval.json"
        output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        summary.append(record)
        print(
            f"{checkpoint_path}: instance={record['instance_acc']:.6f}, "
            f"class={record['class_acc']:.6f}",
            flush=True,
        )

    (output_dir / "official_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
