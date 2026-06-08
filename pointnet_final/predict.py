from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    CachedPointCloudDataset,
    build_cache,
    list_samples_from_ids,
    list_prediction_samples,
    read_class_names,
)
from .dgcnn import DGCNNClassifier
from .metrics import class_accuracy, instance_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ModelNet40 labels and write the required CSV.")
    parser.add_argument("--data-root", default="modelnet40_normal_resampled")
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--test-list", default=None, help="Optional file containing sample ids to predict.")
    parser.add_argument("--unlabeled-list", action="store_true", help="Do not infer labels from --test-list ids.")
    parser.add_argument("--cache-dir", default="cache/modelnet40")
    parser.add_argument("--cache-name", default="predict")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--points-per-shape", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--votes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[DGCNNClassifier, dict, list[str]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})
    class_names = ckpt["class_names"]
    input_dims = 3 if args.get("no_normals", False) else 6
    model = DGCNNClassifier(
        num_classes=len(class_names),
        input_dims=input_dims,
        k=int(args.get("k", 20)),
        emb_dims=int(args.get("emb_dims", 1024)),
        dropout=float(args.get("dropout", 0.5)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, args, class_names


@torch.no_grad()
def main() -> None:
    args = parse_args()
    class_names = read_class_names(args.data_root)
    if args.test_list:
        samples = list_samples_from_ids(args.data_root, args.test_list, class_names, labeled=not args.unlabeled_list)
    elif args.test_root:
        samples = list_prediction_samples(args.test_root, class_names)
    else:
        raise SystemExit("Provide either --test-root or --test-list.")
    if not samples:
        raise SystemExit(f"No .txt point clouds found under {args.test_root}")
    build_cache(samples, args.cache_dir, args.cache_name, args.points_per_shape, force=args.force_cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = [Path(p) for p in args.checkpoints]
    models = []
    model_args = []
    for ckpt_path in checkpoints:
        model, ckpt_args, ckpt_classes = load_model(ckpt_path, device)
        if ckpt_classes != class_names:
            raise ValueError(f"Class names in checkpoint do not match {args.data_root}: {ckpt_path}")
        models.append(model)
        model_args.append(ckpt_args)

    num_points = args.num_points or int(model_args[0].get("num_points", 1024))
    use_normals = not bool(model_args[0].get("no_normals", False))
    logits_sum = torch.zeros((len(samples), len(class_names)), dtype=torch.float32)
    labels_np: np.ndarray | None = None
    ids: list[str] | None = None

    for vote in range(args.votes):
        dataset = CachedPointCloudDataset(
            args.cache_dir,
            args.cache_name,
            num_points=num_points,
            use_normals=use_normals,
            augment=False,
            fixed_sample_seed=args.seed + vote * 7919,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )
        if labels_np is None:
            labels_np = dataset.labels.astype(np.int64)
            ids = dataset.ids
        offset = 0
        for points, _labels in loader:
            points = points.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
            batch_logits = torch.zeros((points.size(0), len(class_names)), device=device)
            for model in models:
                batch_logits += model(points)
            batch_logits /= len(models)
            batch = points.size(0)
            logits_sum[offset : offset + batch] += batch_logits.detach().float().cpu()
            offset += batch
        print(f"finished vote {vote + 1}/{args.votes}", flush=True)

    assert ids is not None and labels_np is not None
    preds = logits_sum.argmax(dim=1).numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sample_id, pred in zip(ids, preds):
            writer.writerow([sample_id, class_names[int(pred)]])

    labeled = labels_np >= 0
    print(f"wrote {len(preds)} predictions to {output}", flush=True)
    if np.any(labeled):
        inst = instance_accuracy(preds, labels_np)
        cls_acc = class_accuracy(preds, labels_np, len(class_names))
        report = {"instance_acc": inst, "class_acc": cls_acc, "num_samples": int(np.sum(labeled))}
        report_path = output.with_suffix(output.suffix + ".metrics.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"metrics: instance={inst:.6f}, class={cls_acc:.6f}", flush=True)


if __name__ == "__main__":
    main()
