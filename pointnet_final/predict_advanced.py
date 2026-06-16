from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import build_cache, cache_paths, list_prediction_samples, list_samples_from_ids, normalize_points, read_class_names
from .dgcnn import DGCNNClassifier
from .metrics import class_accuracy, instance_accuracy


class AdvancedVoteDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        name: str,
        num_points: int,
        use_normals: bool,
        vote: int,
        seed: int,
        sampling: str,
    ) -> None:
        paths = cache_paths(cache_dir, name)
        self.points = np.load(paths["points"], mmap_mode="r")
        self.labels = np.load(paths["labels"])
        self.meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        self.num_points = num_points
        self.use_normals = use_normals
        self.vote = vote
        self.seed = seed
        self.sampling = sampling

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def ids(self) -> list[str]:
        return list(self.meta["ids"])

    def _coverage_indices(self, index: int, length: int) -> np.ndarray:
        # A coprime stride traverses the full cached cloud without constructing a permutation.
        stride = 7919
        while math.gcd(stride, length) != 1:
            stride -= 2
        sequence_offset = self.vote * self.num_points
        base = (self.seed + index * 10007) % length
        sequence = sequence_offset + np.arange(self.num_points, dtype=np.int64)
        return (base + sequence * stride) % length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        points = np.asarray(self.points[index], dtype=np.float32)
        points = normalize_points(points)
        if self.sampling == "coverage":
            choice = self._coverage_indices(index, len(points))
        else:
            rng = np.random.default_rng(self.seed + self.vote * 7919 + index * 10007)
            choice = rng.choice(len(points), size=self.num_points, replace=len(points) < self.num_points)
        points = points[choice]
        if not self.use_normals:
            points = points[:, :3]
        return torch.from_numpy(points.astype(np.float32)), torch.tensor(int(self.labels[index]), dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast multi-vote inference with coverage sampling and AMP.")
    parser.add_argument("--data-root", default="data/modelnet40")
    parser.add_argument("--class-names", default=None, help="Optional class-name txt/json file.")
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--unlabeled-list", action="store_true")
    parser.add_argument("--cache-dir", default="cache/modelnet40")
    parser.add_argument("--cache-name", default="predict_advanced")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--model-weights", nargs="+", type=float, default=None)
    parser.add_argument("--output", default="submission_advanced.csv")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--points-per-shape", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--votes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampling", choices=["coverage", "random"], default="coverage")
    parser.add_argument(
        "--reflection-tta",
        choices=["none", "xz4"],
        default="none",
        help="Average original, x-reflected, z-reflected and xz-reflected predictions.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    return parser.parse_args()


def read_class_names_file(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return list(json.loads(path.read_text(encoding="utf-8")))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[DGCNNClassifier, dict, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get("args", {})
    class_names = checkpoint["class_names"]
    model = DGCNNClassifier(
        num_classes=len(class_names),
        input_dims=3 if model_args.get("no_normals", False) else 6,
        k=int(model_args.get("k", 20)),
        emb_dims=int(model_args.get("emb_dims", 1024)),
        dropout=float(model_args.get("dropout", 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, model_args, class_names


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    if args.class_names:
        class_names = read_class_names_file(args.class_names)
    else:
        checkpoint_preview = torch.load(checkpoint_paths[0], map_location="cpu")
        class_names = list(checkpoint_preview["class_names"])
    if args.test_list:
        samples = list_samples_from_ids(args.data_root, args.test_list, class_names, labeled=not args.unlabeled_list)
    elif args.test_root:
        samples = list_prediction_samples(args.test_root, class_names)
    else:
        raise SystemExit("Provide either --test-root or --test-list.")
    if not samples:
        raise SystemExit("No point clouds found.")

    build_cache(samples, args.cache_dir, args.cache_name, args.points_per_shape, force=args.force_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    loaded = [load_model(path, device) for path in checkpoint_paths]
    models = [item[0] for item in loaded]
    model_args = [item[1] for item in loaded]
    for path, (_, _, checkpoint_classes) in zip(checkpoint_paths, loaded):
        if checkpoint_classes != class_names:
            raise ValueError(f"Class names do not match: {path}")

    model_weights = args.model_weights or [1.0] * len(models)
    if len(model_weights) != len(models):
        raise ValueError("--model-weights must contain one value per checkpoint")
    weight_sum = sum(model_weights)
    model_weights = [weight / weight_sum for weight in model_weights]

    num_points = args.num_points or int(model_args[0].get("num_points", 1024))
    use_normals = not bool(model_args[0].get("no_normals", False))
    logits_sum = torch.zeros((len(samples), len(class_names)), dtype=torch.float32)
    labels_np: np.ndarray | None = None
    ids: list[str] | None = None
    vote_history: list[dict[str, float | int]] = []
    amp_enabled = device.type == "cuda" and not args.no_amp
    reflection_signs = [(1.0, 1.0)]
    if args.reflection_tta == "xz4":
        reflection_signs = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
    start_time = time.perf_counter()

    for vote in range(args.votes):
        dataset = AdvancedVoteDataset(
            args.cache_dir,
            args.cache_name,
            num_points=num_points,
            use_normals=use_normals,
            vote=vote,
            seed=args.seed,
            sampling=args.sampling,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        if labels_np is None:
            labels_np = dataset.labels.astype(np.int64)
            ids = dataset.ids

        offset = 0
        for points, _labels in loader:
            points = points.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
            batch_logits = torch.zeros((points.size(0), len(class_names)), dtype=torch.float32, device=device)
            for x_sign, z_sign in reflection_signs:
                transformed = points
                if x_sign < 0 or z_sign < 0:
                    transformed = points.clone()
                    transformed[:, 0, :] *= x_sign
                    transformed[:, 2, :] *= z_sign
                    if transformed.size(1) >= 6:
                        transformed[:, 3, :] *= x_sign
                        transformed[:, 5, :] *= z_sign
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    for weight, model in zip(model_weights, models):
                        batch_logits.add_(
                            model(transformed).float(),
                            alpha=weight / len(reflection_signs),
                        )
            batch = points.size(0)
            logits_sum[offset : offset + batch] += batch_logits.cpu()
            offset += batch
        elapsed = time.perf_counter() - start_time
        vote_report: dict[str, float | int] = {
            "vote": vote + 1,
            "elapsed_seconds": elapsed,
        }
        if labels_np is not None and np.any(labels_np >= 0):
            current_predictions = logits_sum.argmax(dim=1).numpy()
            vote_report["instance_acc"] = instance_accuracy(current_predictions, labels_np)
            vote_report["class_acc"] = class_accuracy(current_predictions, labels_np, len(class_names))
            print(
                f"finished vote {vote + 1}/{args.votes}, "
                f"instance={vote_report['instance_acc']:.6f}, "
                f"class={vote_report['class_acc']:.6f}, elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(f"finished vote {vote + 1}/{args.votes}, elapsed={elapsed:.1f}s", flush=True)
        vote_history.append(vote_report)

    assert ids is not None and labels_np is not None
    predictions = logits_sum.argmax(dim=1).numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for sample_id, prediction in zip(ids, predictions):
            writer.writerow([sample_id, class_names[int(prediction)]])

    elapsed = time.perf_counter() - start_time
    report: dict[str, object] = {
        "num_samples": len(predictions),
        "votes": args.votes,
        "sampling": args.sampling,
        "reflection_tta": args.reflection_tta,
        "amp": amp_enabled,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "seconds_per_vote": elapsed / max(args.votes, 1),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "model_weights": model_weights,
        "vote_history": vote_history,
    }
    if np.any(labels_np >= 0):
        report["instance_acc"] = instance_accuracy(predictions, labels_np)
        report["class_acc"] = class_accuracy(predictions, labels_np, len(class_names))
    report_path = output.with_suffix(output.suffix + ".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.save_logits:
        np.save(output.with_suffix(output.suffix + ".logits.npy"), logits_sum.numpy())

    print(f"wrote {len(predictions)} predictions to {output}", flush=True)
    if "instance_acc" in report:
        print(
            f"metrics: instance={report['instance_acc']:.6f}, "
            f"class={report['class_acc']:.6f}, elapsed={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
