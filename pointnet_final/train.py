from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import (
    CachedPointCloudDataset,
    build_cache,
    list_modelnet_split,
    list_teacher_train,
    read_class_names,
)
from .dgcnn import DGCNNClassifier
from .metrics import class_accuracy, confusion_matrix, instance_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DGCNN on the teacher ModelNet40 train split.")
    parser.add_argument("--data-root", default="modelnet40_normal_resampled")
    parser.add_argument("--teacher-root", default="dataset/train")
    parser.add_argument("--cache-dir", default="cache/modelnet40")
    parser.add_argument("--run-dir", default="runs/dgcnn_normals_seed1")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--points-per-shape", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--emb-dims", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-votes", type=int, default=1)
    parser.add_argument("--final-votes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-normals", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-model-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def ensure_caches(args: argparse.Namespace, class_names: list[str]) -> None:
    cache_dir = Path(args.cache_dir)
    train_samples = list_teacher_train(args.teacher_root, class_names)
    test_samples = list_modelnet_split(args.data_root, "test", class_names)
    build_cache(train_samples, cache_dir, "teacher_train", args.points_per_shape)
    build_cache(test_samples, cache_dir, "rehearsal_test", args.points_per_shape)


def make_loader(
    cache_dir: str | Path,
    name: str,
    args: argparse.Namespace,
    train: bool,
    fixed_sample_seed: int | None = None,
) -> DataLoader:
    dataset = CachedPointCloudDataset(
        cache_dir=cache_dir,
        name=name,
        num_points=args.num_points,
        use_normals=not args.no_normals,
        augment=train,
        fixed_sample_seed=fixed_sample_seed,
    )
    sampler = None
    shuffle = train
    if train and getattr(args, "balanced_sampler", False):
        labels = dataset.labels
        counts = np.bincount(labels[labels >= 0], minlength=40).astype(np.float64)
        weights = np.zeros_like(labels, dtype=np.float64)
        for cls in range(len(counts)):
            weights[labels == cls] = 1.0 / max(counts[cls], 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=worker_init_fn if train else None,
        persistent_workers=args.workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    args: argparse.Namespace,
    class_names: list[str],
    votes: int = 1,
    split_name: str = "rehearsal_test",
) -> dict[str, object]:
    device = next(model.parameters()).device
    model.eval()
    num_classes = len(class_names)
    logits_sum: torch.Tensor | None = None
    labels_np: np.ndarray | None = None
    ids: list[str] | None = None

    for vote in range(votes):
        loader = make_loader(
            args.cache_dir,
            split_name,
            args,
            train=False,
            fixed_sample_seed=args.seed + 100000 + vote * 7919,
        )
        offset = 0
        if logits_sum is None:
            dataset = loader.dataset
            logits_sum = torch.zeros((len(dataset), num_classes), dtype=torch.float32)
            labels_np = dataset.labels.astype(np.int64)
            ids = dataset.ids
        for points, _labels in loader:
            points = points.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
            logits = model(points).detach().float().cpu()
            batch = logits.shape[0]
            logits_sum[offset : offset + batch] += logits
            offset += batch

    assert logits_sum is not None and labels_np is not None and ids is not None
    preds = logits_sum.argmax(dim=1).numpy()
    inst = instance_accuracy(preds, labels_np)
    cls_acc = class_accuracy(preds, labels_np, num_classes)
    return {
        "instance_acc": inst,
        "class_acc": cls_acc,
        "preds": preds.tolist(),
        "labels": labels_np.tolist(),
        "ids": ids,
        "confusion": confusion_matrix(preds, labels_np, num_classes).tolist(),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_instance_acc: float,
    best_class_acc: float,
    args: argparse.Namespace,
    class_names: list[str],
    best_class_only_acc: float | None = None,
    best_balanced_score: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_instance_acc": best_instance_acc,
            "best_class_acc": best_class_acc,
            "best_class_only_acc": best_class_only_acc,
            "best_balanced_score": best_balanced_score,
            "args": vars(args),
            "class_names": class_names,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(args.data_root)
    (run_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    ensure_caches(args, class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = DGCNNClassifier(
        num_classes=len(class_names),
        input_dims=3 if args.no_normals else 6,
        k=args.k,
        emb_dims=args.emb_dims,
        dropout=args.dropout,
    ).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    loss_weight = None
    if args.class_weight_power > 0:
        train_labels = np.load(Path(args.cache_dir) / "teacher_train_labels.npy")
        counts = np.bincount(train_labels[train_labels >= 0], minlength=len(class_names)).astype(np.float64)
        weights = np.power(np.maximum(counts, 1.0), -args.class_weight_power)
        weights = weights / weights.mean()
        loss_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=args.label_smoothing)
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_instance_acc = -math.inf
    best_class_acc = -math.inf
    best_class_only_acc = -math.inf
    best_balanced_score = -math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if not args.resume_model_only:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt.get("scaler_state", {}))
            start_epoch = int(ckpt["epoch"]) + 1
            best_instance_acc = float(ckpt.get("best_instance_acc", best_instance_acc))
            best_class_acc = float(ckpt.get("best_class_acc", best_class_acc))
            best_class_only_acc = float(ckpt.get("best_class_only_acc", best_class_only_acc))
            best_balanced_score = float(ckpt.get("best_balanced_score", best_balanced_score))

    train_loader = make_loader(args.cache_dir, "teacher_train", args, train=True)
    metrics_path = run_dir / "metrics.jsonl"
    print(f"Training on {device} for {args.epochs} epochs; run_dir={run_dir}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        for points, labels in train_loader:
            points = points.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(points)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu()) * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().detach().cpu())
            total += labels.size(0)

        scheduler.step()
        train_loss = epoch_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        elapsed = time.time() - start
        lr = scheduler.get_last_lr()[0]

        row: dict[str, object] = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "seconds": elapsed,
        }

        if epoch % args.eval_every == 0:
            eval_result = evaluate(model, args, class_names, votes=args.eval_votes)
            inst = float(eval_result["instance_acc"])
            cls_acc = float(eval_result["class_acc"])
            row.update({"rehearsal_instance_acc": inst, "rehearsal_class_acc": cls_acc})
            improved = inst > best_instance_acc or (math.isclose(inst, best_instance_acc) and cls_acc > best_class_acc)
            if improved:
                best_instance_acc = inst
                best_class_acc = cls_acc
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_instance_acc,
                    best_class_acc,
                    args,
                    class_names,
                    best_class_only_acc,
                    best_balanced_score,
                )
                (run_dir / "best_eval.json").write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
            if cls_acc > best_class_only_acc:
                best_class_only_acc = cls_acc
                save_checkpoint(
                    run_dir / "best_class.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_instance_acc,
                    best_class_acc,
                    args,
                    class_names,
                    best_class_only_acc,
                    best_balanced_score,
                )
                (run_dir / "best_class_eval.json").write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
            balanced_score = inst + cls_acc
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                save_checkpoint(
                    run_dir / "best_balanced.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_instance_acc,
                    best_class_acc,
                    args,
                    class_names,
                    best_class_only_acc,
                    best_balanced_score,
                )
                (run_dir / "best_balanced_eval.json").write_text(json.dumps(eval_result, indent=2), encoding="utf-8")

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_instance_acc,
            best_class_acc,
            args,
            class_names,
            best_class_only_acc,
            best_balanced_score,
        )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            "epoch {epoch:03d} loss={loss:.4f} train={train:.4f} "
            "rehearsal={inst:.4f}/{cls:.4f} best={best_inst:.4f}/{best_cls:.4f} "
            "lr={lr:.6f} {sec:.1f}s".format(
                epoch=epoch,
                loss=train_loss,
                train=train_acc,
                inst=row.get("rehearsal_instance_acc", float("nan")),
                cls=row.get("rehearsal_class_acc", float("nan")),
                best_inst=best_instance_acc,
                best_cls=best_class_acc,
                lr=lr,
                sec=elapsed,
            ),
            flush=True,
        )

    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists() and args.final_votes > 0:
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        final = evaluate(model, args, class_names, votes=args.final_votes)
        (run_dir / f"final_eval_{args.final_votes}votes.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(
            f"Final {args.final_votes}-vote rehearsal: "
            f"instance={final['instance_acc']:.6f}, class={final['class_acc']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
