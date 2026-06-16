from __future__ import annotations

import argparse
from pathlib import Path

from .data import (
    build_cache,
    infer_class_names_from_dirs,
    list_prediction_samples,
    list_training_samples,
    read_class_names,
    stratified_train_val_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fast .npy caches for ModelNet40 txt point clouds.")
    parser.add_argument("--data-root", default="data/modelnet40")
    parser.add_argument("--train-root", default="dataset/train")
    parser.add_argument("--cache-dir", default="cache/modelnet40")
    parser.add_argument("--points-per-shape", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--predict-root", default=None, help="Optional unlabeled/labeled test root to cache.")
    parser.add_argument("--predict-name", default="predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        class_names = read_class_names(args.data_root)
    except FileNotFoundError:
        class_names = infer_class_names_from_dirs(args.train_root)
    if not class_names:
        raise ValueError("No class names found. Check --data-root or --train-root.")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_samples = list_training_samples(args.train_root, class_names)
    train_samples, val_samples = stratified_train_val_split(
        all_samples,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )
    build_cache(train_samples, cache_dir, "train_split", args.points_per_shape, force=args.force)
    build_cache(val_samples, cache_dir, "val_split", args.points_per_shape, force=args.force)

    if args.predict_root:
        predict_samples = list_prediction_samples(args.predict_root, class_names)
        build_cache(predict_samples, cache_dir, args.predict_name, args.points_per_shape, force=args.force)


if __name__ == "__main__":
    main()

