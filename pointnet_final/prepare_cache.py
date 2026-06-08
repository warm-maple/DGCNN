from __future__ import annotations

import argparse
from pathlib import Path

from .data import (
    build_cache,
    list_modelnet_split,
    list_prediction_samples,
    list_teacher_train,
    read_class_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fast .npy caches for ModelNet40 txt point clouds.")
    parser.add_argument("--data-root", default="modelnet40_normal_resampled")
    parser.add_argument("--teacher-root", default="dataset/train")
    parser.add_argument("--cache-dir", default="cache/modelnet40")
    parser.add_argument("--points-per-shape", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--predict-root", default=None, help="Optional unlabeled/labeled test root to cache.")
    parser.add_argument("--predict-name", default="predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = read_class_names(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_samples = list_teacher_train(args.teacher_root, class_names)
    build_cache(train_samples, cache_dir, "teacher_train", args.points_per_shape, force=args.force)

    rehearsal_samples = list_modelnet_split(args.data_root, "test", class_names)
    build_cache(rehearsal_samples, cache_dir, "rehearsal_test", args.points_per_shape, force=args.force)

    if args.predict_root:
        predict_samples = list_prediction_samples(args.predict_root, class_names)
        build_cache(predict_samples, cache_dir, args.predict_name, args.points_per_shape, force=args.force)


if __name__ == "__main__":
    main()

