from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: Path
    label: int | None


def read_class_names(data_root: str | Path) -> list[str]:
    path = Path(data_root) / "modelnet40_shape_names.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def infer_class_names_from_dirs(root: str | Path) -> list[str]:
    root = Path(root)
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def list_modelnet_split(data_root: str | Path, split: str, class_names: list[str]) -> list[Sample]:
    data_root = Path(data_root)
    name_to_label = {name: i for i, name in enumerate(class_names)}
    split_path = data_root / f"modelnet40_{split}.txt"
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    samples: list[Sample] = []
    for sample_id in ids:
        cls = class_from_sample_id(sample_id, class_names)
        samples.append(Sample(sample_id, data_root / cls / f"{sample_id}.txt", name_to_label[cls]))
    return samples


def list_training_samples(train_root: str | Path, class_names: list[str]) -> list[Sample]:
    train_root = Path(train_root)
    name_to_label = {name: i for i, name in enumerate(class_names)}
    samples: list[Sample] = []
    for cls in class_names:
        cls_dir = train_root / cls
        for path in sorted(cls_dir.glob("*.txt")):
            samples.append(Sample(path.stem, path, name_to_label[cls]))
    return samples


def stratified_train_val_split(
    samples: list[Sample],
    val_fraction: float = 0.1,
    seed: int = 2026,
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if any(sample.label is None for sample in samples):
        raise ValueError("All samples must have labels for a stratified split")

    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        assert sample.label is not None
        by_label.setdefault(sample.label, []).append(sample)

    rng = np.random.default_rng(seed)
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    for label in sorted(by_label):
        class_samples = sorted(by_label[label], key=lambda sample: sample.sample_id)
        if len(class_samples) < 2:
            raise ValueError(f"Class {label} needs at least two samples")
        order = rng.permutation(len(class_samples))
        val_count = int(round(len(class_samples) * val_fraction))
        val_count = min(max(val_count, 1), len(class_samples) - 1)
        val_indices = set(int(index) for index in order[:val_count])
        for index, sample in enumerate(class_samples):
            if index in val_indices:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    train_samples.sort(key=lambda sample: sample.sample_id)
    val_samples.sort(key=lambda sample: sample.sample_id)
    return train_samples, val_samples


def list_prediction_samples(root: str | Path, class_names: list[str]) -> list[Sample]:
    root = Path(root)
    name_to_label = {name: i for i, name in enumerate(class_names)}
    samples: list[Sample] = []

    class_dirs = [root / cls for cls in class_names if (root / cls).is_dir()]
    if class_dirs:
        for cls in class_names:
            cls_dir = root / cls
            if not cls_dir.is_dir():
                continue
            for path in sorted(cls_dir.glob("*.txt")):
                if path.name.startswith("._"):
                    continue
                samples.append(Sample(path.stem, path, name_to_label[cls]))
        return samples

    for path in sorted(root.glob("*.txt")):
        if path.name.startswith("._"):
            continue
        inferred = infer_label_from_id(path.stem, class_names)
        samples.append(Sample(path.stem, path, inferred))
    return samples


def list_samples_from_ids(
    data_root: str | Path,
    id_list: str | Path,
    class_names: list[str],
    labeled: bool = True,
) -> list[Sample]:
    data_root = Path(data_root)
    ids = [line.strip().split(",")[0] for line in Path(id_list).read_text().splitlines() if line.strip()]
    samples: list[Sample] = []
    for sample_id in ids:
        cls = class_from_sample_id(sample_id, class_names)
        label = class_names.index(cls) if labeled else None
        samples.append(Sample(sample_id, data_root / cls / f"{sample_id}.txt", label))
    return samples


def class_from_sample_id(sample_id: str, class_names: Iterable[str]) -> str:
    for cls in sorted(class_names, key=len, reverse=True):
        if sample_id.startswith(cls + "_"):
            return cls
    raise ValueError(f"Cannot infer class from sample id: {sample_id}")


def infer_label_from_id(sample_id: str, class_names: list[str]) -> int | None:
    try:
        cls = class_from_sample_id(sample_id, class_names)
    except ValueError:
        return None
    return class_names.index(cls)


def sample_hash(samples: list[Sample]) -> str:
    payload = "\n".join(f"{s.sample_id}|{s.path.resolve()}" for s in samples)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def read_point_file(path: Path) -> np.ndarray:
    try:
        points = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except ValueError:
        points = np.loadtxt(path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"Point file has fewer than 3 columns: {path}")
    if points.shape[1] < 6:
        normals = np.zeros((points.shape[0], 3), dtype=np.float32)
        points = np.concatenate([points[:, :3], normals], axis=1)
    return points[:, :6].astype(np.float32, copy=False)


def cache_paths(cache_dir: str | Path, name: str) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {
        "points": cache_dir / f"{name}_points.npy",
        "labels": cache_dir / f"{name}_labels.npy",
        "meta": cache_dir / f"{name}_meta.json",
    }


def build_cache(
    samples: list[Sample],
    cache_dir: str | Path,
    name: str,
    points_per_shape: int = 10000,
    force: bool = False,
) -> dict[str, Path]:
    paths = cache_paths(cache_dir, name)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if not force and all(path.exists() for path in paths.values()):
        meta = json.loads(paths["meta"].read_text())
        if meta.get("hash") == sample_hash(samples) and meta.get("points_per_shape") == points_per_shape:
            return paths

    points_mm = np.lib.format.open_memmap(
        paths["points"],
        mode="w+",
        dtype=np.float16,
        shape=(len(samples), points_per_shape, 6),
    )
    labels = np.full((len(samples),), -1, dtype=np.int64)
    rng = np.random.default_rng(12345)
    for i, sample in enumerate(samples):
        points = read_point_file(sample.path)
        if len(points) >= points_per_shape:
            points = points[:points_per_shape]
        else:
            choice = rng.choice(len(points), size=points_per_shape - len(points), replace=True)
            points = np.concatenate([points, points[choice]], axis=0)
        points_mm[i] = points.astype(np.float16)
        labels[i] = -1 if sample.label is None else sample.label
        if (i + 1) % 500 == 0 or i + 1 == len(samples):
            print(f"cached {i + 1}/{len(samples)} samples for {name}", flush=True)
    points_mm.flush()
    np.save(paths["labels"], labels)
    meta = {
        "name": name,
        "hash": sample_hash(samples),
        "points_per_shape": points_per_shape,
        "num_samples": len(samples),
        "ids": [s.sample_id for s in samples],
        "paths": [str(s.path) for s in samples],
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.copy()
    xyz = points[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    scale = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    if scale > 0:
        xyz /= scale
    points[:, :3] = xyz
    normals = points[:, 3:6]
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normal_norm[normal_norm < 1e-6] = 1.0
    points[:, 3:6] = normals / normal_norm
    return points


def augment_points(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    points = points.copy()
    if rng.random() < 0.8:
        dropout_ratio = rng.random() * 0.875
        drop_idx = rng.random(points.shape[0]) <= dropout_ratio
        if np.any(drop_idx):
            points[drop_idx, :] = points[0, :]

    scale = rng.uniform(2.0 / 3.0, 3.0 / 2.0)
    shift = rng.uniform(-0.2, 0.2, size=(1, 3))
    points[:, :3] = points[:, :3] * scale + shift
    jitter = np.clip(0.01 * rng.standard_normal(size=points[:, :3].shape), -0.02, 0.02)
    points[:, :3] += jitter
    rng.shuffle(points, axis=0)
    return points


class CachedPointCloudDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        name: str,
        num_points: int = 1024,
        use_normals: bool = True,
        augment: bool = False,
        fixed_sample_seed: int | None = None,
    ) -> None:
        paths = cache_paths(cache_dir, name)
        self.points = np.load(paths["points"], mmap_mode="r")
        self.labels = np.load(paths["labels"])
        self.meta = json.loads(paths["meta"].read_text())
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment = augment
        self.fixed_sample_seed = fixed_sample_seed

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def ids(self) -> list[str]:
        return list(self.meta["ids"])

    def _rng(self, index: int) -> np.random.Generator:
        if self.fixed_sample_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.fixed_sample_seed + index * 10007)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._rng(index)
        points = np.asarray(self.points[index], dtype=np.float32)
        replace = len(points) < self.num_points
        choice = rng.choice(len(points), size=self.num_points, replace=replace)
        points = points[choice]
        points = normalize_points(points)
        if self.augment:
            points = augment_points(points, rng)
        if not self.use_normals:
            points = points[:, :3]
        label = int(self.labels[index])
        return torch.from_numpy(points.astype(np.float32)), torch.tensor(label, dtype=torch.long)
