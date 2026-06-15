from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average compatible checkpoints into one inference model.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    checkpoints = [torch.load(path, map_location="cpu") for path in checkpoint_paths]

    class_names = checkpoints[0]["class_names"]
    model_keys = list(checkpoints[0]["model_state"].keys())
    for path, checkpoint in zip(checkpoint_paths[1:], checkpoints[1:]):
        if checkpoint["class_names"] != class_names:
            raise ValueError(f"Class names do not match: {path}")
        if list(checkpoint["model_state"].keys()) != model_keys:
            raise ValueError(f"Model structure does not match: {path}")

    weights = args.weights or [1.0] * len(checkpoints)
    if len(weights) != len(checkpoints):
        raise ValueError("--weights must contain one value per checkpoint")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Checkpoint weights must have a positive sum")
    weights = [weight / weight_sum for weight in weights]

    averaged_state: dict[str, torch.Tensor] = {}
    for key in model_keys:
        tensors = [checkpoint["model_state"][key] for checkpoint in checkpoints]
        if torch.is_floating_point(tensors[0]):
            averaged = torch.zeros_like(tensors[0], dtype=torch.float32)
            for weight, tensor in zip(weights, tensors):
                averaged.add_(tensor.float(), alpha=weight)
            averaged_state[key] = averaged.to(dtype=tensors[0].dtype)
        else:
            averaged_state[key] = tensors[-1].clone()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": averaged_state,
            "args": checkpoints[-1].get("args", {}),
            "class_names": class_names,
            "soup_sources": [str(path) for path in checkpoint_paths],
            "soup_weights": weights,
        },
        output,
    )
    print(f"wrote model soup to {output}")
    print("weights:", ", ".join(f"{path.name}={weight:.3f}" for path, weight in zip(checkpoint_paths, weights)))


if __name__ == "__main__":
    main()

