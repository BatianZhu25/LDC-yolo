"""Train MKL-YOLO with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds

import mkl_yolo.register  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MKL-YOLO")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training YAML config")
    parser.add_argument("--data", default=None, help="Override dataset YAML")
    parser.add_argument("--model", default=None, help="Override model YAML")
    parser.add_argument("--device", default=None, help="CUDA device id/list, e.g. 0 or 0,1,2,3")
    parser.add_argument("--name", default=None, help="Run name")
    return parser.parse_args()


def _parse_device(value):
    if value is None:
        return None
    if isinstance(value, str) and "," in value:
        return [int(x) for x in value.split(",") if x.strip()]
    return int(value) if isinstance(value, str) and value.isdigit() else value


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.data:
        cfg["data"] = args.data
    if args.model:
        cfg["model"] = args.model
    if args.device:
        cfg["device"] = _parse_device(args.device)
    if args.name:
        cfg["name"] = args.name

    init_seeds(0, deterministic=True)
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)


if __name__ == "__main__":
    main()
