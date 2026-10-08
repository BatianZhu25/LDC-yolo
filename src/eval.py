"""Evaluate LDC-YOLO checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

import ldc_yolo.register  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LDC-YOLO")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to evaluation YAML config")
    parser.add_argument("--model", default=None, help="Override checkpoint path")
    parser.add_argument("--data", default=None, help="Override dataset YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["model"] = args.model
    if args.data:
        cfg["data"] = args.data
    model = YOLO(cfg.pop("model"))
    model.val(**cfg)


if __name__ == "__main__":
    main()
