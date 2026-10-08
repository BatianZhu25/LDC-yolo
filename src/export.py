"""Export LDC-YOLO checkpoints."""

from __future__ import annotations

import argparse

from ultralytics import YOLO

import ldc_yolo.register  # noqa: F401
from ldc_yolo import fuse_all_reparam_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LDC-YOLO")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--format", default="onnx", help="Export format supported by Ultralytics")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--fuse-reparam", action="store_true", help="Fuse re-parameterizable LDC blocks before export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    if args.fuse_reparam:
        model.model.eval()
        fuse_all_reparam_modules(model.model)
    model.export(format=args.format, imgsz=args.imgsz, device=args.device, half=args.half)


if __name__ == "__main__":
    main()
