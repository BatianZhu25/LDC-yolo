#!/usr/bin/env bash
set -euo pipefail

python src/train.py --config configs/train.yaml --data configs/data/visdrone.yaml --name mkl_yolo_s_visdrone
