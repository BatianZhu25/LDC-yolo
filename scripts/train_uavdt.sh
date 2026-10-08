#!/usr/bin/env bash
set -euo pipefail

python src/train.py --config configs/train.yaml --data configs/data/uavdt.yaml --name ldc_yolo_s_uavdt "$@"
