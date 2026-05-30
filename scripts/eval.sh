#!/usr/bin/env bash
set -euo pipefail

python src/eval.py --config configs/eval.yaml "$@"
