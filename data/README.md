# Datasets

LDC-YOLO is trained separately on **VisDrone2019-DET** and **UAVDT**, as described in the manuscript. This repository contains configuration and preparation instructions; the original images and labels are obtained from the dataset providers.

| Dataset | Source | Detector classes | Configuration |
|---|---|---:|---|
| VisDrone2019-DET | https://github.com/VisDrone/VisDrone-Dataset | 10 | `configs/data/visdrone.yaml` |
| UAVDT | https://sites.google.com/view/grli-uavdt/ | 3 | `configs/data/uavdt.yaml` |

## YOLO directory layout

Set `path` in the dataset YAML to this local root:

```text
VisDrone/ (or UAVDT/)
  images/train/
  images/val/
  images/test/       # optional; only use when the selected split has annotations
  labels/train/
  labels/val/
  labels/test/
```

Each label line is `class_id x_center y_center width height`, with zero-based class IDs and coordinates normalized by image width/height. Use an image's basename for the corresponding `.txt` file.

## VisDrone2019-DET

Class IDs 0–9: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor. Original categories 1–10 map to YOLO IDs 0–9.

```bash
python scripts/prepare_visdrone.py --source /path/to/VisDrone2019-DET-train --output /path/to/VisDrone --split train
python scripts/prepare_visdrone.py --source /path/to/VisDrone2019-DET-val --output /path/to/VisDrone --split val
```

The converter copies images, clips boxes to image boundaries and skips ignored annotations (score 0), category 0 and category 11. It does not mask ignored image regions. Ultralytics evaluation on converted labels is distinct from the official VisDrone evaluation protocol; exact manuscript evaluator settings remain to be supplied.

## UAVDT

Class IDs: `0 car`, `1 truck`, `2 bus`. Prepare matching YOLO-format images and labels, then set `path` in `configs/data/uavdt.yaml`. Preserve sequence-level train/evaluation separation when preparing video frames. The original experimental sequence lists and raw annotation conversion will be supplied in a later release; no new sequence split is presented as the paper's split here.

## Reported experiment data

`results/main_results.csv` and `results/ablation_visdrone.csv` transcribe the latest manuscript tables. They contain reported metrics, not raw predictions or regenerated evaluation outputs. Dataset image counts, checkpoints and training logs are pending the original records.
