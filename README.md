______________________________________________________________________

<div align="center">

# MKL-YOLO

<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</a>
<a href="https://docs.ultralytics.com/">
  <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-YOLO-111F68">
</a>
<a href="./LICENSE">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green">
</a>

</div>

---

## Abstract

MKL-YOLO is a lightweight object detector for aerial small-object detection. The model uses a P2/P3/P4 detection head, lightweight GhostNetV2-style downsampling blocks, an edge-enhanced C2f block, a multi-kernel re-parameterizable C2f block, and a DCN-enhanced detection head. Experiments are conducted on VisDrone and UAVDT under a 300-epoch training setting.

---

## Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd MKL-yolo

# 2. Create conda environment
conda env create -f environment.yaml
conda activate mkl-yolo

# 3. Install the project
pip install -e .
```

If you install dependencies with pip only:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Dataset Configuration

### 1. Dataset Config Files Location

```text
configs/data/
├── visdrone.yaml
└── uavdt.yaml
```

### 2. Modify Dataset Path

Edit the dataset config before training:

```yaml
# Example: configs/data/visdrone.yaml
path: /path/to/VisDrone
train: images/train
val: images/val
test: images/test
```

---

## Training

Train MKL-YOLO-S on VisDrone:

```bash
bash scripts/train_visdrone.sh
```

Train MKL-YOLO-S on UAVDT:

```bash
bash scripts/train_uavdt.sh
```

Equivalent Python entry:

```bash
python src/train.py \
  --config configs/train.yaml \
  --data configs/data/visdrone.yaml \
  --name mkl_yolo_s_visdrone
```

---

## Evaluation

```bash
python src/eval.py \
  --config configs/eval.yaml \
  --model runs/train/mkl_yolo_s_visdrone/weights/best.pt \
  --data configs/data/visdrone.yaml
```

---

## Model Configuration

The released full model configs are:

```text
configs/model/mkl_yolov8n.yaml
configs/model/mkl_yolov8s.yaml
```

The original development name of this config was `EGF_head_yolov8.yaml`; the released project uses the final paper name MKL-YOLO.

---

## Results

### VisDrone

| Method | Epochs | AP50 | AP50:95 | Params | FLOPs | Latency fp16 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5-nu (2020) | 300 | 33.8 | 19.7 | 2.65M | 7.7G | 1.15ms |
| YOLOv5-su (2020) | 300 | 40.2 | 24.2 | 9.14M | 24.0G | 1.28ms |
| YOLOv5-mu (2020) | 300 | 44.3 | 27.3 | 25.10M | 64.2G | 2.45ms |
| YOLOv8-n (2023) | 300 | 34.1 | 20.0 | 3.15M | 8.7G | 0.98ms |
| YOLOv8-s (2023) | 300 | 40.1 | 24.2 | 11.16M | 28.6G | 1.45ms |
| YOLOv8-m (2023) | 300 | 45.0 | 27.9 | 25.89M | 78.9G | 3.07ms |
| YOLOv10-n (2024) | 300 | 33.0 | 19.7 | 2.30M | 6.7G | 1.49ms |
| YOLOv10-s (2024) | 300 | 40.5 | 24.8 | 7.25M | 21.6G | 1.63ms |
| YOLOv10-m (2024) | 300 | 44.2 | 27.6 | 15.36M | 59.1G | 3.08ms |
| YOLO11-n (2024) | 300 | 33.4 | 19.7 | 2.62M | 6.5G | 1.10ms |
| YOLO11-s (2024) | 300 | 40.8 | 24.7 | 9.44M | 21.5G | 1.59ms |
| YOLO11-m (2024) | 300 | 46.4 | 28.9 | 20.09M | 68.0G | 2.76ms |
| FBRT-YOLO-N dagger (2025) | 300 | 34.4 | 20.2 | 0.9M | 6.7G | -- |
| FBRT-YOLO-S dagger (2025) | 300 | 42.4 | 25.9 | 2.9M | 22.9G | -- |
| FBRT-YOLO-M dagger (2025) | 300 | 45.9 | 28.4 | 7.2M | 58.7G | -- |
| LEAF-YOLO-N dagger (2025) | 800 | 39.7 | 21.9 | 1.2M | 5.6G | 2.54ms |
| LEAF-YOLO-S dagger (2025) | 800 | 48.3 | 28.2 | 4.28M | 20.9G | 3.01ms |
| LMFF-MFFE dagger (2025) | 200 | 44.5 | 27.2 | 8.8M | 41.7G | -- |
| RemDet-S dagger (2025) | 300 | 41.5 | 24.7 | 16.0M | 40.5G | 2.38ms |
| RemDet-M dagger (2025) | 300 | 44.7 | 27.3 | 29.5M | 89.7G | 3.86ms |
| RemDet-L dagger (2025) | 300 | 47.4 | 29.3 | 42.3M | 160.4G | 5.30ms |
| RT-DETR-R34 dagger (2024) | 300 | 46.0 | 27.2 | 31.0M | 92.0G | -- |
| RT-DETR-R50 dagger (2024) | 300 | 48.3 | 28.8 | 42.0M | 136.0G | -- |
| MKL-YOLO-N (Ours) | 300 | 40.1 | 24.5 | 1.33M | 10.3G | 2.01ms |
| **MKL-YOLO-S (Ours)** | 300 | **49.3** | **30.8** | 3.40M | 29.6G | 3.43ms |

Daggered entries are reported results from the corresponding publications or their reported comparisons, whereas the remaining entries follow the 300-epoch setting.

### UAVDT

| Method | AP50 (%) | AP50:95 (%) |
|---|---:|---:|
| YOLOv5-s | 50.1 | 29.8 |
| YOLOv8-s | 49.5 | 29.2 |
| YOLOv10-s | 50.5 | 30.7 |
| YOLO11-s | 49.3 | 29.4 |
| **MKL-YOLO-S (Ours)** | **55.4** | **34.9** |

---

## References

This project is implemented on top of Ultralytics YOLO and uses modules inspired by lightweight detection and aerial small-object detection research.

---

## Citation

Citation information will be released after the paper metadata is finalized.

```bibtex
@misc{mkl_yolo,
  title  = {MKL-YOLO},
  author = {},
  year   = {},
}
```

---

## License

This project is licensed under the MIT License.
