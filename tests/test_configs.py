from pathlib import Path

import yaml


def test_model_config_contains_ldc_modules():
    cfg_path = Path("configs/model/ldc_yolov8s.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    layers = cfg["backbone"] + cfg["head"]
    module_names = {layer[2] for layer in layers}
    assert {"DLStem", "GhostBottleneckV2", "C2fLGDI", "C2fRPMK", "VisDroneDCNDetect"} <= module_names


def test_dataset_configs_have_names():
    for cfg_path in (Path("configs/data/visdrone.yaml"), Path("configs/data/uavdt.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        assert "path" in cfg
        assert isinstance(cfg["names"], dict)
        assert cfg["names"]
