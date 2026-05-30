from pathlib import Path

import pytest


def test_mkl_modules_importable():
    from mkl_yolo import C2f_E, C2f_G, ConvStem, GhostBottleneckV2, VisDroneDCNDetect

    assert ConvStem
    assert GhostBottleneckV2
    assert C2f_E
    assert C2f_G
    assert VisDroneDCNDetect


@pytest.mark.slow
def test_model_config_builds():
    import mkl_yolo.register  # noqa: F401
    from ultralytics import YOLO

    model = YOLO(str(Path("configs/model/mkl_yolov8s.yaml")))
    assert model.model is not None
