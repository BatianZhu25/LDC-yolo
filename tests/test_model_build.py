from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

@pytest.mark.slow
@pytest.mark.parametrize("scale", ["n", "s"])
def test_model_config_builds_and_predicts(scale):
    import ldc_yolo.register  # noqa: F401
    from ultralytics import YOLO
    model = YOLO(str(Path(f"configs/model/ldc_yolov8{scale}.yaml")))
    model.model.eval()
    assert model.model.stride.tolist() == [4, 8, 16]
    with torch.no_grad():
        predictions, heads = model.model(torch.randn(1, 3, 64, 64))
    assert predictions.shape == (1, 14, 336)
    assert [head.shape[-1] for head in heads] == [16, 8, 4]
    assert torch.isfinite(predictions).all()
    assert model.model.yaml["scale"] == scale


def test_training_criterion_with_positive_targets():
    from ldc_yolo.trainer import LDCDetectionModel
    from ldc_yolo.loss import WIoUBboxLoss
    from ultralytics.cfg import DEFAULT_CFG_DICT
    model = LDCDetectionModel("configs/model/ldc_yolov8n.yaml", nc=3, verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    criterion = model.init_criterion()
    assert isinstance(criterion.bbox_loss, WIoUBboxLoss)
    batch = {"img": torch.randn(2, 3, 64, 64), "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.tensor([[0.], [2.]]), "bboxes": torch.tensor([[.5, .5, .25, .25], [.4, .4, .3, .3]])}
    loss, parts = criterion(model(batch["img"]), batch)
    loss.backward()
    assert torch.isfinite(loss) and (parts > 0).all()
    assert model.model[0].conv1.conv.weight.grad is not None
