"""Dataset-specific LDC-YOLO training with WIoU regression."""
from .register import register_ldc_yolo
from .loss import WIoUBboxLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8DetectionLoss

register_ldc_yolo()

class LDCDetectionModel(DetectionModel):
    def init_criterion(self):
        criterion = v8DetectionLoss(self)
        criterion.bbox_loss = WIoUBboxLoss(self.model[-1].reg_max).to(next(self.parameters()).device)
        return criterion

class LDCDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = LDCDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
