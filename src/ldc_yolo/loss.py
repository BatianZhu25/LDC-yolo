"""WIoU-v1 box regression with the upstream DFL term retained."""
import torch
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou

def wiou_v1(pred, target):
    """Distance-attended IoU loss for paired xyxy boxes."""
    iou = bbox_iou(pred, target, xywh=False, CIoU=False)
    center_distance = (((pred[..., :2] + pred[..., 2:]) -
                        (target[..., :2] + target[..., 2:])) / 2).square().sum(-1, keepdim=True)
    enclosing = torch.maximum(pred[..., 2:], target[..., 2:]) - torch.minimum(pred[..., :2], target[..., :2])
    diagonal = enclosing.square().sum(-1, keepdim=True).detach().clamp_min(1e-7)
    return torch.exp(center_distance / diagonal) * (1 - iou)

class WIoUBboxLoss(BboxLoss):
    """Replace CIoU regression with WIoU-v1; retain task-aligned weights and DFL."""
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        _, loss_dfl = super().forward(pred_dist, pred_bboxes, anchor_points,
                                     target_bboxes, target_scores, target_scores_sum, fg_mask)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        loss_box = (wiou_v1(pred_bboxes[fg_mask], target_bboxes[fg_mask]) * weight).sum() / target_scores_sum
        return loss_box, loss_dfl
