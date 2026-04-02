import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Custom IoU Loss for bounding box regression.
    Input format: [x_center, y_center, width, height] in pixel space.
    Loss = 1 - IoU, so range is [0, 1].
    Supports reduction: 'mean' (default), 'sum', 'none'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), \
            f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: shape (N, 4) — [x_center, y_center, width, height]
        """
        # Convert center format to corner format
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * \
            (pred_y2 - pred_y1).clamp(min=0)
        tgt_area = (tgt_x2 - tgt_x1).clamp(min=0) * \
            (tgt_y2 - tgt_y1).clamp(min=0)
        union_area = pred_area + tgt_area - inter_area + 1e-6  # eps for stability

        iou = inter_area / union_area
        loss = 1.0 - iou   # range [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
