import torch
import torch.nn as nn
 
 
def boxes_to_corners(boxes: torch.Tensor) -> tuple:
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return x1, y1, x2, y2
 
 
def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px1, py1, px2, py2 = boxes_to_corners(pred)
    tx1, ty1, tx2, ty2 = boxes_to_corners(target)
 
    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
 
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
 
    pred_area = (pred[:, 2] * pred[:, 3]).clamp(min=0)
    target_area = (target[:, 2] * target[:, 3]).clamp(min=0)
    union_area = pred_area + target_area - inter_area + 1e-6
 
    return (inter_area / union_area).clamp(0.0, 1.0)
 
 
class IoULoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        self.reduction = reduction
 
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        iou = compute_iou(pred, target)
        loss = 1.0 - iou  
 
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss