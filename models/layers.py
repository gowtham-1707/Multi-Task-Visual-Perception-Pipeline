
import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    """Inverted dropout implemented without torch.nn.Dropout."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = torch.empty_like(x).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)


class SigmoidBBox(nn.Module):
    """
    Final activation for bounding box regression heads.
    Sigmoid maps logits to (0,1) and multiplying by scale maps to (0, image_size).
    """
    def __init__(self, scale: float = 224.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale
