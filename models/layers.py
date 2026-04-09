import torch
import torch.nn as nn
class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # Bernoulli mask: 1 with probability (1 - p)
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p))
        # Inverted dropout scaling
        return x * mask / (1.0 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"