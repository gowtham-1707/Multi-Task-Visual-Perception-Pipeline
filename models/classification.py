
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """VGG11 encoder + fully connected classification head (37 breeds)."""
    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, return_features=False)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
