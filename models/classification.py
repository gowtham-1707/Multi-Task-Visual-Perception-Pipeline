import torch
import torch.nn as nn
from models.vgg11 import VGG11
class ClassificationModel(nn.Module):

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.vgg = VGG11(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x)

    def get_backbone(self) -> nn.Sequential:
        return self.vgg.get_backbone()

    def get_avgpool(self) -> nn.Module:
        return self.vgg.avgpool