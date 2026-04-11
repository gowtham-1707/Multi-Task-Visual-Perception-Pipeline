import torch
import torch.nn as nn
from models.layers import CustomDropout

VGG11_CONFIG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

def make_conv_layers(cfg: list, batch_norm: bool = True) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers) 
class VGG11(nn.Module):

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.4, batch_norm: bool = True):
        super().__init__()
        self.features  = make_conv_layers(VGG11_CONFIG, batch_norm=batch_norm)
        self.avgpool   = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
 
    def get_backbone(self) -> nn.Sequential:
        return self.features