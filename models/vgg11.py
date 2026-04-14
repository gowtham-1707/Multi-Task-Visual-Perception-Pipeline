
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        def conv_bn_relu(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.block1 = nn.Sequential(conv_bn_relu(in_channels, 64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = nn.Sequential(conv_bn_relu(64, 128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = nn.Sequential(conv_bn_relu(128, 256), conv_bn_relu(256, 256))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = nn.Sequential(conv_bn_relu(256, 512), conv_bn_relu(512, 512))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = nn.Sequential(conv_bn_relu(512, 512), conv_bn_relu(512, 512))
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @property
    def last_conv(self):
        return self.block5[1][0]

    def forward(self, x, return_features=False):
        f1 = self.block1(x);  x = self.pool1(f1)
        f2 = self.block2(x);  x = self.pool2(f2)
        f3 = self.block3(x);  x = self.pool3(f3)
        f4 = self.block4(x);  x = self.pool4(f4)
        f5 = self.block5(x);  bottleneck = self.pool5(f5)
        if return_features:
            return bottleneck, {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}
        return bottleneck
