
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    
    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.bottleneck = double_conv(512, 1024)
        self.up5  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = double_conv(512 + 512, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(256 + 512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(128 + 256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(64 + 128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = double_conv(32 + 64, 32)
        self.dropout = CustomDropout(p=dropout_p)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, features = self.encoder(x, return_features=True)
        x = self.bottleneck(bottleneck)
        x = self.up5(x); x = torch.cat([x, features["f5"]], dim=1); x = self.dec5(x)
        x = self.up4(x); x = torch.cat([x, features["f4"]], dim=1); x = self.dec4(x)
        x = self.up3(x); x = torch.cat([x, features["f3"]], dim=1); x = self.dec3(x)
        x = self.up2(x); x = torch.cat([x, features["f2"]], dim=1); x = self.dec2(x)
        x = self.up1(x); x = torch.cat([x, features["f1"]], dim=1); x = self.dec1(x)
        x = self.dropout(x)
        return self.head(x)
