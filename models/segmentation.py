

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg11 import VGG11

def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def up_block(in_ch: int, skip_ch: int, out_ch: int) -> tuple:
    
    upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
    conv = double_conv(in_ch // 2 + skip_ch, out_ch)
    return upsample, conv
class VGG11Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        vgg = VGG11(num_classes=37)
        feats = list(vgg.features.children())
        blocks, current = [], []
        for layer in feats:
            if isinstance(layer, nn.MaxPool2d):
                blocks.append(nn.Sequential(*current))
                current = []
            else:
                current.append(layer)
        if current:
            blocks.append(nn.Sequential(*current))

        assert len(blocks) == 5, f"Expected 5 encoder blocks, got {len(blocks)}"
        self.block1 = blocks[0] 
        self.block2 = blocks[1]   
        self.block3 = blocks[2]   
        self.block4 = blocks[3]   
        self.block5 = blocks[4]   
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        s1 = self.block1(x)
        s2 = self.block2(self.pool(s1))
        s3 = self.block3(self.pool(s2))
        s4 = self.block4(self.pool(s3))
        s5 = self.block5(self.pool(s4))
        bottleneck = self.pool(s5)
        return bottleneck, [s1, s2, s3, s4, s5]

    def load_from_vgg11(self, state_dict: dict):
        backbone_state = {
            k.replace("vgg.features.", ""): v
            for k, v in state_dict.items()
            if k.startswith("vgg.features.")
        }
        vgg_feats = VGG11(num_classes=37).features
        vgg_feats.load_state_dict(backbone_state, strict=True)
        feats = list(vgg_feats.children())
        blocks, current = [], []
        for layer in feats:
            if isinstance(layer, nn.MaxPool2d):
                blocks.append(nn.Sequential(*current))
                current = []
            else:
                current.append(layer)
        if current:
            blocks.append(nn.Sequential(*current))
        for dst, src in zip(
            [self.block1, self.block2, self.block3, self.block4, self.block5], blocks
        ):
            dst.load_state_dict(src.state_dict(), strict=True)

class SegmentationModel(nn.Module):
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.encoder = VGG11Encoder()
        self.bottleneck = double_conv(512, 1024)
        self.up5, self.dec5 = up_block(1024, 512, 512)
        self.up4, self.dec4 = up_block(512, 512, 256)
        self.up3, self.dec3 = up_block(256, 256, 128)
        self.up2, self.dec2 = up_block(128, 128, 64)
        self.up1, self.dec1 = up_block(64, 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _cat_skip(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape != skip.shape:
            up = F.interpolate(up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder(x)
        s1, s2, s3, s4, s5 = skips

        x = self.bottleneck(bottleneck)

        x = self._cat_skip(self.up5(x), s5)
        x = self.dec5(x)
        x = self._cat_skip(self.up4(x), s4)
        x = self.dec4(x)
        x = self._cat_skip(self.up3(x), s3)
        x = self.dec3(x)
        x = self._cat_skip(self.up2(x), s2)
        x = self.dec2(x)
        x = self._cat_skip(self.up1(x), s1)
        x = self.dec1(x)

        return self.out_conv(x)

    def load_encoder_from_classifier(self, classifier_path: str, device: str = "cpu"):
        state = torch.load(classifier_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.encoder.load_from_vgg11(state)