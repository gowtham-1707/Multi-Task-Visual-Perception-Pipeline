
import os
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout, SigmoidBBox
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):
    
    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        image_size: float = 224.0,
        dropout_p: float = 0.5,
        classifier_path: str = "1CACv7TeUxX5kmeSdQzmV8z5UolojimMw",
        localizer_path: str = "1Yx2RCOTdnDJm0hEzlEZgx31sazQWogQn",
        unet_path: str = "1Dn9C_Gn0XHmulrpT_cXTkOp_cX1MmutX",
    ):
        super().__init__()
        self.backbone     = VGG11Encoder(in_channels=in_channels)
        self.seg_backbone = VGG11Encoder(in_channels=in_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds),
        )
        self.loc_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024), nn.ReLU(inplace=True), CustomDropout(p=dropout_p),
            nn.Linear(1024, 4), SigmoidBBox(scale=image_size),
        )
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
        self.seg_dropout = CustomDropout(p=dropout_p)
        self.seg_head    = nn.Conv2d(32, seg_classes, kernel_size=1)
        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        device = torch.device("cpu")
        if os.path.exists(unet_path):
            unet = VGG11UNet()
            unet.load_state_dict(torch.load(unet_path, map_location=device))
            self.seg_backbone.load_state_dict(unet.encoder.state_dict())
            self.bottleneck.load_state_dict(unet.bottleneck.state_dict())
            self.up5.load_state_dict(unet.up5.state_dict())
            self.dec5.load_state_dict(unet.dec5.state_dict())
            self.up4.load_state_dict(unet.up4.state_dict())
            self.dec4.load_state_dict(unet.dec4.state_dict())
            self.up3.load_state_dict(unet.up3.state_dict())
            self.dec3.load_state_dict(unet.dec3.state_dict())
            self.up2.load_state_dict(unet.up2.state_dict())
            self.dec2.load_state_dict(unet.dec2.state_dict())
            self.up1.load_state_dict(unet.up1.state_dict())
            self.dec1.load_state_dict(unet.dec1.state_dict())
            self.seg_head.load_state_dict(unet.head.state_dict())
            print(f"[MultiTask] Loaded U-Net weights from {unet_path}")
        if os.path.exists(classifier_path):
            clf = VGG11Classifier()
            clf.load_state_dict(torch.load(classifier_path, map_location=device))
            self.backbone.load_state_dict(clf.encoder.state_dict())
            self.cls_head.load_state_dict(clf.classifier.state_dict())
            print(f"[MultiTask] Loaded classifier weights from {classifier_path}")
        if os.path.exists(localizer_path):
            loc = VGG11Localizer()
            loc.load_state_dict(torch.load(localizer_path, map_location=device))
            self.loc_head.load_state_dict(loc.regressor.state_dict())
            print(f"[MultiTask] Loaded localizer weights from {localizer_path}")

    def forward(self, x):
        bottleneck, _ = self.backbone(x, return_features=True)
        pooled = self.adaptive_pool(bottleneck)
        flat = torch.flatten(pooled, 1)
        cls_out = self.cls_head(flat)
        loc_out = self.loc_head(flat)
        seg_bottleneck, features = self.seg_backbone(x, return_features=True)
        s = self.bottleneck(seg_bottleneck)
        s = self.up5(s); s = torch.cat([s, features["f5"]], dim=1); s = self.dec5(s)
        s = self.up4(s); s = torch.cat([s, features["f4"]], dim=1); s = self.dec4(s)
        s = self.up3(s); s = torch.cat([s, features["f3"]], dim=1); s = self.dec3(s)
        s = self.up2(s); s = torch.cat([s, features["f2"]], dim=1); s = self.dec2(s)
        s = self.up1(s); s = torch.cat([s, features["f1"]], dim=1); s = self.dec1(s)
        s = self.seg_dropout(s)
        seg_out = self.seg_head(s)
        return {"classification": cls_out, "localization": loc_out, "segmentation": seg_out}
