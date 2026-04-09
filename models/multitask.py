import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.localization import LocalizationModel
from models.segmentation import SegmentationModel


classifier_path = "checkpoints/classifier.pth"
localizer_path  = "checkpoints/localizer.pth"
unet_path       = "checkpoints/unet.pth"


class MultiTaskPerceptionModel(nn.Module):
    
    def __init__(self, num_classes: int = 37, num_seg_classes: int = 3, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes

        vgg = VGG11(num_classes=num_classes)
        self.backbone = vgg.get_backbone()          
        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))

        from models.layers import CustomDropout
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), CustomDropout(0.5),
            nn.Linear(4096, 4096),        nn.ReLU(inplace=True), CustomDropout(0.5),
            nn.Linear(4096, num_classes),
        )

        self.loc_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(1024, 256),          nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

        from models.segmentation import VGG11Encoder, double_conv, up_block
        import torch.nn.functional as F
        self._F = F

        self.seg_encoder = VGG11Encoder()   

        self.seg_bottleneck = double_conv(512, 1024)
        self.up5, self.dec5 = up_block(1024, 512, 512)
        self.up4, self.dec4 = up_block(512,  512, 256)
        self.up3, self.dec3 = up_block(256,  256, 128)
        self.up2, self.dec2 = up_block(128,  128,  64)
        self.up1, self.dec1 = up_block(64,    64,  64)
        self.seg_out = nn.Conv2d(64, num_seg_classes, kernel_size=1)

    def init(self):
        """Download checkpoints from Google Drive and load weights."""
        import gdown
        gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth drive id>",  output=localizer_path,  quiet=False)
        gdown.download(id="<unet.pth drive id>",       output=unet_path,       quiet=False)

        self._load_classifier(classifier_path)
        self._load_localizer(localizer_path)
        self._load_unet(unet_path)

    def _load_state(self, path: str) -> dict:
        state = torch.load(path, map_location="cpu")
        return state.get("model_state_dict", state)

    def _load_classifier(self, path: str):
        state = self._load_state(path)
        bb = {k.replace("vgg.features.", ""): v for k, v in state.items() if "vgg.features." in k}
        self.backbone.load_state_dict(bb, strict=True)
        cls = {k.replace("vgg.classifier.", ""): v for k, v in state.items() if "vgg.classifier." in k}
        mapped = {}
        vgg_to_head = {"0": "0", "3": "3", "6": "6"}  
        for k, v in cls.items():
            parts = k.split(".", 1)
            if parts[0] in vgg_to_head:
                mapped[f"{vgg_to_head[parts[0]]}.{parts[1]}"] = v
        self.cls_head.load_state_dict(mapped, strict=False)

    def _load_localizer(self, path: str):
        state = self._load_state(path)
        bb = {k.replace("backbone.", ""): v for k, v in state.items() if "backbone." in k}
        self.backbone.load_state_dict(bb, strict=False)  
        reg = {k.replace("reg_head.", ""): v for k, v in state.items() if "reg_head." in k}
        self.loc_head.load_state_dict(reg, strict=False)

    def _load_unet(self, path: str):
        state = self._load_state(path)
        seg_model = SegmentationModel(num_classes=self.num_seg_classes)
        seg_model.load_state_dict(state, strict=False)
        decoder_modules = [
            "seg_bottleneck", "up5", "dec5", "up4", "dec4",
            "up3", "dec3", "up2", "dec2", "up1", "dec1", "seg_out"
        ]
        src_map = {
            "seg_bottleneck": seg_model.bottleneck,
            "up5": seg_model.up5, "dec5": seg_model.dec5,
            "up4": seg_model.up4, "dec4": seg_model.dec4,
            "up3": seg_model.up3, "dec3": seg_model.dec3,
            "up2": seg_model.up2, "dec2": seg_model.dec2,
            "up1": seg_model.up1, "dec1": seg_model.dec1,
            "seg_out": seg_model.out_conv,
        }
        for attr, src in src_map.items():
            dst = getattr(self, attr)
            dst.load_state_dict(src.state_dict(), strict=True)
        # Encoder backbone
        self.seg_encoder.load_state_dict(seg_model.encoder.state_dict(), strict=True)
        # Also sync shared backbone from seg encoder block weights
        self.seg_encoder.block1.load_state_dict(
            {k: v for k, v in seg_model.encoder.block1.state_dict().items()}, strict=True
        )

    def _seg_forward(self, x: torch.Tensor) -> torch.Tensor:
        F = self._F
        bottleneck, skips = self.seg_encoder(x)
        s1, s2, s3, s4, s5 = skips

        def cat_skip(up, skip):
            if up.shape != skip.shape:
                up = F.interpolate(up, size=skip.shape[2:], mode="bilinear", align_corners=False)
            return torch.cat([up, skip], dim=1)

        t = self.seg_bottleneck(bottleneck)
        t = self.dec5(cat_skip(self.up5(t), s5))
        t = self.dec4(cat_skip(self.up4(t), s4))
        t = self.dec3(cat_skip(self.up3(t), s3))
        t = self.dec2(cat_skip(self.up2(t), s2))
        t = self.dec1(cat_skip(self.up1(t), s1))
        return self.seg_out(t)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat_pool = torch.flatten(self.avgpool(feat), 1)

        cls_logits = self.cls_head(feat_pool)

        raw = self.loc_head(feat_pool)
        cx = torch.sigmoid(raw[:, 0]) * self.img_size
        cy = torch.sigmoid(raw[:, 1]) * self.img_size
        w  = torch.nn.functional.softplus(raw[:, 2]) * self.img_size
        h  = torch.nn.functional.softplus(raw[:, 3]) * self.img_size
        bbox = torch.stack([cx, cy, w, h], dim=1)

        seg_logits = self._seg_forward(x)

        return cls_logits, bbox, seg_logits