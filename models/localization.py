import torch
import torch.nn as nn
from models.vgg11 import VGG11
class LocalizationModel(nn.Module):
    def __init__(self, freeze_early: bool = True, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        vgg = VGG11(num_classes=37)
        self.backbone = vgg.get_backbone()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if freeze_early:
            self._freeze_early_blocks()
        self.reg_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )
        self._init_head()

    def _freeze_early_blocks(self):
        frozen = 0
        for module in self.backbone:
            if frozen >= 7:
                break
            for param in module.parameters():
                param.requires_grad = False
            frozen += 1

    def _init_head(self):
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def load_backbone_from_classifier(self, classifier_path: str, device: str = "cpu"):
        state = torch.load(classifier_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        backbone_state = {
            k.replace("vgg.features.", ""): v
            for k, v in state.items()
            if k.startswith("vgg.features.")
        }
        self.backbone.load_state_dict(backbone_state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        out = self.reg_head(feat)
        cx = torch.sigmoid(out[:, 0]) * self.img_size
        cy = torch.sigmoid(out[:, 1]) * self.img_size
        w = torch.nn.functional.softplus(out[:, 2]) * self.img_size
        h = torch.nn.functional.softplus(out[:, 3]) * self.img_size
        return torch.stack([cx, cy, w, h], dim=1)