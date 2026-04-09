import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as tvd
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.0),
    )


def get_val_transform(img_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.0),
    )
class PetsDataset(Dataset):
    CLASSES = [
        "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
        "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
        "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
        "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
        "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
        "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
        "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier",
    ]

    def __init__(
        self,
        root: str = "data/pets",
        split: str = "train",
        task: str = "all",
        transform=None,
        img_size: int = 224,
    ):
        self.task      = task
        self.img_size  = img_size
        self.transform = transform or get_val_transform(img_size)
        tv_split = "test" if split == "test" else "trainval"
        self._base = tvd.OxfordIIITPet(
            root=root,
            split=tv_split,
            target_types=["category", "segmentation"],
            download=True,
        )

        indices = list(range(len(self._base)))
        if split in ("train", "val"):
            cut = int(0.9 * len(indices))
            self.indices = indices[:cut] if split == "train" else indices[cut:]
        else:
            self.indices = indices
        self.bbox_dir = os.path.join(root, "oxford-iiit-pet", "annotations", "xmls")

    def _load_bbox(self, name: str):
        import xml.etree.ElementTree as ET
        xml_path = os.path.join(self.bbox_dir, f"{name}.xml")
        if not os.path.exists(xml_path):
            return None
        try:
            root = ET.parse(xml_path).getroot()
            obj  = root.find("object/bndbox")
            xmin = float(obj.find("xmin").text)
            ymin = float(obj.find("ymin").text)
            xmax = float(obj.find("xmax").text)
            ymax = float(obj.find("ymax").text)
            return [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]
        pil_img, (label, pil_mask) = self._base[real_idx]

        img_path = self._base._images[real_idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        image = np.array(pil_img.convert("RGB"))

        mask = None
        if self.task in ("segment", "all"):
            mask = (np.array(pil_mask).astype(np.int32) - 1).clip(0, 2).astype(np.uint8)

        bbox_raw = self._load_bbox(name)
        bboxes, bbox_labels = [], [int(label)]
        if bbox_raw is not None:
            cx, cy, bw, bh = bbox_raw
            orig_h, orig_w = image.shape[:2]
            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            bw = min(bw, orig_w - x1)
            bh = min(bh, orig_h - y1)
            if bw > 1 and bh > 1:
                bboxes = [[x1, y1, bw, bh]]

        if mask is not None:
            aug = self.transform(image=image, mask=mask, bboxes=bboxes, class_labels=bbox_labels)
            raw_mask = aug["mask"]
            
            if isinstance(raw_mask, torch.Tensor):
                mask_t = raw_mask.long()
            else:
                mask_t = torch.from_numpy(np.array(raw_mask, dtype=np.int64))
        else:
            aug = self.transform(image=image, bboxes=bboxes, class_labels=bbox_labels)
            mask_t = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
 
        image_t = aug["image"]
        label_t = torch.tensor(int(label), dtype=torch.long)
 
        if aug["bboxes"]:
            x1, y1, bw, bh = aug["bboxes"][0]
            bbox_t = torch.tensor([x1 + bw / 2, y1 + bh / 2, bw, bh], dtype=torch.float32)
        else:
            bbox_t = torch.zeros(4, dtype=torch.float32)
 
        return {
            "image": image_t,
            "label": label_t,
            "bbox":  bbox_t,
            "mask":  mask_t,
            "name":  name,
        }