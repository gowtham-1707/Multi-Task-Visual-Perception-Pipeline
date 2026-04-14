
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = [
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
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root: str, split: str = "trainval", transform=None, mask_transform=None, require_bbox=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.require_bbox = require_bbox
        self.image_dir = os.path.join(root, "images")
        self.xml_dir = os.path.join(root, "annotations", "xmls")
        self.mask_dir = os.path.join(root, "annotations", "trimaps")
        self.samples = self._parse_split()

    def _parse_split(self):
        list_file = os.path.join(self.root, "annotations", "list.txt")
        all_samples = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name = parts[0]
                class_id = int(parts[1]) - 1
                img_path = os.path.join(self.image_dir, f"{name}.jpg")
                mask_path = os.path.join(self.mask_dir, f"{name}.png")
                xml_path = os.path.join(self.xml_dir, f"{name}.xml")
                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    continue
                if self.require_bbox and not os.path.exists(xml_path):
                    continue
                all_samples.append((name, class_id))

        import random
        rng = random.Random(42)
        all_samples_sorted = sorted(all_samples)
        rng.shuffle(all_samples_sorted)
        split_idx = int(0.85 * len(all_samples_sorted))

        if self.split in ["trainval"]:
            return all_samples_sorted[:split_idx]
        else:
            return all_samples_sorted[split_idx:]

    def _load_bbox(self, name):
        xml_path = os.path.join(self.xml_dir, f"{name}.xml")
        if not os.path.exists(xml_path):
            return None
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        obj = root.find("object")
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        bw = xmax - xmin
        bh = ymax - ymin
        return [cx, cy, bw, bh], w, h

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, class_id = self.samples[idx]
        img_path = os.path.join(self.image_dir, f"{name}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        mask = np.array(Image.open(mask_path))
        mask = (mask - 1).clip(0, 2).astype(np.int64)

        bbox_data = self._load_bbox(name)
        if bbox_data is not None:
            bbox, orig_w, orig_h = bbox_data
        else:
            h, w = image.shape[:2]
            bbox = [w / 2.0, h / 2.0, float(w), float(h)]

        if self.transform is not None:
            h, w = image.shape[:2]
            bx = max(0.0, bbox[0] - bbox[2] / 2)
            by = max(0.0, bbox[1] - bbox[3] / 2)
            bw = min(bbox[2], w - bx)
            bh = min(bbox[3], h - by)
            transformed = self.transform(
                image=image, mask=mask,
                bboxes=[[bx, by, bw, bh]], bbox_labels=[0],
            )
            image = transformed["image"]
            mask = transformed["mask"].long()
            boxes = transformed["bboxes"]
            if len(boxes) > 0:
                bx2, by2, bw2, bh2 = boxes[0]
                bbox = [bx2 + bw2 / 2, by2 + bh2 / 2, bw2, bh2]
            else:
                bbox = [112.0, 112.0, 224.0, 224.0]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        label_tensor = torch.tensor(class_id, dtype=torch.long)
        return image, label_tensor, bbox_tensor, mask
