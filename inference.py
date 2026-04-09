import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import PetsDataset

CLASSES = PetsDataset.CLASSES
SEG_COLOURS = [(0, 128, 0), (255, 0, 0), (0, 0, 255)]  # fg / bg / boundary


def preprocess(img_path: str, img_size: int = 224) -> torch.Tensor:
    tfm = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img = np.array(Image.open(img_path).convert("RGB"))
    return tfm(image=img)["image"].unsqueeze(0)


def draw_bbox(img: Image.Image, bbox: list, label: str) -> Image.Image:
    cx, cy, w, h = bbox
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1 - 12), label, fill="red")
    return img


def mask_to_colour(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, col in enumerate(SEG_COLOURS):
        colour[mask == cls_id] = col
    return Image.fromarray(colour)


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskPerceptionModel().to(device)
    model.init()
    model.eval()

    for img_path in args.images:
        tensor = preprocess(img_path, args.img_size).to(device)
        cls_logits, bbox, seg_logits = model(tensor)

        cls_id   = cls_logits.argmax(1).item()
        cls_name = CLASSES[cls_id]
        bbox_px  = bbox[0].cpu().tolist()
        seg_mask = seg_logits[0].argmax(0).cpu().numpy()

        print(f"\n{img_path}")
        print(f"  Class: {cls_name} ({cls_id})")
        print(f"  BBox:  cx={bbox_px[0]:.1f} cy={bbox_px[1]:.1f} w={bbox_px[2]:.1f} h={bbox_px[3]:.1f}")

        # Visualise
        orig = Image.open(img_path).convert("RGB").resize((args.img_size, args.img_size))
        annotated = draw_bbox(orig.copy(), bbox_px, cls_name)
        seg_vis   = mask_to_colour(seg_mask)

        out_stem = img_path.rsplit(".", 1)[0]
        annotated.save(f"{out_stem}_detection.jpg")
        seg_vis.save(f"{out_stem}_segmentation.png")
        print(f"  Saved: {out_stem}_detection.jpg, {out_stem}_segmentation.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",   nargs="+", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()