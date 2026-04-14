
"""Training script for classification, localization, and segmentation tasks."""
import argparse, os
import albumentations as A
import numpy as np
import torch, torch.nn as nn
import wandb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_transforms(train=True):
    if train:
        return A.Compose(
            [A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.HorizontalFlip(p=0.5),
             A.ColorJitter(p=0.3), A.Normalize(mean=MEAN, std=STD), ToTensorV2()],
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"], min_visibility=0.3),
        )
    return A.Compose(
        [A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"], min_visibility=0.0),
    )

def dice_score(pred_mask, true_mask, num_classes=3, eps=1e-6):
    pred_mask = pred_mask.argmax(dim=1)
    scores = []
    for c in range(num_classes):
        p = (pred_mask == c).float(); t = (true_mask == c).float()
        scores.append((2*(p*t).sum()+eps)/(p.sum()+t.sum()+eps))
    return torch.stack(scores).mean().item()

# ... (full implementations of train_classifier, train_localizer, train_unet are in the notebook)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["classify","localize","segment"])
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--finetune_strategy", type=str, default="full", choices=["frozen","partial","full"])
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Task: {args.task} | Run the Kaggle notebook for full training with W&B logging.")
