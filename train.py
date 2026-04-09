"""
train.py — Train classifier, localizer, or segmentation model.

Usage:
    python train.py --task classify  --epochs 30 --lr 1e-3
    python train.py --task localize  --epochs 30 --lr 1e-4
    python train.py --task segment   --epochs 30 --lr 1e-4
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import PetsDataset, get_train_transform, get_val_transform
from models.classification import ClassificationModel
from models.localization import LocalizationModel
from models.segmentation import SegmentationModel
from losses.iou_loss import IoULoss



def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3, smooth: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for multi-class segmentation."""
    probs = torch.softmax(pred_logits, dim=1)  # (N, C, H, W)
    target_one_hot = torch.zeros_like(probs)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    cardinality   = (probs + target_one_hot).sum(dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def seg_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CrossEntropy + Dice for segmentation."""
    ce = nn.CrossEntropyLoss()(pred, target)
    dc = dice_loss(pred, target)
    return ce + dc


def loc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE + IoU loss for localization."""
    mse = nn.MSELoss()(pred, target)
    iou = IoULoss(reduction="mean")(pred, target)
    return mse + iou

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.argmax(1) == target).float().mean().item()


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3, smooth: float = 1e-6) -> float:
    probs = torch.softmax(pred_logits, dim=1)
    target_one_hot = torch.zeros_like(probs)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    cardinality   = (probs + target_one_hot).sum(dims)
    return ((2.0 * intersection + smooth) / (cardinality + smooth)).mean().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device, task):
    model.train()
    total_loss, total_metric = 0.0, 0.0
    for batch in loader:
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()
        if task == "classify":
            out  = model(imgs)
            loss = nn.CrossEntropyLoss()(out, labels)
            metric = accuracy(out, labels)
        elif task == "localize":
            out  = model(imgs)
            loss = loc_loss(out, bboxes)
            metric = IoULoss(reduction="mean")(out, bboxes).item()
        else:  # segment
            out  = model(imgs)
            loss = seg_loss(out, masks)
            metric = dice_score(out, masks)

        loss.backward()
        optimizer.step()
        total_loss   += loss.item()
        total_metric += metric

    n = len(loader)
    return total_loss / n, total_metric / n


@torch.no_grad()
def validate(model, loader, loss_fn, device, task):
    model.eval()
    total_loss, total_metric = 0.0, 0.0
    for batch in loader:
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks  = batch["mask"].to(device)

        if task == "classify":
            out  = model(imgs)
            loss = nn.CrossEntropyLoss()(out, labels)
            metric = accuracy(out, labels)
        elif task == "localize":
            out  = model(imgs)
            loss = loc_loss(out, bboxes)
            metric = 1.0 - IoULoss(reduction="mean")(out, bboxes).item()  # report IoU
        else:
            out  = model(imgs)
            loss = seg_loss(out, masks)
            metric = dice_score(out, masks)

        total_loss   += loss.item()
        total_metric += metric

    n = len(loader)
    return total_loss / n, total_metric / n


def build_model(task: str, device: str, classifier_ckpt: str = None):
    if task == "classify":
        return ClassificationModel().to(device)
    elif task == "localize":
        model = LocalizationModel(freeze_early=True).to(device)
        if classifier_ckpt and os.path.exists(classifier_ckpt):
            model.load_backbone_from_classifier(classifier_ckpt, device)
            print("Loaded backbone from classifier checkpoint.")
        return model
    else:
        model = SegmentationModel().to(device)
        if classifier_ckpt and os.path.exists(classifier_ckpt):
            model.load_encoder_from_classifier(classifier_ckpt, device)
            print("Loaded encoder from classifier checkpoint.")
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",      type=str,   default="classify", choices=["classify", "localize", "segment"])
    parser.add_argument("--data_root", type=str,   default="data/pets")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--img_size",  type=int,   default=224)
    parser.add_argument("--classifier_ckpt", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--wandb_project",   type=str, default="da6401_assignment2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)

    wandb.init(project=args.wandb_project, config=vars(args), name=f"{args.task}_run")

    train_ds = PetsDataset(args.data_root, "train", "all", get_train_transform(args.img_size), args.img_size)
    val_ds   = PetsDataset(args.data_root, "val",   "all", get_val_transform(args.img_size),   args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model     = build_model(args.task, device, args.classifier_ckpt)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_metric = -1.0
    task_to_ckpt = {"classify": "classifier", "localize": "localizer", "segment": "unet"}
    ckpt_path = f"checkpoints/{task_to_ckpt[args.task]}.pth"

    metric_name = {"classify": "acc", "localize": "IoU", "segment": "dice"}[args.task]

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_metric = train_one_epoch(model, train_loader, optimizer, None, device, args.task)
        val_loss, val_metric = validate(model, val_loader, None, device, args.task)
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            f"train/loss": tr_loss,  f"train/{metric_name}": tr_metric,
            f"val/loss":   val_loss, f"val/{metric_name}":   val_metric,
            "lr": scheduler.get_last_lr()[0],
        })

        print(f"[{epoch:03d}/{args.epochs}] loss {tr_loss:.4f}/{val_loss:.4f}  {metric_name} {tr_metric:.4f}/{val_metric:.4f}")

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"  saved checkpoint (val {metric_name}={val_metric:.4f})")

    wandb.finish()
    print("Done. Best val metric:", best_val_metric)


if __name__ == "__main__":
    main()