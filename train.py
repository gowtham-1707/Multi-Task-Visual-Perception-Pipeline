
import os
import platform
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


def dice_loss(pred_logits, target, smooth=1e-6):
    probs = torch.softmax(pred_logits, dim=1)
    target_one_hot = torch.zeros_like(probs)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    cardinality   = (probs + target_one_hot).sum(dims)
    return (1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)).mean()

def seg_loss(pred, target):
    return nn.CrossEntropyLoss()(pred, target) + dice_loss(pred, target)

def loc_loss(pred, target):
    return nn.MSELoss()(pred, target) + IoULoss(reduction="mean")(pred, target)

def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

def mean_iou(pred, target):
    from losses.iou_loss import compute_iou
    return compute_iou(pred, target).mean().item()

def dice_score(pred_logits, target, smooth=1e-6):
    probs = torch.softmax(pred_logits, dim=1)
    target_one_hot = torch.zeros_like(probs)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    cardinality   = (probs + target_one_hot).sum(dims)
    return ((2.0 * intersection + smooth) / (cardinality + smooth)).mean().item()


def run_epoch(model, loader, optimizer, device, task, training):
    model.train() if training else model.eval()
    total_loss, total_metric = 0.0, 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)

            if training:
                optimizer.zero_grad()

            if task == "classify":
                out    = model(imgs)
                loss   = nn.CrossEntropyLoss()(out, labels)
                metric = accuracy(out, labels)
            elif task == "localize":
                out    = model(imgs)
                loss   = loc_loss(out, bboxes)
                metric = mean_iou(out, bboxes)
            else:
                out    = model(imgs)
                loss   = seg_loss(out, masks)
                metric = dice_score(out, masks)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss   += loss.item()
            total_metric += metric

    n = len(loader)
    return total_loss / n, total_metric / n


def build_model(task, device, classifier_ckpt=None):
    if task == "classify":
        return ClassificationModel().to(device)
    elif task == "localize":
        model = LocalizationModel(freeze_early=True).to(device)
        if classifier_ckpt and os.path.exists(classifier_ckpt):
            model.load_backbone_from_classifier(classifier_ckpt, device)
            print("  Loaded backbone from classifier checkpoint.")
        return model
    else:
        model = SegmentationModel().to(device)
        if classifier_ckpt and os.path.exists(classifier_ckpt):
            model.load_encoder_from_classifier(classifier_ckpt, device)
            print("  Loaded encoder from classifier checkpoint.")
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",      type=str,   default="classify",
                        choices=["classify", "localize", "segment"])
    parser.add_argument("--data_root", type=str,   default="data/pets")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--img_size",  type=int,   default=224)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--classifier_ckpt", type=str,
                        default="checkpoints/classifier.pth")
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs("checkpoints", exist_ok=True)

    num_workers = 0 if platform.system() == "Windows" else 2

    wandb.init(project=args.wandb_project, config=vars(args),
               name=f"{args.task}_run")

    train_ds = PetsDataset(args.data_root, "train", "all",
                           get_train_transform(args.img_size), args.img_size)
    val_ds   = PetsDataset(args.data_root, "val",   "all",
                           get_val_transform(args.img_size),   args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=num_workers, pin_memory=(device=="cuda"))

    model = build_model(args.task, device, args.classifier_ckpt)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    task_to_ckpt = {"classify": "classifier", "localize": "localizer", "segment": "unet"}
    ckpt_path    = f"checkpoints/{task_to_ckpt[args.task]}.pth"
    metric_name  = {"classify": "acc", "localize": "IoU", "segment": "dice"}[args.task]
    best_val     = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss,  tr_metric  = run_epoch(model, train_loader, optimizer, device, args.task, True)
        val_loss, val_metric = run_epoch(model, val_loader,   optimizer, device, args.task, False)
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            f"train/loss":          tr_loss,
            f"train/{metric_name}": tr_metric,
            f"val/loss":            val_loss,
            f"val/{metric_name}":   val_metric,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"[{epoch:03d}/{args.epochs}]  "
              f"loss {tr_loss:.4f}/{val_loss:.4f}  "
              f"{metric_name} {tr_metric:.4f}/{val_metric:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_metric > best_val:
            best_val = val_metric
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f" Saved checkpoint  (val {metric_name} = {val_metric:.4f})")

    wandb.finish()
    print(f"\nDone. Best val {metric_name}: {best_val:.4f}")


if __name__ == "__main__":
    main()