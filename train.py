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
from losses.iou_loss import IoULoss, compute_iou


def dice_coefficient(pred_logits, target, smooth=1e-6):
    probs   = torch.softmax(pred_logits, dim=1)
    one_hot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1)
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    card  = (probs + one_hot).sum(dims)
    return (2.0 * inter + smooth) / (card + smooth)

def seg_loss(pred, target):
    return nn.CrossEntropyLoss()(pred, target) + \
           (1.0 - dice_coefficient(pred, target).mean())

def loc_loss(pred, target):
    return nn.MSELoss()(pred, target) + IoULoss()(pred, target)

def macro_f1(logits, labels, num_classes=37):
    preds  = logits.argmax(1)
    device = logits.device
    f1s    = []
    for c in range(num_classes):
        tp    = ((preds == c) & (labels == c)).sum().float()
        fp    = ((preds == c) & (labels != c)).sum().float()
        fn    = ((preds != c) & (labels == c)).sum().float()
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0
                   else torch.tensor(0.0, device=device))
    return torch.stack(f1s).mean().item()

TASK_FNS = {
    "classify": (
        lambda out, b: nn.CrossEntropyLoss()(out, b["label"]),
        lambda out, b: macro_f1(out, b["label"]),
    ),
    "localize": (
        lambda out, b: loc_loss(out, b["bbox"]),
        lambda out, b: compute_iou(out, b["bbox"]).mean().item(),
    ),
    "segment": (
        lambda out, b: seg_loss(out, b["mask"]),
        lambda out, b: dice_coefficient(out, b["mask"]).mean().item(),
    ),
}


def run_epoch(model, loader, optimizer, device, task, training):
    model.train() if training else model.eval()
    loss_fn, metric_fn = TASK_FNS[task]
    total_loss, total_metric = 0.0, 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out    = model(batch["image"])
            loss   = loss_fn(out, batch)
            metric = metric_fn(out, batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss   += loss.item()
            total_metric += metric

    n = len(loader)
    return total_loss / n, total_metric / n


def build_model(task, device, classifier_ckpt=None):
    ckpt_exists = classifier_ckpt and os.path.exists(classifier_ckpt)

    if task == "classify":
        return ClassificationModel().to(device)

    model  = LocalizationModel(freeze_early=True).to(device) \
             if task == "localize" else SegmentationModel().to(device)
    loader = model.load_backbone_from_classifier \
             if task == "localize" else model.load_encoder_from_classifier

    if ckpt_exists:
        loader(classifier_ckpt, device)
        print(f"  Loaded weights from {classifier_ckpt}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",            type=str,   default="classify",
                        choices=["classify", "localize", "segment"])
    parser.add_argument("--data_root",       type=str,   default="data/pets")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--batch",           type=int,   default=32)
    parser.add_argument("--img_size",        type=int,   default=224)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--dropout_p",       type=float, default=0.5)
    parser.add_argument("--classifier_ckpt", type=str,
                        default="checkpoints/classifier.pth")
    parser.add_argument("--wandb_project",   type=str,
                        default="da6401_assignment2")
    args = parser.parse_args()

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0 if platform.system() == "Windows" else 2
    print(f"Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    wandb.init(project=args.wandb_project, config=vars(args),
               name=f"{args.task}_run")

    train_ds = PetsDataset(args.data_root, "train", "all",
                           get_train_transform(args.img_size), args.img_size)
    val_ds   = PetsDataset(args.data_root, "val", "all",
                           get_val_transform(args.img_size), args.img_size)

    loader_kw    = dict(num_workers=num_workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, **loader_kw)

    model     = build_model(args.task, device, args.classifier_ckpt)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    task_to_name = {"classify": "classifier", "localize": "localizer", "segment": "unet"}
    ckpt_path    = f"checkpoints/{task_to_name[args.task]}.pth"
    metric_name  = {"classify": "F1", "localize": "IoU", "segment": "dice"}[args.task]
    best_val     = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss,  tr_m  = run_epoch(model, train_loader, optimizer,
                                    device, args.task, True)
        val_loss, val_m = run_epoch(model, val_loader,   optimizer,
                                    device, args.task, False)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        wandb.log({"epoch": epoch,
                   "train/loss": tr_loss,  f"train/{metric_name}": tr_m,
                   "val/loss":   val_loss, f"val/{metric_name}":   val_m,
                   "lr": lr_now})

        print(f"[{epoch:03d}/{args.epochs}] "
              f"loss {tr_loss:.4f}/{val_loss:.4f}  "
              f"{metric_name} {tr_m:.4f}/{val_m:.4f}  "
              f"lr {lr_now:.2e}")

        if val_m > best_val:
            best_val = val_m
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch}, ckpt_path)
            print(f" checkpoint saved (val {metric_name}={val_m:.4f})")

    wandb.finish()
    print(f"\nDone. Best val {metric_name}: {best_val:.4f}")


if __name__ == "__main__":
    main()