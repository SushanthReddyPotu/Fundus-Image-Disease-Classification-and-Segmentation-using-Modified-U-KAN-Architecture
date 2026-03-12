import os
import sys
import argparse
import random
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from albumentations import (
    RandomRotate90,
    HorizontalFlip,
    VerticalFlip,
    Affine,
    Resize,
    Normalize
)
from albumentations.core.composition import Compose

import archs
import losses
from dataset import FIVESDataset
from metrics import iou_score


# -------------------------------------------------
# Reproducibility (seed matches train/val split random_state=42)
# -------------------------------------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# Arguments
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--data_dir', default='datasets/FIVES')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    args = parse_args()
    seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config for val/test scripts and reproducibility
    import yaml
    config = {
        'arch': 'UKAN',
        'num_classes': 1,
        'cls_classes': 4,
        'input_h': 256,
        'input_w': 256,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'dataseed': args.seed,
    }
    with open(os.path.join(args.output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print("Config saved to %s" % os.path.join(args.output_dir, 'config.yml'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------------------------
    # DATASET SPLIT (works with standard FIVES folder: train/Original, train/Ground truth)
    # -------------------------------------------------
    train_img_dir = os.path.join(args.data_dir, "train", "images")
    train_mask_dir = os.path.join(args.data_dir, "train", "masks")
    if not os.path.isdir(train_img_dir):
        train_img_dir = os.path.join(args.data_dir, "train", "Original")
        train_mask_dir = os.path.join(args.data_dir, "train", "Ground truth")
        if os.path.isdir(train_img_dir):
            print("Using FIVES layout: train/Original, train/Ground truth")
    if not os.path.isdir(train_img_dir):
        raise FileNotFoundError(
            "Train images not found. Expect either %s/train/images or %s/train/Original"
            % (args.data_dir, args.data_dir)
        )

    all_paths = sorted(glob(os.path.join(train_img_dir, "*.png")))
    all_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_paths]

    train_ids, val_ids = train_test_split(
        all_ids, test_size=0.2, random_state=42
    )

    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples:   {len(val_ids)}")

    # -------------------------------------------------
    # AUGMENTATIONS
    # -------------------------------------------------
    train_transform = Compose([
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Affine(scale=(0.9, 1.1),
               translate_percent=(0.05, 0.05),
               rotate=(-15, 15),
               p=0.5),
        Resize(256, 256),
        Normalize()
    ])

    val_transform = Compose([
        Resize(256, 256),
        Normalize()
    ])

    train_dataset = FIVESDataset(train_ids, train_img_dir, train_mask_dir, transform=train_transform)
    val_dataset = FIVESDataset(val_ids, train_img_dir, train_mask_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    model = archs.UKAN(num_classes=1, cls_classes=4).to(device)

    seg_criterion = losses.BCEDiceLoss().to(device)

    class_weights = torch.tensor([1.6, 1.0, 1.1, 1.0]).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -------------------------------------------------
    # Early Stopping Setup
    # -------------------------------------------------
    best_score = 0
    patience = 10
    early_stop_counter = 0

    log_rows = []

    # -------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------
    for epoch in range(args.epochs):

        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0
        train_iou = 0
        train_dice_sum = 0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []

        for images, masks, labels in tqdm(train_loader, disable=not sys.stdout.isatty(), desc='Train'):

            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            seg_out, cls_out = model(images)

            seg_loss = seg_criterion(seg_out, masks)
            cls_loss = cls_criterion(cls_out, labels)

            loss = seg_loss + 1.2 * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iou, dice, _ = iou_score(seg_out, masks)
            train_dice_sum += dice

            _, preds = torch.max(cls_out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_train_preds.append(preds.detach().cpu().numpy())
            all_train_labels.append(labels.cpu().numpy())

            train_loss += loss.item()
            train_iou += iou

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_dice = train_dice_sum / len(train_loader)
        train_acc = 100 * correct / total
        all_train_preds = np.concatenate(all_train_preds)
        all_train_labels = np.concatenate(all_train_labels)
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice_sum = 0
        correct = 0
        total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, masks, labels in val_loader:

                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                seg_out, cls_out = model(images)

                seg_loss = seg_criterion(seg_out, masks)
                cls_loss = cls_criterion(cls_out, labels)

                loss = seg_loss + 1.2 * cls_loss

                iou, dice, _ = iou_score(seg_out, masks)
                val_dice_sum += dice

                _, preds = torch.max(cls_out, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_val_preds.append(preds.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

                val_loss += loss.item()
                val_iou += iou

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice = val_dice_sum / len(val_loader)
        val_acc = 100 * correct / total

        all_val_preds = np.concatenate(all_val_preds)
        all_val_labels = np.concatenate(all_val_labels)
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        # Combined score: (iou + acc/100) / 2 (val used for checkpoint selection)
        train_combined_score = (train_iou + train_acc / 100) / 2
        combined_score = (val_iou + val_acc / 100) / 2

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1_macro:.4f} | Combined: {train_combined_score:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1_macro:.4f} | Combined: {combined_score:.4f}")

        log_rows.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'iou': train_iou,
            'dice': train_dice,
            'acc': train_acc,
            'f1_macro': train_f1_macro,
            'train_combined_score': train_combined_score,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'val_acc': val_acc,
            'val_f1_macro': val_f1_macro,
            'combined_score': combined_score,
        })

        # ---------------- SAVE + EARLY STOP ----------------
        if combined_score > best_score:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pth"))
            best_score = combined_score
            early_stop_counter = 0
            print("Saved best model (combined score = (val_iou + val_acc/100) / 2).")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # ---------------- SAVE LOG AND TRAINING CURVES (downloadable) ----------------
    if log_rows:
        import csv
        log_path = os.path.join(args.output_dir, "log.csv")
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            w.writeheader()
            w.writerows(log_rows)
        print("Log saved to %s" % log_path)

        # Save training curve PNGs for download
        epochs_list = [r['epoch'] for r in log_rows]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_list, [r['loss'] for r in log_rows], label='train_loss')
        plt.plot(epochs_list, [r['val_loss'] for r in log_rows], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train / Val Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(epochs_list, [r['acc'] for r in log_rows], label='train_acc')
        plt.plot(epochs_list, [r['val_acc'] for r in log_rows], label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Train / Val Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join(args.output_dir, 'accuracy_curves.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(epochs_list, [r['iou'] for r in log_rows], label='train_iou')
        plt.plot(epochs_list, [r['val_iou'] for r in log_rows], label='val_iou')
        plt.plot(epochs_list, [r['dice'] for r in log_rows], label='train_dice')
        plt.plot(epochs_list, [r['val_dice'] for r in log_rows], label='val_dice')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('IoU / Dice over epochs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join(args.output_dir, 'iou_dice_curves.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(epochs_list, [r['f1_macro'] for r in log_rows], label='train_f1_macro')
        plt.plot(epochs_list, [r['val_f1_macro'] for r in log_rows], label='val_f1_macro')
        plt.xlabel('Epoch')
        plt.ylabel('F1 (macro)')
        plt.title('F1 score over epochs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join(args.output_dir, 'f1_curves.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(epochs_list, [r['train_combined_score'] for r in log_rows], label='train_combined (iou+acc/100)/2')
        plt.plot(epochs_list, [r['combined_score'] for r in log_rows], label='val_combined (for checkpoint)')
        plt.xlabel('Epoch')
        plt.ylabel('Combined score')
        plt.title('Combined score: (IoU + Acc/100) / 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join(args.output_dir, 'combined_score_curve.png'), bbox_inches='tight')
        plt.close()

        print("Training curves saved to %s (loss_curves.png, accuracy_curves.png, iou_dice_curves.png, f1_curves.png, combined_score_curve.png)" % args.output_dir)

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()
