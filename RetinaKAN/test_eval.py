"""
RetinaKAN test evaluation: same metrics as two-stage (test_stage2).
Saves confusion_matrix_test.png and roc_test.png to output_dir (downloadable).
"""
import os
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

import archs
from dataset import FIVESDataset
from metrics import iou_score
from albumentations import Resize, Normalize
from albumentations.core.composition import Compose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='datasets/FIVES')
    parser.add_argument('--model_path', default='outputs/best_model.pth')
    parser.add_argument('--output_dir', default='outputs', help='Save confusion matrix and ROC curves here (downloadable)')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--save_pred', action='store_true', help='Save predicted vessel masks to output_dir/out_test/')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_img_dir = os.path.join(args.data_dir, "test", "images")
    test_mask_dir = os.path.join(args.data_dir, "test", "masks")
    if not os.path.isdir(test_img_dir):
        test_img_dir = os.path.join(args.data_dir, "test", "Original")
        test_mask_dir = os.path.join(args.data_dir, "test", "Ground truth")

    all_paths = sorted(glob(os.path.join(test_img_dir, "*.png")))
    test_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_paths]
    print("Test samples: %d" % len(test_ids))

    test_transform = Compose([
        Resize(256, 256),
        Normalize(),
    ])
    test_dataset = FIVESDataset(
        test_ids, test_img_dir, test_mask_dir, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = archs.UKAN(num_classes=1, cls_classes=4).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    iou_sum = 0
    dice_sum = 0
    n_total = 0

    out_test_dir = os.path.join(args.output_dir, 'out_test') if args.save_pred else None
    if out_test_dir:
        os.makedirs(out_test_dir, exist_ok=True)

    batch_start = 0
    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            seg_out, cls_out = model(images)
            probs = torch.softmax(cls_out, dim=1)
            _, preds = torch.max(cls_out, 1)

            iou, dice, _ = iou_score(seg_out, masks)
            n = images.size(0)
            iou_sum += iou * n
            dice_sum += dice * n
            n_total += n

            if out_test_dir:
                pred_masks = (torch.sigmoid(seg_out) >= 0.5).cpu().numpy()
                for i in range(n):
                    img_id = test_ids[batch_start + i]
                    m = (pred_masks[i, 0] * 255).astype(np.uint8)
                    Image.fromarray(m, mode='L').save(os.path.join(out_test_dir, '%s.png' % img_id))
                batch_start += n

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)

    test_iou = iou_sum / n_total
    test_dice = dice_sum / n_total
    accuracy = 100 * np.mean(all_preds == all_labels)
    acc_global = accuracy_score(all_labels, all_preds)
    test_combined_score = (test_iou + acc_global) / 2  # same formula as val: (iou + acc/100) / 2

    print("[TEST SET]")
    print("Accuracy:          %.4f (%.2f%%)" % (acc_global, accuracy))
    print("IoU:               %.4f" % test_iou)
    print("Dice:              %.4f" % test_dice)
    print("Combined score:    %.4f  (IoU + Acc/100) / 2" % test_combined_score)

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    num_classes = cm.shape[0]
    specificities = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2, 3], zero_division=0
    )

    print("F1 (macro):        %.4f" % f1_macro)
    print("F1 (weighted):     %.4f" % f1_weighted)
    target_names = ["Normal (0)", "Diabetic (1)", "Glaucoma (2)", "AMD (3)"]
    digits = 4
    width = max(len(t) for t in target_names)
    head_fmt = "{:>{width}s}  {:>9}  {:>9}  {:>9}  {:>11}  {:>7}".format
    report = [head_fmt("", "precision", "recall", "f1-score", "specificity", "support", width=width)]
    report.append("")
    for i, name in enumerate(target_names):
        report.append(
            head_fmt(
                name,
                round(precision[i], digits),
                round(recall[i], digits),
                round(f1[i], digits),
                round(specificities[i], digits),
                int(support[i]),
                width=width,
            )
        )
    report.append("")
    report.append(
        head_fmt(
            "macro avg",
            round(precision.mean(), digits),
            round(recall.mean(), digits),
            round(f1.mean(), digits),
            round(np.mean(specificities), digits),
            int(support.sum()),
            width=width,
        )
    )
    report.append(
        head_fmt(
            "weighted avg",
            round(np.average(precision, weights=support), digits),
            round(np.average(recall, weights=support), digits),
            round(np.average(f1, weights=support), digits),
            round(np.average(specificities, weights=support), digits),
            int(support.sum()),
            width=width,
        )
    )
    print("Classification report (test):")
    print("\n".join(report))

    # AUC-ROC per class and macro
    aucs = []
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for idx, name in enumerate(["Normal", "Diabetic", "Glaucoma", "AMD"]):
        y_true = (all_labels == idx).astype(int)
        y_score = all_probs[:, idx]
        if np.unique(y_true).size < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax_roc.plot(fpr, tpr, label="%s (AUC=%.3f)" % (name, auc))
        aucs.append(auc)
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    print("AUC-ROC per class (test):")
    for name, auc in zip(["Normal", "Diabetic", "Glaucoma", "AMD"], aucs):
        print("  %s: %s" % (name, "nan" if np.isnan(auc) else "%.4f" % auc))
    print("Macro AUC-ROC (test): %s" % ("nan" if np.isnan(macro_auc) else "%.4f" % macro_auc))

    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curves (test set)")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    roc_path = os.path.join(args.output_dir, "roc_test.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close(fig_roc)
    print("ROC curves saved to %s" % roc_path)

    # Confusion matrix (downloadable)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Normal", "Diabetic", "Glaucoma", "AMD"],
        yticklabels=["Normal", "Diabetic", "Glaucoma", "AMD"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix_test.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    print("Confusion matrix saved to %s" % cm_path)

    if args.save_pred:
        print("Predicted masks saved to %s" % out_test_dir)

    print("Test evaluation complete. All artifacts are in %s (downloadable)." % args.output_dir)


if __name__ == "__main__":
    main()
