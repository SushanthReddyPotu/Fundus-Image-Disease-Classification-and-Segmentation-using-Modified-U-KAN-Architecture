"""
RetinaKAN validation: same metrics as two-stage (val_stage2).
Loads best_model.pth and config from output_dir; saves confusion_matrix.png and roc_validation.png.
Combined score (used for checkpoint selection during training) = (val_iou + val_acc/100) / 2.
"""
import argparse
import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
import yaml
from albumentations import Resize, Normalize
from albumentations.core.composition import Compose
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

import archs
from dataset import FIVESDataset
from metrics import iou_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='outputs', help='Directory with config.yml and best_model.pth')
    parser.add_argument('--data_dir', default=None, help='Override data_dir from config')
    parser.add_argument('--save_pred', action='store_true', help='Save predicted vessel masks to output_dir/out_val/')
    return parser.parse_args()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    config_path = os.path.join(args.output_dir, 'config.yml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError('Config not found: %s (run train.py first)' % config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    seed_torch(seed=config.get('seed', 42))

    data_dir = args.data_dir or config.get('data_dir', 'datasets/FIVES')
    train_img_dir = os.path.join(data_dir, 'train', 'images')
    train_mask_dir = os.path.join(data_dir, 'train', 'masks')
    if not os.path.isdir(train_img_dir):
        train_img_dir = os.path.join(data_dir, 'train', 'Original')
        train_mask_dir = os.path.join(data_dir, 'train', 'Ground truth')

    all_paths = sorted(glob(os.path.join(train_img_dir, '*.png')))
    all_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_paths]
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=config.get('dataseed', 42))

    print('Validation samples: %d' % len(val_ids))

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])
    val_dataset = FIVESDataset(val_ids, train_img_dir, train_mask_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.get('batch_size', 4), shuffle=False
    )

    model = archs.UKAN(
        num_classes=config['num_classes'],
        cls_classes=config['cls_classes'],
    )
    ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError('Checkpoint not found: %s' % ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    acc_meter = 0
    n_acc = 0
    iou_sum = 0
    dice_sum = 0
    all_labels = []
    all_preds = []
    all_probs = []

    out_val_dir = os.path.join(args.output_dir, 'out_val') if args.save_pred else None
    if out_val_dir:
        os.makedirs(out_val_dir, exist_ok=True)

    batch_start = 0
    with torch.no_grad():
        for images, masks, labels in tqdm(val_loader, desc='Val'):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            seg_out, cls_out = model(images)
            probs = torch.softmax(cls_out, dim=1)
            preds = probs.argmax(1)
            iou, dice, _ = iou_score(seg_out, masks)

            n = images.size(0)
            acc_meter += (preds == labels).float().sum().item()
            n_acc += n
            iou_sum += iou * n
            dice_sum += dice * n
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if out_val_dir:
                pred_masks = (torch.sigmoid(seg_out) >= 0.5).cpu().numpy()
                for i in range(n):
                    img_id = val_ids[batch_start + i]
                    m = (pred_masks[i, 0] * 255).astype(np.uint8)
                    Image.fromarray(m, mode='L').save(os.path.join(out_val_dir, '%s.png' % img_id))
                batch_start += n

    n_total = sum(a.size for a in all_labels)
    val_acc_batch = acc_meter / n_acc
    val_iou = iou_sum / n_total
    val_dice = dice_sum / n_total

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    # Combined score (same as used for saving best checkpoint in train.py)
    combined_score = (val_iou + val_acc_batch) / 2
    print('Combined score (val_iou + val_acc/100) / 2: %.4f (used for checkpoint selection)' % combined_score)

    print('[VALIDATION]')
    print('Accuracy (batch-avg): %.4f' % val_acc_batch)
    print('IoU:                  %.4f' % val_iou)
    print('Dice:                 %.4f' % val_dice)

    acc_global = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

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

    print('Accuracy (global):    %.4f' % acc_global)
    print('F1 (macro):           %.4f' % f1_macro)
    print('F1 (weighted):         %.4f' % f1_weighted)
    target_names = ['Normal (0)', 'Diabetic (1)', 'Glaucoma (2)', 'AMD (3)']
    digits = 4
    width = max(len(t) for t in target_names)
    head_fmt = '{:>{width}s}  {:>9}  {:>9}  {:>9}  {:>11}  {:>7}'.format
    report = [head_fmt('', 'precision', 'recall', 'f1-score', 'specificity', 'support', width=width)]
    report.append('')
    for i, name in enumerate(target_names):
        report.append(head_fmt(name, round(precision[i], digits), round(recall[i], digits),
                             round(f1[i], digits), round(specificities[i], digits), int(support[i]), width=width))
    report.append('')
    report.append(head_fmt('macro avg', round(precision.mean(), digits), round(recall.mean(), digits),
                           round(f1.mean(), digits), round(np.mean(specificities), digits), int(support.sum()), width=width))
    report.append(head_fmt('weighted avg', round(np.average(precision, weights=support), digits),
                           round(np.average(recall, weights=support), digits),
                           round(np.average(f1, weights=support), digits),
                           round(np.average(specificities, weights=support), digits), int(support.sum()), width=width))
    print('Classification report (validation):')
    print('\n'.join(report))

    # AUC-ROC per class and macro
    aucs = []
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for idx, name in enumerate(['Normal', 'Diabetic', 'Glaucoma', 'AMD']):
        y_true = (all_labels == idx).astype(int)
        y_score = all_probs[:, idx]
        if np.unique(y_true).size < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax_roc.plot(fpr, tpr, label='%s (AUC=%.3f)' % (name, auc))
        aucs.append(auc)
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float('nan')
    print('AUC-ROC per class (validation):')
    for name, auc in zip(['Normal', 'Diabetic', 'Glaucoma', 'AMD'], aucs):
        print('  %s: %s' % (name, 'nan' if np.isnan(auc) else '%.4f' % auc))
    print('Macro AUC-ROC (validation): %s' % ('nan' if np.isnan(macro_auc) else '%.4f' % macro_auc))

    ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC curves (validation)')
    ax_roc.legend(loc='lower right')
    fig_roc.tight_layout()
    roc_path = os.path.join(args.output_dir, 'roc_validation.png')
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close(fig_roc)
    print('ROC curves saved to %s' % roc_path)

    # Confusion matrix plot (downloadable)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=['Normal', 'Diabetic', 'Glaucoma', 'AMD'],
        yticklabels=['Normal', 'Diabetic', 'Glaucoma', 'AMD'],
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion matrix (validation)',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close(fig)
    print('Confusion matrix saved to %s' % cm_path)

    if args.save_pred:
        print('Predicted masks saved to %s' % out_val_dir)


if __name__ == '__main__':
    main()
