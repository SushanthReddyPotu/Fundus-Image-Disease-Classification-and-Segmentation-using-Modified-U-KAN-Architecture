"""
Regenerate training curve PNGs from log.csv (e.g. after training).
Saves loss_curves.png, accuracy_curves.png, iou_dice_curves.png to run_dir (downloadable).
Usage: python plot_training_curves.py --run_dir outputs
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(run_dir: str) -> None:
    """
    Read log.csv from a RetinaKAN training run and save metric plots (PNG):
      - train/val loss
      - train/val accuracy
      - train IoU, val IoU, val Dice
    """
    log_path = os.path.join(run_dir, "log.csv")
    if not os.path.isfile(log_path):
        raise FileNotFoundError("log.csv not found at: %s" % log_path)

    df = pd.read_csv(log_path)
    if "epoch" not in df.columns:
        df["epoch"] = range(len(df))

    epochs = df["epoch"]

    # 1) Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df["loss"], label="train_loss")
    plt.plot(epochs, df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(run_dir, "loss_curves.png"), bbox_inches="tight")
    plt.close()

    # 2) Accuracy
    if "acc" in df.columns and "val_acc" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, df["acc"], label="train_acc")
        plt.plot(epochs, df["val_acc"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train / Val Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(os.path.join(run_dir, "accuracy_curves.png"), bbox_inches="tight")
        plt.close()

    # 3) IoU and Dice
    if "iou" in df.columns and "val_iou" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, df["iou"], label="train_iou")
        plt.plot(epochs, df["val_iou"], label="val_iou")
        if "dice" in df.columns:
            plt.plot(epochs, df["dice"], label="train_dice")
        if "val_dice" in df.columns:
            plt.plot(epochs, df["val_dice"], label="val_dice")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("IoU / Dice over epochs")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(os.path.join(run_dir, "iou_dice_curves.png"), bbox_inches="tight")
        plt.close()

    # 4) F1 score (separate plot)
    if "val_f1_macro" in df.columns:
        plt.figure(figsize=(6, 4))
        if "f1_macro" in df.columns:
            plt.plot(epochs, df["f1_macro"], label="train_f1_macro")
        plt.plot(epochs, df["val_f1_macro"], label="val_f1_macro")
        plt.xlabel("Epoch")
        plt.ylabel("F1 (macro)")
        plt.title("F1 score over epochs")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(os.path.join(run_dir, "f1_curves.png"), bbox_inches="tight")
        plt.close()

    # 5) Combined score (separate plot: train + val)
    if "combined_score" in df.columns:
        plt.figure(figsize=(6, 4))
        if "train_combined_score" in df.columns:
            plt.plot(epochs, df["train_combined_score"], label="train_combined (iou+acc/100)/2")
        plt.plot(epochs, df["combined_score"], label="val_combined (for checkpoint)")
        plt.xlabel("Epoch")
        plt.ylabel("Combined score")
        plt.title("Combined score: (IoU + Acc/100) / 2")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(os.path.join(run_dir, "combined_score_curve.png"), bbox_inches="tight")
        plt.close()

    print("Curves saved to %s" % run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        default="outputs",
        help="Path to run directory containing log.csv (e.g. outputs)",
    )
    args = parser.parse_args()
    plot_curves(args.run_dir)


if __name__ == "__main__":
    main()
