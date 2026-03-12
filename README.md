# Fundus-Image-Disease-Classification-and-Segmentation-using-Modified-U-KAN-Architecture

## RetinaKAN (U-KAN) — Retinal Vessel Segmentation + Disease Classification

This repo contains a **PyTorch multi-task model** (`UKAN`) that jointly:

- **Segments retinal vessels** (binary mask; evaluated with IoU/Dice)
- **Classifies retinal condition** into 4 classes (Normal / Diabetic / Glaucoma / AMD; evaluated with Accuracy, F1, ROC-AUC)

The training pipeline is end-to-end and produces **reproducible outputs** (saved config, best checkpoint, and validation/test plots).

## Highlights

- **Multi-task learning**: segmentation head + 4-class classification head in one model forward pass (`seg_out, cls_out = model(images)`).
- **Reproducible splits**: fixed seed utilities + deterministic behavior; train/val split uses `random_state=42`.
- **Evaluation artifacts**: confusion matrix + ROC curves saved as PNGs; optional mask export for qualitative review.

## Results

Full results (metrics + plots) are available in the Kaggle notebook:

- `https://www.kaggle.com/code/sushanthreddypotu/fives-ukan5`

This notebook includes the reported segmentation and classification performance (e.g., IoU/Dice, Accuracy/F1, ROC-AUC) and visual outputs such as confusion matrices and ROC curves.

## Tech stack

- **Python**: 3.8+ recommended
- **Deep learning**: PyTorch (`torch`, `torchvision`)
- **Augmentations**: Albumentations + OpenCV
- **Metrics/plots**: scikit-learn, matplotlib

## Repository structure

```
.
├── archs.py                 # UKAN model definition
├── kan.py                   # KAN building blocks
├── dataset.py               # FIVESDataset (loads images/masks + class label from filename)
├── losses.py                # segmentation loss (e.g., BCE+Dice)
├── metrics.py               # IoU/Dice utilities
├── train.py                 # training + checkpointing (writes outputs/config.yml)
├── val.py                   # validation (plots confusion matrix + ROC; optional mask export)
├── test_eval.py             # test evaluation (plots confusion matrix + ROC; optional mask export)
├── prepare_fives.py         # dataset prep helper (if needed)
├── plot_training_curves.py  # plots training history from saved logs
├── requirements.txt
└── environment.yml          # legacy conda environment (older linux pins)
```

## Dataset layout (expected)

By default scripts look for `datasets/FIVES`. They support either of these folder conventions:

### Dataset attribution / download

- **Attribution**: This project uses the **FIVES** dataset for experiments. All credit and thanks go to the **original FIVES dataset creators** for making it publicly available.
- **Download link (Kaggle)**: `https://www.kaggle.com/datasets/sushanthreddypotu/fives-dataset`

**Option A**

```
datasets/FIVES/
  train/images/*.png
  train/masks/*.png
  test/images/*.png
  test/masks/*.png
```

**Option B (original FIVES-like naming)**

```
datasets/FIVES/
  train/Original/*.png
  train/Ground truth/*.png
  test/Original/*.png
  test/Ground truth/*.png
```

### Filename convention for classification labels

`dataset.py` extracts the class label from the **suffix after the final underscore** in the filename (without extension):

- `_N` → Normal (0)
- `_D` → Diabetic Retinopathy (1)
- `_G` → Glaucoma (2)
- `_A` → AMD (3)

Example: `000123_N.png` or `patient42_left_G.png`

## Setup

### 1) Create environment (venv + pip)

```bash
python -m venv .venv
```

Activate:

- Windows (PowerShell):

```bash
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

Install PyTorch first (pick the command for your CUDA/CPU from the official selector), then install the rest:

```bash
pip install -r requirements.txt
```

## Training

Trains the model and writes `outputs/config.yml` plus the best checkpoint `outputs/best_model.pth`.

```bash
python train.py --data_dir datasets/FIVES --output_dir outputs --epochs 70 --batch_size 4 --lr 3e-4 --seed 42
```

## Validation

Validates using the checkpoint at `outputs/best_model.pth` and the config saved during training.

```bash
python val.py --output_dir outputs
```

Optional: save predicted vessel masks to `outputs/out_val/`:

```bash
python val.py --output_dir outputs --save_pred
```

## Test evaluation

Evaluates on `datasets/FIVES/test` and saves plots to `outputs/`.

```bash
python test_eval.py --data_dir datasets/FIVES --model_path outputs/best_model.pth --output_dir outputs
```

Optional: save predicted vessel masks to `outputs/out_test/`:

```bash
python test_eval.py --data_dir datasets/FIVES --model_path outputs/best_model.pth --output_dir outputs --save_pred
```

## Outputs

After a typical run, `outputs/` will contain:

- `config.yml` — training config used by `val.py`
- `best_model.pth` — best checkpoint chosen during training
- `confusion_matrix.png`, `roc_validation.png` — validation plots
- `confusion_matrix_test.png`, `roc_test.png` — test plots
- `out_val/` and/or `out_test/` — optional predicted masks

## Notes

- `environment.yml` contains older pinned versions (Linux/conda). For a clean setup on modern machines, prefer `requirements.txt` with a recent Python and an appropriate PyTorch install.
- If you get a label error like “Unknown label extracted”, confirm your image filenames end with `_N`, `_D`, `_G`, or `_A` before the extension.
