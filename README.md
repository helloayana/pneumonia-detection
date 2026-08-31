# pneumonia-detection


# Pneumonia Detection from Chest X-Rays

Binary image classification (Normal vs Pneumonia) using transfer learning on chest X-ray images.

## Problem

Given a chest X-ray image, predict whether it shows signs of pneumonia or is normal. This is a binary classification task on medical images with a real-world class imbalance.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle, 5,863 images total.

| Split | Normal | Pneumonia |
|---|---|---|
| Train | 1,341 | 3,875 |
| Validation | 8 | 8 |
| Test | 234 | 390 |

Note: train set is imbalanced (~3:1 Pneumonia:Normal), and the validation set is very small (16 images) — addressed via `class_weight` during training and by relying on the larger, independent test set for final evaluation.

## Approach

**Base model:** MobileNetV2, pretrained on ImageNet (`include_top=False`), chosen for its small size and fast training — a practical fit given limited compute (free Colab GPU) and a tight timeline.

**Two-stage training:**
1. **Frozen base + custom head.** Froze all MobileNetV2 layers, trained only a new classification head (`GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)`, ~164K trainable params) for 10 epochs.
2. **Fine-tuning.** Unfroze the last 20 layers of MobileNetV2 and continued training for 5 epochs with a low learning rate (`1e-5`) to gently adapt high-level features to the X-ray domain without destroying the pretrained weights.

**Handling class imbalance:** used `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` to weight the loss function, computed as `{Normal: 1.94, Pneumonia: 0.67}` — penalizing errors on the minority class (Normal) more heavily so the model doesn't default to predicting the majority class.

**Data augmentation** (train only): rotation (±15°), zoom (±10%), horizontal flip. Validation/test data is only rescaled, kept unaugmented to reflect real-world conditions.

## Results

| Model | Test Accuracy | Normal Recall | Pneumonia Recall |
|---|---|---|---|
| v1 — frozen base only | 88.94% | 0.74 | 0.98 |
| v2 — fine-tuned (last 20 layers) | **90.71%** | **0.87** | 0.93 |

**Confusion matrices:**

v1 (frozen):
```
                Predicted
              Normal  Pneumonia
Actual Normal   174      60
   Pneumonia      9     381
```

v2 (fine-tuned):
```
                Predicted
              Normal  Pneumonia
Actual Normal   203      31
   Pneumonia     27     363
```

### Interpretation

Fine-tuning improved overall accuracy (88.94% → 90.71%) but **shifted the error trade-off**, not just improved everything uniformly: false positives on Normal dropped sharply (60 → 31), while missed Pneumonia cases increased (9 → 27, recall 0.98 → 0.93).

This matters because **in a medical screening context, the cost of a missed diagnosis (false negative) is typically higher than a false alarm (false positive)**. By that criterion, v1 (frozen) may actually be the more appropriate model despite its lower raw accuracy — a reminder that "higher accuracy" doesn't automatically mean "better for the task." Both model versions are included in this repo (`model_v1_frozen.keras`, `model_v2_finetuned.keras`) rather than silently picking one, so this trade-off stays visible.

## What I'd improve with more time

- **Threshold tuning / ROC-AUC analysis** — instead of a fixed 0.5 cutoff, tune the decision threshold based on the desired recall/precision balance for the deployment context.
- **Error analysis** — manually inspect the misclassified images to check for patterns (image quality, ambiguous cases, possible duplicate patients across splits — a known issue with this dataset).
- **Grad-CAM visualization** — to verify the model is attending to clinically relevant regions of the X-ray, not spurious artifacts.
- **Second architecture comparison** (e.g. EfficientNetB0) to check whether the trade-off above is specific to MobileNetV2 or general to the approach.

## How to run

1. Open `pneumonia_detection.ipynb` in Google Colab (button below)
2. Runtime → Change runtime type → GPU
3. Add your own Kaggle API token in the first cell
4. Run cells top to bottom

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ТВОЙ-USERNAME/ТВОЙ-РЕПОЗИТОРИЙ/blob/main/pneumonia_detection.ipynb)

## Stack

Python · TensorFlow / Keras · scikit-learn · Google Colab (GPU)

## Author

[Ayana] — [https://github.com/helloayana]
