# Task 1: Noisy Weather Outside (Qualification Stage)

Multi-class weather classification (`0: rain`, `1: fog`, `2: snow`) from distorted onboard autonomous-driving camera images.

## Dataset
- **Train:** `2,087` images (`rain`: 698, `fog`: 637, `snow`: 752), `512×512` RGB PNG
- **Test:** `1,431` unlabeled images

## Solution Architecture
- **Backbone:** `efficientnet_v2_s` (ImageNet pretrained, `torchvision`)
- **Input Size:** `224×224`
- **Augmentations:** `TrivialAugmentWide`, `RandomHorizontalFlip`, `RandomErasing(p=0.2, scale=(0.02, 0.2))`
- **Training:** 5-Fold `StratifiedKFold`, `CrossEntropyLoss`, `AdamW(lr=2e-4, weight_decay=1e-4)`, `CosineAnnealingLR`, 15 epochs/fold
- **Inference:** 10-pass ensemble averaging softmax probabilities across **5 folds × 2 TTA views** (original + horizontal flip)

## Results
- **Rank:** 5th Place (Qualification Stage)
- **Macro F1:** `0.9477` *(dashboard screenshot recorded `0.937` Macro F1 / `0.876` scaled competition points)*

## Usage

```bash
pip install -r requirements.txt

# Train 5 folds and generate outputs/submission.csv
python train_and_infer.py \
  --train-csv path/to/train.csv \
  --test-csv path/to/test.csv \
  --train-dir path/to/train \
  --test-dir path/to/test \
  --output-dir outputs

# Regenerate test prediction distribution chart
python visualize_results.py
```
