# Task 2: 3D Mesh Quality Control (Main Stage)

Automated multi-label defect detection and quality scoring for 3D meshes used in generative 3D training pipelines.

## Dataset & Metric
- **Train:** `8,964` 3D assets (`1,372` good / `7,592` defective)
- **Test:** `669` 3D assets
- **Input Files per `item_id`:**
  - `{item_id}.png`: `1536×1024` RGB 2×3 stitched multi-view render (4 azimuth views at 90° + top + bottom)
  - `{item_id}.npz`: Mesh geometry (`vertices` `(N, 3)`, `faces` `(M, 3)`)
- **Targets (11 binary columns):** `abstract`, `artifacts`, `intersection`, `lowpoly`, `noisy`, `open`, `partial`, `scale`, `set`, `simple`, and `quality`
- **Evaluation Metric (max 20.0):**
  $$\text{Score} = 10 \times \text{F1}(\text{quality}) + 10 \times \text{F1}_{\text{weighted}}(\text{10 defects})$$

## Experimental Progression

| Version | Backbone | Resolution | Loss & Regularization | Key Change | Recorded Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| V1 | `efficientnet_b0` | `384×576` | `BCEWithLogitsLoss()` | 10-defect baseline | — |
| V2 | `efficientnet_b3` | `384×576` | `BCE(pos_weight)` | Direct 11-target + per-class threshold tuning + H-flip TTA | `12.12` |
| V3-MVCNN | `efficientnet_b1` | `6 × 256×256` | `BCE(pos_weight<=15)` | 6-view cropping + max-pooling (unstable) | — |
| V3-Focal | `efficientnet_b3` | `512×768` | `FocalLoss(gamma=2.0)` | High-res + Focal loss | — |
| **V5 (Final CSV)** | `tf_efficientnetv2_s.in21k_ft_in1k` | `512×768` | `BCE(pos_weight<=25)` | Grad clip `1.0` + **Deterministic Quality Rule** | **`13.711`** |
| V6 | `tf_efficientnetv2_s.in21k_ft_in1k` | `512×768` | `WeightedFocalLoss` | Vertical flip/rotation + 3-way TTA | — |
| V7 | `convnext_tiny.in12k_ft_in1k_384` | `384×576` | `BCE(pos_weight<=25)` | Reverted to orientation-safe augs | — |
| V8 (Safeguard) | `tf_efficientnetv2_s.in21k_ft_in1k` | `384×576` | `BCE(pos_weight<=25)` | Anti-overfit fast baseline | `12.712` |
| V9–V10 | `tf_efficientnetv2_m.in21k_ft_in1k` | `512×768` | Unweighted `BCE` | Medium backbone + AMP mixed precision | — |
| **V11 (Latest)** | `tf_efficientnetv2_m.in21k_ft_in1k` | `512×768` | `BCE(pos_weight<=5)` + 5% LS | Dropout `0.3` + AMP + Grad Accum (eff. batch 8) | — |

- **Final Rank:** **12th Place** (Main Stage)

## Usage

```bash
pip install -r requirements.txt

# 1. Run dataset EDA
python eda.py --train-csv path/to/train.csv --train-dir path/to/train

# 2. Train & predict with V5 preset (or --preset v11)
python train_and_infer.py \
  --preset v5 \
  --train-csv path/to/train.csv \
  --test-csv path/to/test.csv \
  --train-dir path/to/train \
  --test-dir path/to/test \
  --output-dir outputs

# 3. Regenerate defect distribution chart
python visualize_results.py
```
