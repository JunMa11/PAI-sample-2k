# Deoxy & HbO Single-Channel Experiment Report

**Date:** 2026-02-23 | **Architecture:** ResNet18-UNet | **Dataset:** 2000 samples (400 val) | **Metric checkpoint:** best MAE

---

## 1. Deoxy Hemoglobin Results

> **Target:** GTdeoxy (1 channel) | **A_MAX:** 0.038

### Raw Predictions

| Loss | MAE ↓ | PSNR ↑ | SSIM ↑ | Best Epoch |
|---|---|---|---|---|
| **L1** | **0.0736** | **23.92** | **0.813** | — |
| L1 + SSIM | 0.0811 | 22.68 | 0.773 | 263 |
| Amp-L1 | 0.0941 | 22.33 | 0.771 | — |
| ROI-Weighted | 0.1379 | 19.34 | 0.579 | 157 |

### Quantized Predictions (k=3 k-means)

| Loss | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|
| **L1** | **0.0737** | **24.33** | **0.860** |
| L1 + SSIM | 0.0811 | 22.99 | 0.814 |
| Amp-L1 | 0.0943 | 22.55 | 0.810 |
| ROI-Weighted | 0.1379 | 19.43 | 0.598 |

**Winner: L1 loss** — best across all metrics (raw and quantized). Quantization consistently improves PSNR (+0.3–0.4 dB) and SSIM (+3–5%), confirming piecewise-constant GT structure.

---

## 2. HbO (Oxyhemoglobin) Results

> **Target:** GTHbO (1 channel) | **A_MAX:** 0.062

### Raw Predictions

| Loss | MAE ↓ | PSNR ↑ | SSIM ↑ | Best Epoch |
|---|---|---|---|---|
| **L1** | **0.0885** | **21.68** | **0.796** | — |
| L1 + SSIM | 0.1011 | 20.54 | 0.767 | 210 |
| Amp-L1 | 0.1157 | 19.83 | 0.752 | — |
| ROI-Weighted | 0.1728 | 16.23 | 0.515 | 242 |

### Quantized Predictions (k=3 k-means)

| Loss | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|
| **L1** | **0.0885** | **22.10** | **0.859** |
| L1 + SSIM | 0.1011 | 20.80 | 0.811 |
| Amp-L1 | 0.1162 | 19.99 | 0.793 |
| ROI-Weighted | 0.1729 | 16.24 | 0.537 |

**Winner: L1 loss** — again best across all metrics. Same quantization benefit observed.

---

## 3. Cross-Task Comparison

### Best model per task (all ResNet18 + L1, raw predictions)

| Task | Channels | A_MAX | MAE ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|
| **AbsMaps** | 2 (755nm + 808nm) | 0.020 | **0.067** | **23.16** | **0.809** |
| Deoxy | 1 | 0.038 | 0.074 | 23.92 | 0.813 |
| HbO | 1 | 0.062 | 0.088 | 21.68 | 0.796 |

Despite being single-channel (simpler architecture output), deoxy and HbO perform worse than the 2-channel absMaps model.

---

## 4. Distribution Analysis: Why Deoxy & HbO Are Harder

To understand the performance gap, we analyzed the normalized GT distributions across all 2000 samples:

### Normalized Foreground Statistics (value / A_MAX)

| Metric | AbsMaps ch0 | AbsMaps ch1 | Deoxy | HbO |
|---|---|---|---|---|
| Norm mean | 0.250 | 0.228 | 0.215 | **0.262** |
| Norm std | 0.107 | 0.090 | **0.127** | 0.119 |
| Per-sample max/A_MAX mean | 0.494 | 0.451 | 0.425 | **0.517** |
| Per-sample max/A_MAX std | 0.179 | 0.148 | **0.232** | 0.208 |
| Coeff of variation | 0.427 | 0.396 | **0.589** | 0.456 |
| Saturation (>0.9×A_MAX) | 0.03% | 0.00% | 0.06% | **0.15%** |

### Key Findings

1. **Deoxy has the highest coefficient of variation (0.589)** — nearly 50% higher than absMaps channels (~0.41). The normalized target values vary far more across samples, meaning the model faces highly inconsistent difficulty levels. Some samples have very low foreground values while others nearly saturate A_MAX.

2. **HbO has the highest normalized mean (0.262) and per-sample max (0.517)** — its values occupy a larger fraction of the normalized range. Higher average values produce proportionally larger absolute errors under L1 loss, consistent with our earlier finding that GT magnitude is the primary driver of prediction difficulty.

3. **Both deoxy and HbO have wider per-sample max spread** (std 0.21–0.23 vs 0.15–0.18 for absMaps). This heterogeneity means the model cannot rely on a stable target range — it must generalize across samples with very different intensity profiles.

4. **HbO has the most saturation** (0.15% of all foreground pixels and up to 10.1% per sample near A_MAX), while absMaps ch1 has zero saturation. Near-ceiling values are hardest to reconstruct accurately.


The lower performance of deoxy and HbO models could be explained by **greater distribution heterogeneity**. Deoxy/HbO targets have:
- More variable intensity ranges across samples (high CoV)
- More extreme per-sample peaks relative to A_MAX
- More inconsistent difficulty levels for the model

---

## New Idea: Two-Head Model (Joint Regression + Segmentation)

Build a model that jointly predicts both the continuous absorption map and the discrete tissue segmentation mask.

Needs several days to implment this idea and train the model.

### Architecture

The model shares a ResNet18 encoder backbone but splits into two decoder heads:

- **Regression head** (2-ch output): predicts continuous absorption values, same as the standard model
- **Segmentation head** (3-ch output): predicts a 3-class tissue label map (background=0, tissue A=1, tissue B=2)

At inference time, the predicted foreground probability from the segmentation head is used to mask-guide the regression output: `final_pred = reg_pred × fg_prob`, zeroing out background regions and sharpening tissue boundaries.

### Segmentation Label Generation

GT segmentation labels are generated on-the-fly during training using **Otsu's thresholding** on channel 0 of the normalized absorption map. Otsu's method finds the optimal threshold that minimises intra-class variance between the two foreground tissue types, producing a robust 3-class label without requiring manual annotation.

### Combined Loss

The training loss combines regression and segmentation objectives:

```
L_total = L_regression + λ_seg × L_segmentation
```

Where:
- `L_regression` = L1 (or L1+SSIM) on the continuous absorption prediction
- `L_segmentation` = Cross-Entropy + Dice loss on the 3-class segmentation
- `λ_seg` = segmentation loss weight (default: 0.5)
