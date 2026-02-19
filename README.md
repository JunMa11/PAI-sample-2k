# Photoacoustic Reconstruction — Experiment Report

> **Dataset:** 2,000 samples (1,600 train / 400 validation, random_state=42)  
> **Training:** AdamW (lr=1e-4, wd=1e-2), ReduceLROnPlateau, 300 epochs max, early stopping (patience=50)  
> **Batch size:** 64 &nbsp;|&nbsp; **GPU:** CUDA  

---

## Experiment Overview

The goal was to identify the optimal loss function and network architecture for photoacoustic image reconstruction.

### Pipeline
1. **Phase 1** — Train 6 loss variants with ResNet18 (no augmentation, no standardization)
2. **Phase 2** — Select the best loss by validation MAE
3. **Phase 3** — Test data augmentation and input standardization on the best loss (ResNet18)
4. **Phase 4** — Compare network architectures (ResNet18 / ResNet34 / ResNet50) with top settings

---

## Phase 1: Loss Function Comparison (ResNet18)

All models use `resnet18` backbone, no augmentation, no input standardization.

### Loss Function Definitions

| Loss Function | Formula | Description |
|--------------|---------|-------------|
| `l1` | L1(pred, target) | Standard Mean Absolute Error over all pixels. Simple baseline. |
| `roi_weighted` | Σ\|pred − target\| × w, where w = 1 + 4 × mask | Spatially-weighted L1 that applies 5× weight (`ROI_WEIGHT=5.0`) to non-zero ROI pixels. Forces the model to prioritize the signal region over the background. |
| `l1_ssim` | 0.5 × L1 + 0.5 × (1 − SSIM) | Combines pixel-level L1 with structural similarity loss (`SSIM_WEIGHT=0.5`). Encourages both pixel accuracy and structural fidelity. |
| `l1_ssim_vgg` | 0.4 × L1 + 0.5 × (1 − SSIM) + 0.1 × VGG | Adds VGG16 perceptual loss (`VGG_WEIGHT=0.1`) to L1+SSIM. Compares feature representations via VGG16 relu3\_3 layers for high-level structural similarity. |
| `amp_l1` | Σ\|pred − target\| × (1 + alpha × target/target\_max) | Amplitude-weighted L1 (`alpha=3.0`). High-absorption pixels receive proportionally higher loss weight, focusing the model on reconstructing strong signals accurately. |
| `amp_l1_ssim` | 0.5 × amp\_L1 + 0.5 × (1 − SSIM) | Combines amplitude-weighted L1 with SSIM loss (`SSIM_WEIGHT=0.5`). Balances amplitude-aware pixel accuracy with structural preservation. |

### Training Metrics

| # | Loss Function | Best MAE | MAE Epoch | Best PSNR (dB) | PSNR Epoch | Best SSIM | SSIM Epoch |
|---|--------------|----------|-----------|----------------|------------|-----------|------------|
| 1 | `l1` | 0.02730 | 259 | 27.47 | 259 | 0.8432 | 197 |
| 2 | `l1_ssim` (run1) | 0.07720 | 17 | 20.80 | — | 0.6671 | — |
| 3 | `l1_ssim` (run2) | 0.05895 | 282 | 22.87 | 282 | 0.7718 | 297 |
| 4 | **`roi_weighted`** | **0.01816** | **256** | **30.55** | **259** | 0.8691 | 208 |
| 5 | `l1_ssim_vgg` | 0.06894 | 18 | 21.51 | 66 | 0.7009 | 138 |
| 6 | `amp_l1` | 0.02157 | 258 | 29.67 | 258 | **0.8707** | 300 |
| 7 | `amp_l1_ssim` | 0.04616 | 282 | 24.44 | 282 | 0.8102 | 300 |

### Inference Results (N=400 validation cases, best-MAE checkpoint)

| # | Loss | Results Folder | MAE Mean ± Std | PSNR Mean ± Std | SSIM Mean ± Std |
|---|------|---------------|---------------|-----------------|-----------------|
| 1 | `l1` | `resnet18_l1_noaug` | **0.06706 ± 0.03302** | **23.16 ± 4.52** | **0.8085 ± 0.0464** |
| 2 | `l1_ssim` (run1) | `resnet18_l1_ssim_noaug` | 0.08855 ± 0.13544 | 20.66 ± 3.86 | 0.6893 ± 0.0918 |
| 3 | `l1_ssim` (run2) | `resnet18_l1_ssim_noaug_run2` | 0.08865 ± 0.05840 | 20.63 ± 4.24 | 0.7402 ± 0.0746 |
| 4 | `roi_weighted` | `resnet18_roi_weighted_noaug` | 0.13432 ± 0.05836 | 18.25 ± 4.81 | 0.6260 ± 0.0983 |
| 5 | `l1_ssim_vgg` | `resnet18_l1_ssim_vgg_noaug` | 0.07892 ± 0.06567 | 21.02 ± 3.53 | 0.7098 ± 0.0662 |
| 6 | `amp_l1` | `resnet18_amp_l1_noaug` | 0.08612 ± 0.04298 | 21.95 ± 5.01 | 0.7928 ± 0.0636 |
| 7 | `amp_l1_ssim` | `resnet18_amp_l1_ssim_noaug` | 0.08924 ± 0.06084 | 20.80 ± 4.23 | 0.7566 ± 0.0792 |

### Key Observations

- **`l1`** achieved the best inference MAE (0.067), PSNR (23.16), and SSIM (0.8085) among Phase 1 models — simple and robust.
- **`amp_l1`** was the second-best overall with strong SSIM (0.7928).
- **`roi_weighted`** ranked #1 during training (MAE=0.01816) but dropped significantly during inference (MAE=0.134). The zero-shift post-processing may be poorly suited for this loss.
- The SSIM-based losses (`l1_ssim`, `amp_l1_ssim`) and VGG perceptual loss (`l1_ssim_vgg`) performed substantially worse during training.
- **`l1_ssim` run1** stopped early (epoch 17) with high variance (std=0.135), indicating unstable convergence.

> [!IMPORTANT]
> **Winner by training MAE: `roi_weighted`** — selected for Phase 3 and Phase 4.  
> **Winner by inference MAE: `l1`** — most robust after post-processing.

---

## Phase 2: Best Loss Selection

| Metric | Winner | Value |
|--------|--------|-------|
| Lowest Training MAE | `roi_weighted` | 0.01816 |
| Lowest Inference MAE | `l1` | 0.06706 |

The `roi_weighted` loss (weight=5.0 on non-zero ROI pixels) outperformed all alternatives during training. However, inference results revealed that `l1` is more robust when the zero-shift post-processing is applied.

---

## Phase 3: Augmentation & Standardization (ResNet18)

Both experiments use `roi_weighted` loss with the `resnet18` backbone.

### Training Metrics

| # | Variant | Best MAE | MAE Epoch | Best PSNR (dB) | PSNR Epoch | Best SSIM | SSIM Epoch |
|---|---------|----------|-----------|----------------|------------|-----------|------------|
| 8 | + Data Augmentation | 0.02350 | 186 | 28.81 | 185 | 0.8403 | 235 |
| 9 | **+ Input Standardize** | **0.01705** | **183** | **31.03** | **197** | **0.8743** | **209** |

### Inference Results (N=400 validation cases)

| # | Variant | Results Folder | MAE Mean ± Std | PSNR Mean ± Std | SSIM Mean ± Std |
|---|---------|---------------|---------------|-----------------|-----------------|
| 8 | + Augmentation | `resnet18_roi_weighted_aug` | 0.11614 ± 0.05451 | 19.24 ± 4.66 | 0.6773 ± 0.0952 |
| 9 | **+ Input Std** | `resnet18_roi_weighted_noaug_run2` | **0.07339 ± 0.01976** | **22.50 ± 1.98** | **0.8191 ± 0.0553** |

### Comparison vs Baseline

| Variant | Train MAE | Infer MAE | Infer PSNR | Infer SSIM |
|---------|-----------|-----------|------------|------------|
| Baseline (no aug, no std) | 0.01816 | 0.13432 | 18.25 | 0.6260 |
| + Augmentation | 0.02350 | 0.11614 | 19.24 | 0.6773 |
| **+ Input Standardize** | **0.01705** | **0.07339** | **22.50** | **0.8191** |

### Key Observations

- **Input standardization is the clear winner** — best by both training metrics (MAE=0.01705) and inference metrics (MAE=0.0734, SSIM=0.819).
- **Data augmentation hurt performance** during both training and inference.
- Input standardization reduced inference MAE by **45%** compared to the baseline (0.073 vs 0.134).

---

## Phase 4: Network Architecture Comparison

Tested ResNet34 and ResNet50 backbones with the top 2 loss settings (`roi_weighted` and `amp_l1`).

### Training Metrics (best-MAE checkpoint)

| # | Model | Loss | Std | Best MAE | Epoch | Best PSNR | Best SSIM |
|---|-------|------|-----|----------|-------|-----------|-----------|
| 10 | resnet34 | roi_weighted | ✓ | 0.01683 | 289 | 31.10 | 0.8705 |
| 11 | resnet34 | roi_weighted | — | **0.01595** | 231 | **31.46** | **0.8763** |
| 12 | resnet34 | amp_l1 | — | 0.02191 | 266 | 29.57 | 0.8607 |
| 13 | resnet50 | roi_weighted | ✓ | 0.01646 | 193 | **31.94** | **0.8813** |
| 14 | resnet50 | roi_weighted | — | 0.01806 | 249 | 31.19 | 0.8686 |
| 15 | resnet50 | amp_l1 | — | 0.02104 | 249 | 30.10 | 0.8650 |

### Inference Results (N=400 validation cases, best-MAE checkpoint)

| # | Model | Loss | Std | Results Folder | MAE Mean ± Std | PSNR Mean ± Std | SSIM Mean ± Std |
|---|-------|------|-----|---------------|---------------|-----------------|------------------|
| 10 | resnet34 | roi_weighted | ✓ | `resnet34_roi_weighted_noaug` | 0.08341 ± 0.03305 | 21.57 ± 2.08 | 0.8059 ± 0.0647 |
| 11 | resnet34 | roi_weighted | — | `resnet34_roi_weighted_noaug_run2` | 0.14086 ± 0.06217 | 17.92 ± 4.83 | 0.5950 ± 0.1146 |
| 12 | resnet34 | amp_l1 | — | `resnet34_amp_l1_noaug` | **0.07940 ± 0.03659** | **22.33 ± 4.50** | **0.7985 ± 0.0505** |
| 13 | resnet50 | roi_weighted | ✓ | `resnet50_roi_weighted_noaug` | 0.09443 ± 0.04432 | 20.74 ± 2.47 | 0.8049 ± 0.0872 |
| 14 | resnet50 | roi_weighted | — | `resnet50_roi_weighted_noaug_run2` | 0.13110 ± 0.06037 | 18.68 ± 5.28 | 0.6341 ± 0.1162 |
| 15 | resnet50 | amp_l1 | — | `resnet50_amp_l1_noaug` | 0.08275 ± 0.03972 | 22.16 ± 4.61 | 0.7912 ± 0.0553 |

### Key Observations

- **Input standardization dramatically improves `roi_weighted`** inference: resnet34 MAE drops from 0.141 to 0.083 (−41%), resnet50 from 0.131 to 0.094 (−28%).
- **`amp_l1`** still generalizes well without standardization (MAE ~0.08 for both ResNet34/50).
- **ResNet34 `amp_l1`** produced the best inference MAE (0.0794) among Phase 4 models, closely followed by resnet34 `roi_weighted` + std (0.0834).
- `roi_weighted` without standardization still shows significant training-inference degradation (MAE 0.016 → 0.13–0.14).

---

## Overall Rankings

### By Inference MAE (lower is better)

| Rank | Model | Loss | Std | Infer MAE | Infer PSNR | Infer SSIM |
|------|-------|------|-----|-----------|------------|------------|
| #1 | resnet18 + l1 | l1 | — | **0.06706** | **23.16** | 0.8085 |
| #2 | resnet18 + roi_weighted + std | roi_weighted | ✓ | 0.07339 | 22.50 | **0.8191** |
| #3 | resnet18 + l1_ssim_vgg | l1_ssim_vgg | — | 0.07892 | 21.02 | 0.7098 |
| 4 | resnet34 + amp_l1 | amp_l1 | — | 0.07940 | 22.33 | 0.7985 |
| 5 | resnet50 + amp_l1 | amp_l1 | — | 0.08275 | 22.16 | 0.7912 |
| 6 | resnet34 + roi_weighted + std | roi_weighted | ✓ | 0.08341 | 21.57 | 0.8059 |
| 7 | resnet18 + amp_l1 | amp_l1 | — | 0.08612 | 21.95 | 0.7928 |
| 8 | resnet18 + l1_ssim (run1) | l1_ssim | — | 0.08855 | 20.66 | 0.6893 |
| 9 | resnet18 + l1_ssim (run2) | l1_ssim | — | 0.08865 | 20.63 | 0.7402 |
| 10 | resnet18 + amp_l1_ssim | amp_l1_ssim | — | 0.08924 | 20.80 | 0.7566 |
| 11 | resnet50 + roi_weighted + std | roi_weighted | ✓ | 0.09443 | 20.74 | 0.8049 |
| 12 | resnet18 + roi_weighted + aug | roi_weighted | — | 0.11614 | 19.24 | 0.6773 |
| 13 | resnet50 + roi_weighted | roi_weighted | — | 0.13110 | 18.68 | 0.6341 |
| 14 | resnet18 + roi_weighted | roi_weighted | — | 0.13432 | 18.25 | 0.6260 |
| 15 | resnet34 + roi_weighted | roi_weighted | — | 0.14086 | 17.92 | 0.5950 |

