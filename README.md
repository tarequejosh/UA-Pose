# Uncertainty-Aware Lightweight Pose Estimation (`UA-PoseEstimation`)

A complete research framework built on **[OpenMMLab MMPose](https://github.com/open-mmlab/mmpose)** for **Uncertainty-Aware Lightweight 2D Human Pose Estimation**, tailored to run out-of-the-box on **Windows 11 (NVIDIA RTX 5060 + Intel Core Ultra i7)**.

---

## 1. Research Overview & Architecture

Building on preliminary findings with training-free MC-Dropout on Keypoint R-CNN (where 20 stochastic head passes showed **$2.55\times$ higher uncertainty on occluded joints** and **Spearman $\rho = 0.670$** correlation with localization error), this repository extends lightweight pose estimators (`PoseResNet`, `SOCA-PRNet`, and `ViPNAS-MobileNetV3`) with **single-pass and multi-pass uncertainty quantification** inspired by:

1. **[SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation](https://arxiv.org/pdf/2107.03332) (`arXiv:2107.03332`)**:
   - Replaces heavy 2D deconvolutional heatmaps with 1D sub-pixel coordinate classification vectors ($x, y$).
   - **Intrinsic Distributional Uncertainty (`dist_std` & `entropy`)**: Extracts the sub-pixel variance and Shannon entropy of the 1D coordinate distributions in a **single forward pass at zero extra compute**.
   - **Trained Heteroscedastic SimCC Head (`UASimCCHead` + `HeteroscedasticKLLoss`)**: Predicts per-joint log-variance $(s_{x,k}, s_{y,k})$ and visibility probability $\hat{v}_k$, trained with heteroscedastic KL attenuation and Laplace coordinate residual calibration.
2. **[Human Pose as Compositional Tokens (PCT)](https://arxiv.org/pdf/2303.11638) (`arXiv:2303.11638`)**:
   - **Compositional Sub-Structure Token Mixer (`CompositionalTokenMixer`)**: Instead of predicting joint coordinates and uncertainties in isolation, projects the $K=17$ joints into $M=8$ compositional sub-structure tokens, models cross-limb anatomical dependencies via multi-head self-attention, and decodes structural context back into each joint before predicting coordinates and uncertainty.
3. **SOCA-PRNet Backbone (`SOCAPRNet`)**:
   - Extends `PoseResNet-34` with **Global Context Blocks (GCB)** and **Spatially Oriented Contextual Attention (SOCA)** across residual stages while remaining 100% weight-compatible with `torchvision://resnet34`.
4. **Uncertainty-Aware Heatmap Head (`UAHeatmapHead`)**:
   - Extends `HeatmapHead` (`PoseResNet-50` / `SOCA-PRNet-34`) with both a single-pass heteroscedastic variance branch, 2D soft-argmax spatial spread, and `Dropout2d(p=0.1)` after every deconv ReLU layer to allow direct **20-pass MC-Dropout** comparison on the exact same backbone.

---

## 2. Repository Structure

```text
UA-PoseEstimation/
├── mmpose/                         # Bundled OpenMMLab MMPose framework
├── ua_pose/                        # Custom Uncertainty-Aware MMPose plugin package
│   ├── backbones/
│   │   └── soca_prnet.py           # SOCAPRNet, GlobalContextBlock, SOCA module
│   ├── heads/
│   │   ├── ua_simcc_head.py        # UASimCCHead + PCT CompositionalTokenMixer
│   │   └── ua_heatmap_head.py      # UAHeatmapHead (Heteroscedastic + Deconv Dropout2d)
│   ├── losses/
│   │   └── uncertainty_losses.py   # HeteroscedasticKLLoss & HeteroscedasticHeatmapLoss
│   └── evaluation/
│       └── uncertainty_metric.py   # UncertaintyPoseMetric (PCK@0.1, Spearman rho, AUSE, AUROC)
├── configs/
│   ├── simcc/
│   │   ├── ua_simcc_soca_prnet34_coco_256x192.py
│   │   ├── ua_simcc_res34_coco_256x192.py
│   │   ├── ua_simcc_res50_coco_256x192.py
│   │   ├── ua_simcc_res50_finetune_20e_coco_256x192.py
│   │   ├── ua_simcc_vipnas_mbv3_coco_256x192.py
│   │   └── ua_simcc_vipnas_mbv3_finetune_20e_coco_256x192.py
│   └── heatmap/
│       ├── ua_heatmap_soca_prnet34_coco_256x192.py
│       ├── ua_heatmap_res50_coco_256x192.py
│       └── ua_heatmap_res50_finetune_20e_coco_256x192.py
├── scripts/
│   ├── download_coco_and_weights.py  # Downloads COCO 2017 & pretrained SimCC/Heatmap weights
│   ├── evaluate_uncertainty_coco.py  # Evaluates on 500 COCO val2017 images & plots Figures 1, 2, 3
│   └── train.py                      # Single-GPU AMP training / fine-tuning launcher
├── tests/
│   └── smoke_test.py                 # Verifies RTX 5060 CUDA, models, losses, and metrics
├── setup_windows_rtx5060.ps1         # Automated Windows 11 + RTX 5060 (CUDA 12.8) installer
└── run_experiments_windows.bat       # One-click Windows batch runner for all experiments
```

---

## 3. Quick Start on Windows 11 (NVIDIA RTX 5060 + Core Ultra i7)

### Step 1: Clone & Run the Automated Windows Installer
Open **PowerShell** or **Anaconda PowerShell Prompt** on your Windows PC:

```powershell
git clone <YOUR_GITHUB_REPO_URL> UA-PoseEstimation
cd UA-PoseEstimation

# Creates 'ua-pose' environment, installs PyTorch (CUDA 12.8 for RTX 5060 Blackwell),
# installs MMEngine/MMCV/MMPose, and runs tests\smoke_test.py
powershell -ExecutionPolicy Bypass -File .\setup_windows_rtx5060.ps1
```

### Step 2: Download Pretrained Checkpoints & COCO 2017 Dataset
```cmd
conda activate ua-pose

REM Download pretrained SimCC/Heatmap checkpoints + COCO val2017 (~1 GB)
python scripts\download_coco_and_weights.py

REM If you also need COCO train2017 (~18 GB) for training/fine-tuning:
python scripts\download_coco_and_weights.py --include-train2017
```

### Step 3: Run the 500-Image COCO Uncertainty Benchmark & Generate Figures 1, 2, 3
To immediately evaluate **SimCC (`ResNet-50` & `ViPNAS-MobileNetV3`)** and **PoseResNet Heatmap (`ResNet-50`)** on 500 randomly sampled COCO val2017 images (comparing Peak Confidence, Single-Pass 1D/2D Distribution Spread, Entropy, and 20-Pass MC-Dropout):

```cmd
.\run_experiments_windows.bat eval
```

This outputs:
- `work_dirs\eval_simcc_res50_500imgs\uncertainty_summary.json`
- `work_dirs\eval_simcc_res50_500imgs\figure1_predicted_skeletons_uncertainty.png`
- `work_dirs\eval_simcc_res50_500imgs\figure2_occlusion_and_pck_curves.png`
- `work_dirs\eval_simcc_res50_500imgs\figure3_quintiles_and_error_correlation.png`

### Step 4: Fine-Tune or Train Uncertainty-Aware Heads (Single-Pass Heteroscedastic + PCT Tokens)
```cmd
REM Fast 20-epoch fine-tuning from COCO-pretrained SimCC checkpoints (uses AMP fp16 on RTX 5060):
.\run_experiments_windows.bat finetune

REM Or train SOCA-PRNet-34 + UA-SimCCHead + PCT Compositional Tokens from scratch:
.\run_experiments_windows.bat train_soca
```
