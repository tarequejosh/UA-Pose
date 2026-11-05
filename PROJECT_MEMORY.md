# Project Memory: Uncertainty-Aware Lightweight Pose Estimation (`UA-PoseEstimation`)

> **Purpose of this file**: Persistent research memory, literature review synthesis, theoretical formulation, architectural decisions, and Q1 journal execution strategy stored directly inside the repository so it travels with the project across machines (Mac $\to$ GitHub $\to$ Windows RTX 5060 PC).

---

## 1. Research Objective & Preliminary Baseline

### Core Objective
Improve the reliability and accuracy of lightweight 2D human pose estimation models (`PoseResNet`, `SOCA-PRNet`, `SimCC`, `ViPNAS-MobileNetV3`) by estimating per-joint uncertainty—identifying when predicted keypoints are inaccurate, especially under **occlusion**—within a **single forward pass**.

### Preliminary Baseline (Keypoint R-CNN + 20-Pass MC-Dropout)
- **Model**: Pretrained `torchvision` Keypoint R-CNN (`ResNet-50-FPN`), training-free.
- **Method**: `Dropout2d(p=0.1)` inserted after each of the 8 ReLU layers in the keypoint head, kept active at inference for $T=20$ passes per detected person box.
- **Dataset**: 500 randomly sampled COCO `val2017` images ($1,310$ matched persons; $14,331$ labeled keypoints: $12,650$ visible `v=2`, $1,681$ occluded `v=1`).
- **Key Empirical Findings**:
  1. MC-Dropout uncertainty is **$2.55\times$ higher** on occluded joints than visible joints.
  2. Uncertainty correlates strongly with normalized localization error ($d / \sqrt{\text{box area}}$), achieving **Spearman $\rho = 0.670$**.
  3. Filtering uncertain keypoints monotonically improves `PCK@0.1` accuracy curves (Figure 2b).
  4. **Bottleneck**: Running 20 head passes takes $\approx 0.22\text{ s/img}$ ($111\text{ s}$ for 500 images on an RTX 5060) and only captures uncalibrated epistemic noise, motivating **single-pass, training-time uncertainty-aware heads**.

---

## 2. Comprehensive Literature Review (2021–2026) & Gap Analysis

| Paper & Venue | Core Representation | How Uncertainty / Occlusion is Addressed | Key Gap Exploited by Our Work |
| :--- | :--- | :--- | :--- |
| **RLE** (*Li et al., ICCV 2021*)<br>*"Human Pose Regression with Residual Log-likelihood Estimation"* | Direct 2D/3D Coordinate Regression + Normalizing Flows (`RealNVP`) | Learns residual density $p(\mathbf{y}\mid\mathbf{x})$ and per-joint scale $\hat{\sigma}$. | Applies only to direct regression (no spatial bin grid); predicts each joint's $\hat{\sigma}$ **independently** without skeletal token coupling. |
| **SimCC** (*Li et al., ECCV 2022*)<br>`arXiv:2107.03332` | 1D Coordinate Classification ($x, y$ sub-pixel bins) | Uses **fixed-width** Gaussian target smoothing ($\sigma_0 = 6.0$) with KL divergence. | **Uncertainty-blind**: forces identical sharp supervision (`sigma=6.0`) whether a joint is visible or $100\%$ occluded. |
| **PCT** (*Geng et al., CVPR 2023*)<br>`arXiv:2303.11638` | Compositional Token Classification ($M$ VQ-VAE tokens) | Models correlated joint sub-structures via discrete tokens to infer occluded joints from visible context. | Heavy 2-stage VQ-VAE pipeline; outputs **deterministic poses with zero uncertainty estimation or error calibration**. |
| **Pose Calibration / CCNet** (*Gu et al., ICML 2024*)<br>`arXiv:2311.17105` | Post-hoc Confidence Calibration on Frozen Backbones | Proves raw heatmap peaks are miscalibrated wrt OKS; trains a post-hoc MLP to align confidence with OKS. | **Post-hoc patch only**: does not regularize coordinate learning during training or capture directional $(x,y)$ uncertainty. |
| **RTMO** (*Lu et al., CVPR 2024*)<br>`arXiv:2312.07526` | One-Stage YOLO + Dual 1D Coordinate Classification | Uses Maximum Likelihood Estimation (MLE) with learnable $\hat{\sigma}$ to weight hard vs. easy samples. | Treats $\hat{\sigma}$ purely as a **training loss weight** on independent joints; never evaluates $\hat{\sigma}$ for **occlusion AUROC, Spearman $\rho$, or risk-coverage AUSE**. |
| **SOCA-PRNet** (*Ali et al., Sensors 2024*) | Lightweight `PoseResNet-34` + SOCA + Global Context Blocks | Uses directional coordinate attention (SOCA) + Global Context Blocks (GCB) for heatmap pose accuracy. | **Deterministic heatmap model**: lacks coordinate classification efficiency (`SimCC`) and has no uncertainty head. |

---

## 3. Core Technical Novelties Implemented in `ua_pose`

To ensure high novelty suitable for a **Q1 Journal** (*Pattern Recognition*, *IEEE TCSVT*, *Engineering Applications of AI*, *CVIU*), we designed and implemented **four synergistic contributions**:

### 1. Anatomical-Prior Compositional Token Mixer (`CompositionalTokenMixer`)
- **File**: `ua_pose/heads/ua_simcc_head.py`
- **Concept**: Inspired by PCT (`arXiv:2303.11638`), instead of predicting coordinates and uncertainty for each of the $K=17$ joints in isolation, we route the $K$ joint features into $M=8$ **anatomical sub-structure tokens** (initialized with human body part priors: *Head, Upper Torso, Pelvis, Torso Core, Left Arm, Right Arm, Left Leg, Right Leg*), apply Multi-Head Self-Attention across tokens, and fuse the structural context back into each joint via a **learnable Sigmoid Context Gate**.
- **Why It Matters**: Enables the uncertainty head to distinguish **constrained occlusion** (e.g., an occluded wrist whose position is geometrically locked by a visible elbow/shoulder $\to$ low error) from **unconstrained occlusion** (entire limb hidden $\to$ high error).

### 2. Dual-Source Single-Pass SimCC Uncertainty (`UASimCCHead`)
- **File**: `ua_pose/heads/ua_simcc_head.py`
- **Concept**: Combines two complementary single-pass uncertainty signals:
  1. **Non-Parametric 1D Distributional Spread (`dist_std`) & Shannon Entropy (`entropy`)**: Computed directly from the 1D softmax coordinate distributions $p_x, p_y$:
     $$\sigma_{\text{dist}} = \sqrt{\sum_i p_x(i)(x_i - \hat{\mu}_x)^2 + \sum_j p_y(j)(y_j - \hat{\mu}_y)^2}$$
     Requires **zero extra parameters** and works even on off-the-shelf pretrained SimCC checkpoints!
  2. **Parametric Heteroscedastic Scale (`het_sigma`)**: Predicted by the compositional uncertainty branch $(s_x, s_y) = (\log \hat{\sigma}_x^2, \log \hat{\sigma}_y^2)$.

### 3. Adaptive Variance-Conditioned SimCC + OKS Calibration Loss (`HeteroscedasticKLLoss`)
- **File**: `ua_pose/losses/uncertainty_losses.py`
- **Concept**: Upgrades standard `KLDiscretLoss` with four principled terms:
  1. **Adaptive Target Gaussian Smoothing**: Dynamically adjusts the ground-truth 1D Gaussian target width $\sigma_{\text{dyn}} = \sigma_0 \cdot \exp(0.5 s_k)$ so occluded/ambiguous joints are not penalized by an artificially sharp target.
  2. **Heteroscedastic KL Attenuation**: $\frac{1}{2}\exp(-s_k) D_{\text{KL}} + \frac{1}{2}s_k$.
  3. **Laplace Residual Error Calibration**: Directly aligns $\hat{\sigma}_k = \exp(0.5 s_k)$ with the detached sub-pixel $L_1$ localization residual $|\hat{\mu}_k - \mu_{gt,k}|$.
  4. **OKS-Aligned Confidence Loss**: Supervises the visibility/confidence logit against true per-joint Object Keypoint Similarity $\text{OKS}_k = \exp\big(-d_k^2 / (2 s_{\text{box}}^2 \kappa_k^2)\big)$.

### 4. Uncertainty-Guided OKS Confidence Rescoring (`calibrate_scores=True`)
- **Files**: `ua_pose/heads/ua_simcc_head.py` & `ua_pose/heads/ua_heatmap_head.py`
- **Concept**: At inference, rescales raw peak confidence $c_k$ using predicted uncertainty $\hat{u}_k$:
  $$\tilde{c}_k = c_k \cdot \exp(-\alpha \cdot \hat{u}_k)$$
  This directly reduces **Expected Calibration Error (ECE)** and boosts **COCO OKS Average Precision (AP)**.

---

## 4. Experimental Roadmap & Q1 Paper Tables

When running on the **Windows 11 PC (NVIDIA RTX 5060 + Core Ultra i7)**:

1. **Zero-Shot Pretrained Discovery (`run_experiments_windows.bat eval`)**:
   - Evaluates official COCO-pretrained `SimCC-ResNet50`, `SimCC-ViPNAS-MobileNetV3`, and `Heatmap-PoseResNet50` on the 500-image COCO `val2017` subset.
   - Compares `Baseline Peak Conf` vs. `Single-Pass 1D Dist Std` vs. `Single-Pass Entropy` vs. `20-Pass MC-Dropout`.
   - Automatically generates `figure1_predicted_skeletons_uncertainty.png`, `figure2_occlusion_and_pck_curves.png`, `figure3_quintiles_and_error_correlation.png`, and `q1_paper_table.tex`.
2. **Fast Fine-Tuning (`run_experiments_windows.bat finetune`)**:
   - Fine-tunes `UA-SimCC + CompositionalTokenMixer + HeteroscedasticKLLoss` for 20 epochs from the COCO-pretrained checkpoints using AMP (`fp16`) on RTX 5060 (~20–40 mins).
   - Demonstrates that adding trained heteroscedastic + compositional uncertainty pushes **Spearman $\rho$**, **Occlusion AUROC**, **AUSE**, and **ECE** to SOTA levels in a single forward pass.
3. **Full Lightweight Backbone Training (`run_experiments_windows.bat train_soca`)**:
   - Trains `SOCA-PRNet-34 + UA-SimCCHead` to establish our flagship lightweight uncertainty-aware pose estimator.
