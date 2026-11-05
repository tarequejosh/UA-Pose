"""Uncertainty-Aware Losses for Pose Estimation (SimCC & Heatmap)."""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS

# Official COCO 17-joint OKS falloff constants kappa_k
COCO_OKS_SIGMAS = (
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
)


@MODELS.register_module()
class HeteroscedasticKLLoss(nn.Module):
    """Heteroscedastic Discrete KL Divergence + Variance & OKS Calibration Loss.

    Extends SimCC's ``KLDiscretLoss`` (arXiv:2107.03332) with:
    1. **Adaptive Variance-Conditioned Target Smoothing (RTMO / Dynamic-SimCC)**:
       Optionally constructs dynamic 1D Gaussian soft targets whose width
       ``sigma_dyn = sigma_base * exp(0.5 * s_k)`` adapts to the network's
       predicted per-joint log-variance ``s_k``, preventing overconfident sharp
       supervision on occluded or ambiguous joints.
    2. **Heteroscedastic NLL Attenuation (Kendall & Gal, CVPR 2017)**:
       ``L_het = 0.5 * exp(-s_x) * D_x + 0.5 * s_x + 0.5 * exp(-s_y) * D_y + 0.5 * s_y``
    3. **Direct Laplace Coordinate Residual Calibration**:
       Forces ``exp(0.5 * s_k)`` to track the detached sub-pixel L1 error
       ``|hat_mu_k - mu_gt_k|``.
    4. **OKS Confidence Calibration Loss (inspired by Gu et al., ICML 2024)**:
       Supervises the predicted OKS/visibility logit ``vis_logits`` directly
       against the true per-joint Object Keypoint Similarity
       ``OKS_k = exp(-d_k^2 / (2 * s_box^2 * kappa_k^2))`` so that keypoint
       confidence is strictly aligned with the COCO AP evaluation metric.

    Args:
        beta (float): Temperature factor for SimCC Softmax. Defaults to 1.0.
        label_softmax (bool): Whether to apply Softmax to target labels.
            Defaults to False.
        label_beta (float): Temperature for label Softmax. Defaults to 10.0.
        use_target_weight (bool): Weight loss by per-joint visibility weights.
            Defaults to True.
        adaptive_target_sigma (bool): Dynamically widen target 1D Gaussian
            labels using predicted ``exp(0.5 * s_k)``. Defaults to True.
        base_sigma (float): Base SimCC Gaussian std in bins. Defaults to 6.0.
        het_weight (float): Weight for heteroscedastic KL attenuation term.
            Defaults to 0.5.
        cal_weight (float): Weight for direct error-variance calibration term.
            Defaults to 0.2.
        vis_weight (float): Weight for OKS-aligned confidence/visibility loss.
            Defaults to 0.2.
    """

    def __init__(
        self,
        beta: float = 1.0,
        label_softmax: bool = False,
        label_beta: float = 10.0,
        use_target_weight: bool = True,
        adaptive_target_sigma: bool = True,
        base_sigma: float = 6.0,
        het_weight: float = 0.5,
        cal_weight: float = 0.2,
        vis_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.adaptive_target_sigma = adaptive_target_sigma
        self.base_sigma = base_sigma
        self.het_weight = het_weight
        self.cal_weight = cal_weight
        self.vis_weight = vis_weight

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.register_buffer(
            'oks_sigmas',
            torch.tensor(COCO_OKS_SIGMAS, dtype=torch.float32),
            persistent=False,
        )

    def _per_joint_kl(self, dec_outs: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """Compute mean KL divergence per instance and joint: shape [N, K]."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            labels = F.softmax(labels * self.label_beta, dim=-1)
        kl = torch.mean(self.kl_loss(log_pt, labels), dim=-1)
        return kl

    def _build_adaptive_gaussian_target(
        self,
        gt_bins: torch.Tensor,
        log_var_axis: torch.Tensor,
    ) -> torch.Tensor:
        """Construct 1D Gaussian target with adaptive width sigma_0 * exp(0.5*s).

        Args:
            gt_bins (Tensor): Static SimCC target [N, K, B].
            log_var_axis (Tensor): Clamped log-variance [N, K].
        """
        num_bins = gt_bins.size(-1)
        mu_idx = torch.argmax(gt_bins, dim=-1, keepdim=True).float()  # [N, K, 1]
        grid = torch.arange(
            num_bins, device=gt_bins.device, dtype=gt_bins.dtype).view(1, 1, -1)

        # Clamp dynamic scale multiplier to [0.7, 2.5] around base_sigma
        scale_mult = torch.exp(0.5 * log_var_axis.detach()).clamp(min=0.7, max=2.5)
        sigma = (self.base_sigma * scale_mult).unsqueeze(-1)  # [N, K, 1]

        target = torch.exp(-((grid - mu_idx)**2) / (2.0 * sigma**2))
        norm = sigma * np.sqrt(2.0 * np.pi)
        return target / (norm + 1e-8)

    @staticmethod
    def _expected_coord(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute soft-argmax expected bin coordinate and distribution std."""
        probs = F.softmax(logits, dim=-1)
        num_bins = logits.size(-1)
        grid = torch.linspace(
            0.0, 1.0, num_bins, device=logits.device, dtype=logits.dtype)
        mean_coord = torch.sum(probs * grid, dim=-1)  # [N, K]
        var_coord = torch.sum(
            probs * (grid - mean_coord.unsqueeze(-1))**2, dim=-1)
        return mean_coord, torch.sqrt(var_coord + 1e-6)

    def forward(
        self,
        pred_simcc: Tuple[torch.Tensor, torch.Tensor],
        gt_simcc: Tuple[torch.Tensor, torch.Tensor],
        target_weight: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
        vis_logits: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute total and component losses."""
        pred_x, pred_y = pred_simcc
        gt_x, gt_y = gt_simcc
        n, k, _ = pred_x.shape

        if self.use_target_weight and target_weight is not None:
            weight = target_weight.view(n, k).float()
        else:
            weight = torch.ones((n, k), device=pred_x.device, dtype=pred_x.dtype)

        if log_var is not None:
            s_x = torch.clamp(log_var[..., 0], min=-6.0, max=4.0)
            s_y = torch.clamp(log_var[..., 1], min=-6.0, max=4.0)
        else:
            s_x = None
            s_y = None

        # Use adaptive variance-conditioned Gaussian smoothing if enabled
        if self.adaptive_target_sigma and s_x is not None and s_y is not None:
            target_x = 0.5 * gt_x + 0.5 * self._build_adaptive_gaussian_target(gt_x, s_x)
            target_y = 0.5 * gt_y + 0.5 * self._build_adaptive_gaussian_target(gt_y, s_y)
        else:
            target_x, target_y = gt_x, gt_y

        kl_x = self._per_joint_kl(pred_x, target_x)  # [N, K]
        kl_y = self._per_joint_kl(pred_y, target_y)  # [N, K]

        # Base SimCC KL loss (matches standard KLDiscretLoss scaling)
        base_loss = ((kl_x + kl_y) * weight).sum() / k
        losses = dict(loss_kpt=base_loss)

        pred_mu_x, _ = self._expected_coord(pred_x)
        pred_mu_y, _ = self._expected_coord(pred_y)
        gt_mu_x = torch.argmax(gt_x, dim=-1).float() / max(gt_x.size(-1) - 1, 1)
        gt_mu_y = torch.argmax(gt_y, dim=-1).float() / max(gt_y.size(-1) - 1, 1)

        err_x = torch.abs(pred_mu_x.detach() - gt_mu_x).clamp(min=1e-4)
        err_y = torch.abs(pred_mu_y.detach() - gt_mu_y).clamp(min=1e-4)

        if s_x is not None and s_y is not None:
            # 1. Heteroscedastic KL attenuation loss
            d_x = kl_x * pred_x.size(-1)
            d_y = kl_y * pred_y.size(-1)
            het_x = 0.5 * torch.exp(-s_x) * d_x + 0.5 * s_x
            het_y = 0.5 * torch.exp(-s_y) * d_y + 0.5 * s_y
            loss_het = ((het_x + het_y) * weight).mean() * self.het_weight

            # 2. Direct coordinate error calibration loss (Laplace NLL)
            pred_sigma_x = torch.exp(0.5 * s_x)
            pred_sigma_y = torch.exp(0.5 * s_y)

            cal_x = (err_x / pred_sigma_x + 0.5 * s_x)
            cal_y = (err_y / pred_sigma_y + 0.5 * s_y)
            loss_cal = ((cal_x + cal_y) * weight).mean() * self.cal_weight

            losses['loss_het'] = loss_het
            losses['loss_cal'] = loss_cal

        if vis_logits is not None:
            # 3. OKS-Aligned Confidence Calibration Loss (Gu et al., ICML 2024)
            # Compute per-joint OKS in normalized crop coordinates
            sq_dist = err_x**2 + err_y**2
            if k == len(COCO_OKS_SIGMAS):
                kappa = self.oks_sigmas.to(pred_x.device).view(1, k)
            else:
                kappa = torch.full((1, k), 0.05, device=pred_x.device)
            # In normalized [0, 1] person box space, box area s^2 ~ 0.53
            oks_target = torch.exp(-sq_dist / (2.0 * 0.53 * (kappa**2))) * (weight > 0).float()
            loss_vis = F.binary_cross_entropy_with_logits(
                vis_logits, oks_target.detach()) * self.vis_weight
            losses['loss_oks_cal'] = loss_vis

        return losses


@MODELS.register_module()
class HeteroscedasticHeatmapLoss(nn.Module):
    """Heteroscedastic Heatmap MSE + Coordinate Error Calibration Loss.

    Combines standard ``KeypointMSELoss`` on 2D Gaussian heatmaps with a
    learned per-joint log-variance head trained via Laplace NLL on the
    soft-argmax vs. ground-truth heatmap peaks.

    Args:
        use_target_weight (bool): Use per-joint target weights. Defaults to True.
        loss_weight (float): Base heatmap MSE weight. Defaults to 1.0.
        het_weight (float): Weight for heteroscedastic heatmap residual NLL.
            Defaults to 0.2.
        cal_weight (float): Weight for soft-argmax coordinate error calibration.
            Defaults to 0.2.
    """

    def __init__(
        self,
        use_target_weight: bool = True,
        loss_weight: float = 1.0,
        het_weight: float = 0.2,
        cal_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.het_weight = het_weight
        self.cal_weight = cal_weight

    @staticmethod
    def _heatmap_soft_argmax(heatmaps: torch.Tensor,
                             beta: float = 20.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract normalized (x, y) coordinates in [0, 1] and spatial spread."""
        b, k, h, w = heatmaps.shape
        flat = heatmaps.view(b, k, -1) * beta
        probs = F.softmax(flat, dim=-1).view(b, k, h, w)

        grid_y = torch.linspace(0.0, 1.0, h, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_x = torch.linspace(0.0, 1.0, w, device=heatmaps.device, dtype=heatmaps.dtype)

        prob_x = probs.sum(dim=2)  # [B, K, W]
        prob_y = probs.sum(dim=3)  # [B, K, H]

        mu_x = torch.sum(prob_x * grid_x, dim=-1)
        mu_y = torch.sum(prob_y * grid_y, dim=-1)

        var_x = torch.sum(prob_x * (grid_x - mu_x.unsqueeze(-1))**2, dim=-1)
        var_y = torch.sum(prob_y * (grid_y - mu_y.unsqueeze(-1))**2, dim=-1)

        coords = torch.stack([mu_x, mu_y], dim=-1)  # [B, K, 2]
        spread = torch.sqrt(var_x + var_y + 1e-6)  # [B, K]
        return coords, spread

    def forward(
        self,
        pred_hm: torch.Tensor,
        gt_hm: torch.Tensor,
        target_weight: Optional[torch.Tensor] = None,
        log_var: Optional[torch.Tensor] = None,
    ) -> dict:
        b, k, h, w = pred_hm.shape

        if self.use_target_weight and target_weight is not None:
            weight = target_weight.view(b, k).float()
        else:
            weight = torch.ones((b, k), device=pred_hm.device, dtype=pred_hm.dtype)

        # Per-joint MSE: [B, K]
        mse_per_joint = F.mse_loss(
            pred_hm, gt_hm, reduction='none').mean(dim=(-2, -1))
        base_loss = 0.5 * (mse_per_joint * weight).mean() * self.loss_weight

        losses = dict(loss_kpt=base_loss)

        if log_var is not None:
            s_x = torch.clamp(log_var[..., 0], min=-6.0, max=4.0)
            s_y = torch.clamp(log_var[..., 1], min=-6.0, max=4.0)
            s_joint = 0.5 * (s_x + s_y)

            # 1. Heteroscedastic MSE attenuation
            het_loss = (
                0.5 * torch.exp(-s_joint) * (mse_per_joint * 100.0) +
                0.5 * s_joint)
            losses['loss_het'] = (het_loss * weight).mean() * self.het_weight

            # 2. Coordinate error calibration
            pred_coords, _ = self._heatmap_soft_argmax(pred_hm.detach())
            gt_coords, _ = self._heatmap_soft_argmax(gt_hm.detach(), beta=50.0)

            err_x = torch.abs(pred_coords[..., 0] - gt_coords[..., 0]).clamp(min=1e-4)
            err_y = torch.abs(pred_coords[..., 1] - gt_coords[..., 1]).clamp(min=1e-4)

            sigma_x = torch.exp(0.5 * s_x)
            sigma_y = torch.exp(0.5 * s_y)

            cal = (err_x / sigma_x + 0.5 * s_x) + (err_y / sigma_y + 0.5 * s_y)
            losses['loss_cal'] = (cal * weight).mean() * self.cal_weight

        return losses
