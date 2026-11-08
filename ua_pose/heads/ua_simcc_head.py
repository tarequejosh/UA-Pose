"""Uncertainty-Aware SimCC Head (UA-SimCCHead) with PCT Compositional Mixer.

Combines:
1. Coordinate Classification (SimCC, arXiv:2107.03332)
2. Compositional Sub-Structure Token Mixing (inspired by PCT, arXiv:2303.11638)
3. Single-pass Heteroscedastic Uncertainty + Distributional Entropy/Std
4. Optional Multi-Pass MC-Dropout Epistemic Uncertainty at Inference
"""
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.heads.coord_cls_heads.simcc_head import SimCCHead
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, InstanceList, OptConfigType, OptSampleList


class CompositionalTokenMixer(nn.Module):
    """PCT-Inspired Compositional Sub-Structure Token Mixer (arXiv:2303.11638).

    Instead of treating the K=17 body joints independently, projects the K
    joint representations into M discrete/latent compositional sub-structure
    tokens (encoding anatomical groups such as head, torso, left/right arms,
    left/right legs), models cross-part dependencies via multi-head
    self-attention, and decodes structural context back into the K joint
    embeddings with a learnable Uncertainty-Aware Context Gate.

    Args:
        num_joints (int): Number of body keypoints (default 17 for COCO).
        feat_dim (int): Feature dimension of each flattened joint map.
        token_dim (int): Hidden dimension of compositional tokens.
            Defaults to 128.
        num_tokens (int): Number of compositional sub-structure tokens M.
            Defaults to 8.
        dropout (float): Dropout rate inside the token mixer. Defaults to 0.1.
    """

    # COCO-17 anatomical sub-structure groupings for initializing M=8 tokens:
    # 0: Head/Face (0..4), 1: Shoulders/Upper Torso (5,6), 2: Hips/Pelvis (11,12),
    # 3: Spine/Torso Core (5,6,11,12), 4: Left Arm (5,7,9), 5: Right Arm (6,8,10),
    # 6: Left Leg (11,13,15), 7: Right Leg (12,14,16)
    COCO_PART_GROUPS = (
        (0, 1, 2, 3, 4),
        (5, 6),
        (11, 12),
        (5, 6, 11, 12),
        (5, 7, 9),
        (6, 8, 10),
        (11, 13, 15),
        (12, 14, 16),
    )

    def __init__(
        self,
        num_joints: int,
        feat_dim: int,
        token_dim: int = 128,
        num_tokens: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_tokens = num_tokens

        # Project per-joint spatial features to token space
        self.joint_to_hidden = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, token_dim),
            nn.GELU(),
        )

        # Learnable projection matrix mapping K joints -> M compositional tokens
        self.to_tokens = nn.Linear(num_joints, num_tokens)

        # Self-attention over the M compositional tokens
        self.token_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
            nn.Dropout(dropout),
        )
        self.token_norm2 = nn.LayerNorm(token_dim)

        # Decode M compositional tokens back to K joints
        self.from_tokens = nn.Linear(num_tokens, num_joints)

        # Context Gate: learns per-joint weighting between local visual cue h
        # and global compositional structure struct_ctx (crucial under occlusion)
        self.context_gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.Sigmoid(),
        )

        self.hidden_to_joint = nn.Sequential(
            nn.Linear(token_dim, feat_dim),
            nn.Dropout(dropout),
        )
        self._init_anatomical_priors()

    def _init_anatomical_priors(self) -> None:
        """Initialize token routing weights with anatomical body part priors."""
        if self.num_joints == 17 and self.num_tokens == 8:
            with torch.no_grad():
                prior = torch.zeros(self.num_tokens, self.num_joints)
                for m_idx, group in enumerate(self.COCO_PART_GROUPS):
                    for j_idx in group:
                        prior[m_idx, j_idx] = 1.0 / len(group)
                self.to_tokens.weight.copy_(prior + 0.02 * torch.randn_like(prior))
                self.from_tokens.weight.copy_(
                    prior.t() + 0.02 * torch.randn_like(prior.t()))
        # Zero-init the residual output so the mixer is an exact identity at the
        # start: a pretrained SimCC model keeps its accuracy until training moves it.
        nn.init.zeros_(self.hidden_to_joint[0].weight)
        nn.init.zeros_(self.hidden_to_joint[0].bias)

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x (Tensor): Flattened per-joint features of shape [B, K, D].

        Returns:
            Tuple[Tensor, Tensor]:
                - enhanced_x: Structure-coupled joint features [B, K, D]
                - joint_hidden: Compact joint embeddings [B, K, token_dim]
                  used by the heteroscedastic uncertainty head.
        """
        h = self.joint_to_hidden(x)  # [B, K, token_dim]

        # K joints -> M compositional sub-structure tokens: [B, M, token_dim]
        tokens = self.to_tokens(h.transpose(1, 2)).transpose(1, 2)

        # Cross-token compositional interaction
        attn_out, _ = self.token_attn(tokens, tokens, tokens)
        tokens = self.token_norm(tokens + attn_out)
        tokens = self.token_norm2(tokens + self.token_ffn(tokens))

        # Decode M tokens -> K joints: [B, K, token_dim]
        struct_ctx = self.from_tokens(tokens.transpose(1, 2)).transpose(1, 2)

        # Dynamic gating between local visual feature and compositional context
        gate = self.context_gate(torch.cat([h, struct_ctx], dim=-1))
        joint_hidden = h + gate * struct_ctx

        enhanced_x = x + self.hidden_to_joint(joint_hidden)
        return enhanced_x, joint_hidden


@MODELS.register_module()
class UASimCCHead(SimCCHead):
    """Uncertainty-Aware SimCC Head.

    Extends MMPose's ``SimCCHead`` with:
    1. **PCT-Inspired Compositional Token Mixer**: Couples interdependent
       joints so that uncertainty and coordinate predictions are informed by
       full-body structural context.
    2. **Heteroscedastic Variance + Visibility Head**: Predicts per-joint
       log-variance ``(s_x, s_y)`` and visibility logit ``v_k`` in a single
       forward pass.
    3. **Intrinsic 1D Coordinate Distribution Spread & Entropy**: Computes the
       sub-pixel standard deviation and Shannon entropy of the predicted 1D
       SimCC distributions at zero extra cost.
    4. **Inference-Time MC-Dropout Support**: Can optionally run ``mc_samples``
       passes through the head with dropout active to compare single-pass
       uncertainty against multi-pass MC-Dropout on the exact same backbone.

    Args:
        use_compositional_mixer (bool): Enable the PCT-inspired Compositional
            Token Mixer. Defaults to True.
        token_dim (int): Hidden dimension for the token mixer and uncertainty
            branch. Defaults to 128.
        num_tokens (int): Number of compositional tokens M. Defaults to 8.
        dropout_rate (float): Dropout rate for MC-Dropout and token mixer.
            Defaults to 0.1.
        uncertainty_mode (str): Which uncertainty score to export as the
            primary ``keypoint_uncertainty`` during ``predict()``:
            - ``'combined'``: Geometric/additive fusion of learned
              heteroscedastic sigma and 1D SimCC distributional std.
            - ``'heteroscedastic'``: Learned ``sqrt(sigma_x^2 + sigma_y^2)``.
            - ``'dist_std'``: Training-free 1D SimCC distribution standard
              deviation ``sqrt(Var(p_x) + Var(p_y))``.
            - ``'entropy'``: Shannon entropy ``H(p_x) + H(p_y)``.
            - ``'mc_dropout'``: Coordinate std across ``mc_samples`` stochastic
              forward passes of the head.
            Defaults to ``'combined'``.
        mc_samples (int): Number of stochastic head passes when
            ``uncertainty_mode == 'mc_dropout'``. Defaults to 20.
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        deconv_type: str = 'heatmap',
        deconv_out_channels: Optional[Sequence[int]] = (256, 256, 256),
        deconv_kernel_sizes: Optional[Sequence[int]] = (4, 4, 4),
        deconv_num_groups: Optional[Sequence[int]] = (16, 16, 16),
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
        final_layer: dict = dict(kernel_size=1),
        loss: ConfigType = dict(
            type='HeteroscedasticKLLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        use_compositional_mixer: bool = True,
        token_dim: int = 128,
        num_tokens: int = 8,
        dropout_rate: float = 0.1,
        uncertainty_mode: str = 'combined',
        mc_samples: int = 20,
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            input_size=input_size,
            in_featuremap_size=in_featuremap_size,
            simcc_split_ratio=simcc_split_ratio,
            deconv_type=deconv_type,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            deconv_num_groups=deconv_num_groups,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer=final_layer,
            loss=loss,
            decoder=decoder,
            init_cfg=init_cfg,
        )

        self.use_compositional_mixer = use_compositional_mixer
        self.dropout_rate = dropout_rate
        self.uncertainty_mode = uncertainty_mode
        self.mc_samples = mc_samples

        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]
        self.feat_dropout = nn.Dropout(p=dropout_rate)

        if self.use_compositional_mixer:
            self.token_mixer = CompositionalTokenMixer(
                num_joints=out_channels,
                feat_dim=flatten_dims,
                token_dim=token_dim,
                num_tokens=num_tokens,
                dropout=dropout_rate,
            )
            unc_in_dim = token_dim
        else:
            self.token_mixer = None
            self.unc_proj = nn.Sequential(
                nn.LayerNorm(flatten_dims),
                nn.Linear(flatten_dims, token_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            unc_in_dim = token_dim

        # Heteroscedastic uncertainty head: predicts (s_x, s_y, vis_logit) per joint
        self.uncertainty_head = nn.Sequential(
            nn.Linear(unc_in_dim, unc_in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(unc_in_dim // 2, 3),
        )

    def init_weights(self) -> None:
        """Initialise, then restore the mixer's anatomical prior.

        ``SimCCHead.default_init_cfg`` applies Normal(0, 0.01) to every ``Linear`` in the
        head, including the mixer. Without this override the K-to-M routing prior and
        the zero-init residual are silently destroyed as soon as a Runner starts.
        """
        super().init_weights()
        if self.token_mixer is not None:
            self.token_mixer._init_anatomical_priors()

    def _extract_head_features(self, feats: Tuple[Tensor]) -> Tensor:
        """Run deconv / final_layer to get [B, K, H_hm, W_hm] then flatten."""
        if self.deconv_head is None:
            x = feats[-1]
            if self.final_layer is not None:
                x = self.final_layer(x)
        else:
            x = self.deconv_head(feats)

        # Flatten spatial dimensions: [B, K, H*W]
        return torch.flatten(x, 2)

    def forward_with_uncertainty(
        self, feats: Tuple[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward returning (pred_x, pred_y, log_var, vis_logits)."""
        x = self._extract_head_features(feats)
        x = self.feat_dropout(x)

        if self.token_mixer is not None:
            x, joint_hidden = self.token_mixer(x)
        else:
            joint_hidden = self.unc_proj(x)

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        unc_out = self.uncertainty_head(joint_hidden)  # [B, K, 3]
        log_var = unc_out[..., :2]  # [B, K, 2]
        vis_logits = unc_out[..., 2]  # [B, K]

        return pred_x, pred_y, log_var, vis_logits

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Standard MMPose forward returning (pred_x, pred_y)."""
        pred_x, pred_y, _, _ = self.forward_with_uncertainty(feats)
        return pred_x, pred_y

    @staticmethod
    def compute_simcc_distribution_metrics(
        pred_x: Tensor, pred_y: Tensor, simcc_split_ratio: float = 2.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute 1D SimCC distributional spread, entropy and peak probability.

        Args:
            pred_x (Tensor): Logits along x-axis [B, K, W].
            pred_y (Tensor): Logits along y-axis [B, K, H].
            simcc_split_ratio (float): Bins per input pixel, used so that both axes
                share one unit. Defaults to 2.0.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - dist_std: Distribution std in input pixels [B, K]
                - dist_entropy: Normalized Shannon entropy in [0, 1] [B, K]
                - peak_prob: min(max p_x, max p_y) [B, K]
        """
        px = F.softmax(pred_x, dim=-1)
        py = F.softmax(pred_y, dim=-1)

        w = pred_x.size(-1)
        h = pred_y.size(-1)
        grid_x = torch.arange(w, device=pred_x.device, dtype=pred_x.dtype) / simcc_split_ratio
        grid_y = torch.arange(h, device=pred_y.device, dtype=pred_y.dtype) / simcc_split_ratio

        mu_x = torch.sum(px * grid_x, dim=-1, keepdim=True)
        mu_y = torch.sum(py * grid_y, dim=-1, keepdim=True)

        var_x = torch.sum(px * (grid_x - mu_x)**2, dim=-1)
        var_y = torch.sum(py * (grid_y - mu_y)**2, dim=-1)
        dist_std = torch.sqrt(var_x + var_y + 1e-8)

        # Normalized Shannon entropy in [0, 1]
        ent_x = -torch.sum(px * torch.log(px + 1e-12), dim=-1) / np.log(w)
        ent_y = -torch.sum(py * torch.log(py + 1e-12), dim=-1) / np.log(h)
        dist_entropy = 0.5 * (ent_x + ent_y)

        peak_prob = torch.minimum(px.amax(dim=-1), py.amax(dim=-1))

        return dist_std, dist_entropy, peak_prob

    def _enable_mc_dropout(self) -> None:
        """Enable all head dropout in eval mode.

        ``nn.MultiheadAttention`` applies attention dropout through its own ``training``
        flag rather than a child module, so it has to be switched explicitly.
        """
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.MultiheadAttention)):
                m.train()

    def _sigma_px(self, log_var: Tensor) -> Tensor:
        """Convert predicted log-variance (units of sigma_0) to input pixels."""
        s_x = torch.clamp(log_var[..., 0], min=-8.0, max=6.0)
        s_y = torch.clamp(log_var[..., 1], min=-8.0, max=6.0)
        sigma0_px = float(self.loss_module.base_sigma) / self.simcc_split_ratio \
            if hasattr(self.loss_module, 'base_sigma') else 1.0
        return sigma0_px * torch.sqrt(torch.exp(s_x) + torch.exp(s_y))

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict keypoints and rich uncertainty metrics."""
        mode = test_cfg.get('uncertainty_mode', self.uncertainty_mode)
        mc_samples = test_cfg.get('mc_samples', self.mc_samples)

        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _pred_x, _pred_y, _log_var, _vis = self.forward_with_uncertainty(_feats)
            _pred_x_f, _pred_y_f, _log_var_f, _vis_f = self.forward_with_uncertainty(
                _feats_flip)
            _pred_x_f, _pred_y_f = flip_vectors(
                _pred_x_f, _pred_y_f, flip_indices=flip_indices)
            _log_var_f = _log_var_f[:, flip_indices, :]
            _vis_f = _vis_f[:, flip_indices]

            batch_pred_x = (_pred_x + _pred_x_f) * 0.5
            batch_pred_y = (_pred_y + _pred_y_f) * 0.5
            batch_log_var = (_log_var + _log_var_f) * 0.5
            batch_vis = (_vis + _vis_f) * 0.5
            base_feats_for_mc = _feats
            # Flip disagreement: comes free with flip testing (units: input pixels).
            ratio = self.simcc_split_ratio
            flip_dx = (_pred_x.argmax(-1) - _pred_x_f.argmax(-1)).float() / ratio
            flip_dy = (_pred_y.argmax(-1) - _pred_y_f.argmax(-1)).float() / ratio
            flip_dist = torch.sqrt(flip_dx**2 + flip_dy**2)
        else:
            batch_pred_x, batch_pred_y, batch_log_var, batch_vis = (
                self.forward_with_uncertainty(feats))
            base_feats_for_mc = feats
            flip_dist = None

        preds = self.decode((batch_pred_x, batch_pred_y))

        # 1. Learned heteroscedastic sigma in input pixels
        het_sigma = self._sigma_px(batch_log_var)  # [B, K]
        if flip_dist is None:
            flip_dist = torch.zeros_like(het_sigma)

        # 2. Intrinsic 1D SimCC distribution spread, entropy and peak probability
        dist_std, dist_entropy, peak_prob = self.compute_simcc_distribution_metrics(
            batch_pred_x, batch_pred_y, self.simcc_split_ratio)

        # 3. Optional MC-Dropout multi-pass coordinate std (input pixels). Two decoders:
        # argmax matches the deployed prediction but ties at zero, the soft expectation
        # is continuous. Both are kept so the baseline is not handicapped by a choice.
        if mode == 'mc_dropout':
            was_training = self.training
            self._enable_mc_dropout()
            mc_coords, mc_soft = [], []
            with torch.no_grad():
                for _ in range(mc_samples):
                    px_i, py_i, _, _ = self.forward_with_uncertainty(
                        base_feats_for_mc)
                    ratio = self.simcc_split_ratio
                    cx = torch.argmax(px_i, dim=-1).float() / ratio
                    cy = torch.argmax(py_i, dim=-1).float() / ratio
                    mc_coords.append(torch.stack([cx, cy], dim=-1))
                    grid_x = torch.arange(
                        px_i.size(-1), device=px_i.device, dtype=px_i.dtype) / ratio
                    grid_y = torch.arange(
                        py_i.size(-1), device=py_i.device, dtype=py_i.dtype) / ratio
                    sx = (F.softmax(px_i, dim=-1) * grid_x).sum(-1)
                    sy = (F.softmax(py_i, dim=-1) * grid_y).sum(-1)
                    mc_soft.append(torch.stack([sx, sy], dim=-1))
            self.train(was_training)
            # Stack [T, B, K, 2] -> std across T passes -> norm across (x, y)
            mc_std = torch.sqrt(
                torch.stack(mc_coords, dim=0).var(dim=0).sum(dim=-1) + 1e-8)
            mc_soft_std = torch.sqrt(
                torch.stack(mc_soft, dim=0).var(dim=0).sum(dim=-1) + 1e-8)
        else:
            mc_std = torch.zeros_like(het_sigma)
            mc_soft_std = torch.zeros_like(het_sigma)

        # Select primary uncertainty score
        if mode == 'heteroscedastic':
            primary_unc = het_sigma
        elif mode == 'dist_std':
            primary_unc = dist_std
        elif mode == 'entropy':
            primary_unc = dist_entropy
        elif mode == 'mc_dropout':
            primary_unc = mc_std
        else:  # 'combined'
            primary_unc = 0.5 * het_sigma + 0.5 * dist_std

        primary_unc_np = to_numpy(primary_unc)
        het_sigma_np = to_numpy(het_sigma)
        dist_std_np = to_numpy(dist_std)
        dist_entropy_np = to_numpy(dist_entropy)
        mc_std_np = to_numpy(mc_std)
        mc_soft_np = to_numpy(mc_soft_std)
        flip_np = to_numpy(flip_dist)
        peak_np = to_numpy(peak_prob)
        pred_oks_np = to_numpy(torch.sigmoid(batch_vis))

        for i, pred_inst in enumerate(preds):
            # Shape [1, K] to match pred_inst.keypoint_scores [1, K]
            pred_inst.keypoint_uncertainty = primary_unc_np[i:i + 1]
            pred_inst.uncertainty_het = het_sigma_np[i:i + 1]
            pred_inst.uncertainty_dist_std = dist_std_np[i:i + 1]
            pred_inst.uncertainty_entropy = dist_entropy_np[i:i + 1]
            pred_inst.uncertainty_mc = mc_std_np[i:i + 1]
            pred_inst.uncertainty_mc_soft = mc_soft_np[i:i + 1]
            pred_inst.uncertainty_flip = flip_np[i:i + 1]
            pred_inst.peak_prob = peak_np[i:i + 1]
            pred_inst.predicted_oks = pred_oks_np[i:i + 1]

            # Save raw uncalibrated confidence before optional OKS rescoring
            pred_inst.raw_keypoint_scores = pred_inst.keypoint_scores.copy()

            # Optional OKS-calibrated confidence rescoring (improves COCO AP)
            if test_cfg.get('calibrate_scores', False):
                alpha = float(test_cfg.get('calibration_alpha', 2.0))
                pred_inst.keypoint_scores = pred_inst.keypoint_scores * np.exp(
                    -alpha * primary_unc_np[i:i + 1]
                )

        return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate total SimCC + Heteroscedastic Uncertainty losses."""
        pred_x, pred_y, log_var, vis_logits = self.forward_with_uncertainty(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ], dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ], dim=0)
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ], dim=0)

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        losses = dict()
        loss_out = self.loss_module(
            pred_simcc,
            gt_simcc,
            keypoint_weights,
            log_var=log_var,
            vis_logits=vis_logits,
        )
        if isinstance(loss_out, dict):
            losses.update(loss_out)
        else:
            losses['loss_kpt'] = loss_out

        # Compute training pose accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )
        losses['acc_pose'] = torch.tensor(avg_acc, device=gt_x.device)

        return losses
