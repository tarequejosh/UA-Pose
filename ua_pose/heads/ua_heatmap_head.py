"""Uncertainty-Aware Heatmap Head (UA-HeatmapHead) for PoseResNet & SOCA-PRNet.

Provides:
1. Standard 2D deconvolutional heatmap prediction (SimpleBaseline / PoseResNet)
2. Spatial Dropout2d after each ReLU layer in the deconv head (enabling exact
   replication of the preliminary 20-pass MC-Dropout experiment!)
3. Single-pass Heteroscedastic Uncertainty Branch + 2D Heatmap Spatial Spread
"""
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)


@MODELS.register_module()
class UAHeatmapHead(HeatmapHead):
    """Uncertainty-Aware Heatmap Head.

    Extends MMPose's ``HeatmapHead`` with:
    1. **Deconv Dropout2d**: Inserts ``nn.Dropout2d(p=dropout_rate)`` after each
       deconvolutional ReLU layer so multi-pass MC-Dropout can be run on the
       head during inference (matching the preliminary Keypoint R-CNN setup).
    2. **Single-Pass Heteroscedastic Uncertainty Branch**: A lightweight
       conv + global spatial pooling branch on top of the deconv features that
       predicts per-joint log-variance ``(s_x, s_y)`` in a single pass.
    3. **2D Heatmap Spatial Variance**: Computes the spatial spread of the
       predicted 2D heatmap distribution around its soft-argmax peak.

    Args:
        dropout_rate (float): Spatial Dropout2d probability in deconv layers.
            Defaults to 0.1.
        uncertainty_mode (str): Primary uncertainty score to export:
            - ``'combined'``: 0.5 * heteroscedastic sigma + 0.5 * spatial spread
            - ``'heteroscedastic'``: Learned ``sqrt(sigma_x^2 + sigma_y^2)``
            - ``'spatial_spread'``: 2D soft-argmax spatial standard deviation
            - ``'mc_dropout'``: Coordinate std across ``mc_samples`` stochastic
              forward passes of the head.
            Defaults to ``'combined'``.
        mc_samples (int): Number of stochastic passes when
            ``uncertainty_mode == 'mc_dropout'``. Defaults to 20.
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        deconv_out_channels: Optional[Sequence[int]] = (256, 256, 256),
        deconv_kernel_sizes: Optional[Sequence[int]] = (4, 4, 4),
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
        final_layer: dict = dict(kernel_size=1),
        loss: ConfigType = dict(
            type='HeteroscedasticHeatmapLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        dropout_rate: float = 0.1,
        uncertainty_mode: str = 'combined',
        mc_samples: int = 20,
        init_cfg: OptConfigType = None,
    ) -> None:
        self.dropout_rate = dropout_rate
        self.uncertainty_mode = uncertainty_mode
        self.mc_samples = mc_samples

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer=final_layer,
            loss=loss,
            decoder=decoder,
            init_cfg=init_cfg,
        )

        last_ch = (
            conv_out_channels[-1]
            if conv_out_channels
            else (deconv_out_channels[-1] if deconv_out_channels else in_channels)
        )

        # Single-pass heteroscedastic variance branch predicting [B, K, 2]
        self.uncertainty_branch = nn.Sequential(
            nn.Conv2d(last_ch, last_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(last_ch // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(last_ch // 2, out_channels * 2),
        )

    def _make_deconv_layers(
        self,
        in_channels: int,
        layer_out_channels: Sequence[int],
        layer_kernel_sizes: Sequence[int],
    ) -> nn.Module:
        """Create deconvolutional layers with Dropout2d after each ReLU.

        The ReLU and dropout share one slot so the Sequential keeps HeatmapHead's
        indices (deconv at 0/3/6, BN at 1/4/7). Appending dropout as its own slot shifts
        those keys, and a non-strict load then silently drops pretrained weights.
        """
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            )
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            if self.dropout_rate > 0:
                layers.append(nn.Sequential(
                    nn.ReLU(inplace=True), nn.Dropout2d(p=self.dropout_rate)))
            else:
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward_with_uncertainty(
        self, feats: Tuple[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Forward returning (heatmaps [B, K, H, W], log_var [B, K, 2])."""
        x = feats[-1]
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        heatmaps = self.final_layer(x)
        log_var = self.uncertainty_branch(x).view(
            x.size(0), self.out_channels, 2)
        return heatmaps, log_var

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        heatmaps, _ = self.forward_with_uncertainty(feats)
        return heatmaps

    @staticmethod
    def compute_heatmap_spatial_spread(
        heatmaps: Tensor, beta: float = 20.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute normalized soft-argmax coords, 2D spatial std, and entropy."""
        b, k, h, w = heatmaps.shape
        flat_probs = F.softmax(heatmaps.view(b, k, -1) * beta, dim=-1)
        probs = flat_probs.view(b, k, h, w)

        grid_y = torch.linspace(0.0, 1.0, h, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_x = torch.linspace(0.0, 1.0, w, device=heatmaps.device, dtype=heatmaps.dtype)

        prob_x = probs.sum(dim=2)  # [B, K, W]
        prob_y = probs.sum(dim=3)  # [B, K, H]

        mu_x = torch.sum(prob_x * grid_x, dim=-1)
        mu_y = torch.sum(prob_y * grid_y, dim=-1)

        var_x = torch.sum(prob_x * (grid_x - mu_x.unsqueeze(-1))**2, dim=-1)
        var_y = torch.sum(prob_y * (grid_y - mu_y.unsqueeze(-1))**2, dim=-1)

        coords = torch.stack([mu_x, mu_y], dim=-1)  # [B, K, 2]
        spatial_std = torch.sqrt(var_x + var_y + 1e-8)  # [B, K]
        entropy = -torch.sum(
            flat_probs * torch.log(flat_probs + 1e-12), dim=-1
        ) / float(np.log(h * w))  # [B, K]
        return coords, spatial_std, entropy

    @staticmethod
    def _peak_coords_px(heatmaps: Tensor, stride: float = 4.0) -> Tensor:
        """Integer heatmap peak in input pixels [B, K, 2]."""
        b, k, h, w = heatmaps.shape
        idx = heatmaps.view(b, k, -1).argmax(dim=-1)
        x = (idx % w).float()
        y = torch.div(idx, w, rounding_mode='floor').float()
        return torch.stack([x, y], dim=-1) * stride

    def _peak_distance_px(self, hm_a: Tensor, hm_b: Tensor) -> Tensor:
        """Distance between the peaks of two heatmaps in input pixels [B, K]."""
        stride = float(self.decoder.input_size[0]) / hm_a.shape[-1] \
            if self.decoder is not None else 4.0
        delta = self._peak_coords_px(hm_a, stride) - self._peak_coords_px(hm_b, stride)
        return torch.sqrt((delta**2).sum(dim=-1))

    def _enable_mc_dropout(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def predict(
        self,
        feats: Features,
        batch_data_samples: OptSampleList,
        test_cfg: ConfigType = {},
    ) -> Predictions:
        mode = test_cfg.get('uncertainty_mode', self.uncertainty_mode)
        mc_samples = test_cfg.get('mc_samples', self.mc_samples)

        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _hm, _log_var = self.forward_with_uncertainty(_feats)
            _hm_flip, _log_var_flip = self.forward_with_uncertainty(_feats_flip)
            _hm_flip = flip_heatmaps(
                _hm_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False),
            )
            _log_var_flip = _log_var_flip[:, flip_indices, :]
            batch_heatmaps = (_hm + _hm_flip) * 0.5
            batch_log_var = (_log_var + _log_var_flip) * 0.5
            base_feats_for_mc = _feats
            flip_dist = self._peak_distance_px(_hm, _hm_flip)
        else:
            batch_heatmaps, batch_log_var = self.forward_with_uncertainty(feats)
            base_feats_for_mc = feats
            flip_dist = None

        preds = self.decode(batch_heatmaps)

        s_x = torch.clamp(batch_log_var[..., 0], min=-8.0, max=6.0)
        s_y = torch.clamp(batch_log_var[..., 1], min=-8.0, max=6.0)
        het_sigma = torch.sqrt(torch.exp(s_x) + torch.exp(s_y))  # [B, K]
        if flip_dist is None:
            flip_dist = torch.zeros_like(het_sigma)

        _, spatial_std, dist_entropy = self.compute_heatmap_spatial_spread(
            batch_heatmaps)

        # MC-Dropout std in input pixels: the normalised axes have different lengths.
        h, w = batch_heatmaps.shape[-2:]
        stride = float(self.decoder.input_size[0]) / w if self.decoder is not None else 4.0
        axis_px = batch_heatmaps.new_tensor([(w - 1) * stride, (h - 1) * stride])

        if mode == 'mc_dropout':
            was_training = self.training
            self._enable_mc_dropout()
            mc_coords, mc_peaks = [], []
            with torch.no_grad():
                for _ in range(mc_samples):
                    hm_i = self.forward(base_feats_for_mc)
                    coords_i, _, _ = self.compute_heatmap_spatial_spread(hm_i)
                    mc_coords.append(coords_i * axis_px)
                    mc_peaks.append(self._peak_coords_px(hm_i, stride))
            self.train(was_training)
            mc_soft_std = torch.sqrt(
                torch.stack(mc_coords, dim=0).var(dim=0).sum(dim=-1) + 1e-8)
            mc_std = torch.sqrt(
                torch.stack(mc_peaks, dim=0).var(dim=0).sum(dim=-1) + 1e-8)
        else:
            mc_std = torch.zeros_like(het_sigma)
            mc_soft_std = torch.zeros_like(het_sigma)

        if mode == 'heteroscedastic':
            primary_unc = het_sigma
        elif mode == 'spatial_spread':
            primary_unc = spatial_std
        elif mode == 'entropy':
            primary_unc = dist_entropy
        elif mode == 'mc_dropout':
            primary_unc = mc_std
        else:
            primary_unc = 0.5 * het_sigma + 0.5 * spatial_std

        primary_unc_np = to_numpy(primary_unc)
        het_sigma_np = to_numpy(het_sigma)
        spatial_std_np = to_numpy(spatial_std)
        dist_entropy_np = to_numpy(dist_entropy)
        mc_std_np = to_numpy(mc_std)
        mc_soft_np = to_numpy(mc_soft_std)
        flip_np = to_numpy(flip_dist)

        for i, pred_inst in enumerate(preds):
            pred_inst.keypoint_uncertainty = primary_unc_np[i:i + 1]
            pred_inst.uncertainty_het = het_sigma_np[i:i + 1]
            pred_inst.uncertainty_dist_std = spatial_std_np[i:i + 1]
            pred_inst.uncertainty_entropy = dist_entropy_np[i:i + 1]
            pred_inst.uncertainty_mc = mc_std_np[i:i + 1]
            pred_inst.uncertainty_mc_soft = mc_soft_np[i:i + 1]
            pred_inst.uncertainty_flip = flip_np[i:i + 1]
            pred_inst.raw_keypoint_scores = pred_inst.keypoint_scores.copy()

            if test_cfg.get('calibrate_scores', False):
                alpha = float(test_cfg.get('calibration_alpha', 2.0))
                pred_inst.keypoint_scores = pred_inst.keypoint_scores * np.exp(
                    -alpha * primary_unc_np[i:i + 1]
                )

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: ConfigType = {},
    ) -> dict:
        pred_fields, log_var = self.forward_with_uncertainty(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        losses = dict()
        try:
            loss_out = self.loss_module(
                pred_fields, gt_heatmaps, keypoint_weights, log_var=log_var)
        except TypeError:
            loss_out = self.loss_module(
                pred_fields, gt_heatmaps, keypoint_weights)

        if isinstance(loss_out, dict):
            losses.update(loss_out)
        else:
            losses['loss_kpt'] = loss_out

        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0,
            )
            losses['acc_pose'] = torch.tensor(
                avg_acc, device=gt_heatmaps.device)

        return losses
