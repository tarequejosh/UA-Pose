"""RTMPose head with the same per-keypoint signals as UASimCCHead.

RTMPose (RTMCCHead) is a SimCC-style coordinate-classification head with a GAU block, so the
training-free spread applies unchanged. This wrapper adds no trained parameters: it runs the
published head, decodes as usual, and attaches the signals that scripts/evaluate_uncertainty_coco.py
dumps. Head MC-Dropout uses a dropout on the features just before the two coordinate classifiers,
the same place as in UASimCCHead. It is the identity outside MC sampling, so the published
predictions and AP are unchanged.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import InstanceList, OptConfigType, OptSampleList

from .ua_simcc_head import UASimCCHead


@MODELS.register_module()
class UARTMCCHead(RTMCCHead):

    def __init__(self, *args, mc_dropout_rate: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_dropout = nn.Dropout(p=mc_dropout_rate)

    @property
    def split_ratio(self) -> float:
        return float(self.decoder.simcc_split_ratio)

    @property
    def beta(self) -> float:
        """Softmax temperature of the training loss, so signals read the trained distribution.

        KLDiscretLoss compares targets with softmax(beta * logits). RTMPose trains with beta=10,
        the original SimCC models with the default beta=1, so a plain softmax of RTMPose logits
        would be close to uniform. The value comes from the model's own loss, not from data.
        """
        return float(getattr(self.loss_module, 'beta', 1.0))

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        feats = feats[-1]
        feats = self.final_layer(feats)
        feats = torch.flatten(feats, 2)
        feats = self.mlp(feats)
        feats = self.gau(feats)
        feats = self.mc_dropout(feats)
        return self.cls_x(feats), self.cls_y(feats)

    def predict(self, feats, batch_data_samples: OptSampleList,
                test_cfg: OptConfigType = {}) -> InstanceList:
        mc_samples = int(test_cfg.get('mc_samples', 0)) \
            if test_cfg.get('uncertainty_mode') == 'mc_dropout' else 0
        ratio = self.split_ratio

        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _px, _py = self.forward(_feats)
            _px_f, _py_f = flip_vectors(*self.forward(_feats_flip), flip_indices=flip_indices)
            pred_x = (_px + _px_f) * 0.5
            pred_y = (_py + _py_f) * 0.5
            base_feats = _feats
            flip_dist = torch.sqrt(
                ((_px.argmax(-1) - _px_f.argmax(-1)).float() / ratio) ** 2
                + ((_py.argmax(-1) - _py_f.argmax(-1)).float() / ratio) ** 2)
        else:
            pred_x, pred_y = self.forward(feats)
            base_feats = feats
            flip_dist = None

        preds = self.decode((pred_x, pred_y))
        beta = self.beta
        dist_std, dist_entropy, peak_prob = UASimCCHead.compute_simcc_distribution_metrics(
            pred_x * beta, pred_y * beta, ratio)
        if flip_dist is None:
            flip_dist = torch.zeros_like(dist_std)

        mc_std = torch.zeros_like(dist_std)
        mc_soft_std = torch.zeros_like(dist_std)
        if mc_samples > 0:
            was_training = self.mc_dropout.training
            self.mc_dropout.train()
            coords, soft = [], []
            with torch.no_grad():
                for _ in range(mc_samples):
                    px, py = self.forward(base_feats)
                    coords.append(torch.stack([px.argmax(-1).float() / ratio,
                                               py.argmax(-1).float() / ratio], dim=-1))
                    gx = torch.arange(px.size(-1), device=px.device, dtype=px.dtype) / ratio
                    gy = torch.arange(py.size(-1), device=py.device, dtype=py.dtype) / ratio
                    soft.append(torch.stack([(F.softmax(px * beta, -1) * gx).sum(-1),
                                             (F.softmax(py * beta, -1) * gy).sum(-1)], dim=-1))
            self.mc_dropout.train(was_training)
            mc_std = torch.sqrt(torch.stack(coords).var(dim=0).sum(-1) + 1e-8)
            mc_soft_std = torch.sqrt(torch.stack(soft).var(dim=0).sum(-1) + 1e-8)

        fields = dict(uncertainty_dist_std=dist_std, uncertainty_entropy=dist_entropy,
                      uncertainty_mc=mc_std, uncertainty_mc_soft=mc_soft_std,
                      uncertainty_flip=flip_dist, peak_prob=peak_prob)
        fields = {k: to_numpy(v) for k, v in fields.items()}
        for i, pred_inst in enumerate(preds):
            pred_inst.keypoint_uncertainty = fields['uncertainty_dist_std'][i:i + 1]
            for name, values in fields.items():
                setattr(pred_inst, name, values[i:i + 1])
        return preds
