"""Uncertainty-Aware Heatmap + PoseResNet-50 Config.

Can load official MMPose SimpleBaseline ResNet-50 pretrained weights directly
to evaluate 20-pass MC-Dropout or Spatial Spread without training, or fine-tune
with the HeteroscedasticHeatmapLoss.
"""
_base_ = ['./ua_heatmap_soca_prnet34_coco_256x192.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    head=dict(
        in_channels=2048,
    ),
)
