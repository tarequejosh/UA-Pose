"""Uncertainty-Aware SimCC + PoseResNet-50 Config.

Compatible with official MMPose SimCC ResNet-50 pretrained weights for
zero-shot intrinsic SimCC distribution uncertainty evaluation or fine-tuning.
"""
_base_ = ['./ua_simcc_soca_prnet34_coco_256x192.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    head=dict(
        in_channels=2048,
        use_compositional_mixer=True,
    ),
)
