"""Uncertainty-Aware SimCC + Lightweight PoseResNet-34 (ResNet-34) Config.

Compares standard ResNet-34 against SOCA-PRNet-34 with the UA-SimCC head.
"""
_base_ = ['./ua_simcc_soca_prnet34_coco_256x192.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=34,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34'),
    ),
    head=dict(
        in_channels=512,
        use_compositional_mixer=True,
    ),
)
