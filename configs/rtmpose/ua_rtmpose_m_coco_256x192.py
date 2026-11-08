"""RTMPose-m (COCO, 256x192) with the training-free uncertainty signals of UARTMCCHead.

Published weights, evaluation only. The backbone uses MMPose's own CSPNeXt (same weights as the mmdet
one) and plain BN, since evaluation runs on one GPU without distributed training.
"""
_base_ = ['../../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py']

model = dict(
    backbone=dict(_scope_='mmpose', norm_cfg=dict(type='BN')),
    head=dict(type='UARTMCCHead', mc_dropout_rate=0.1),
    test_cfg=dict(flip_test=True, uncertainty_mode='dist_std', mc_samples=0),
)
