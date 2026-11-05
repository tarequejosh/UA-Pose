"""Uncertainty-Aware SimCC + ViPNAS-MobileNetV3 Ultra-Lightweight Config.

Compatible with official MMPose SimCC ViPNAS-MobileNetV3 pretrained weights:
https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth
"""
_base_ = ['./ua_simcc_soca_prnet34_coco_256x192.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='ViPNAS_MobileNetV3',
    ),
    head=dict(
        in_channels=160,
        deconv_type='vipnas',
        deconv_out_channels=(160, 160, 160),
        deconv_num_groups=(160, 160, 160),
        use_compositional_mixer=True,
    ),
)
