"""Fast Fine-Tuning Config (20 Epochs) for UA-SimCC + ViPNAS-MobileNetV3.

Loads the official COCO-pretrained MMPose SimCC ViPNAS-MobileNetV3 checkpoint
(69.5 AP, ultra-lightweight mobile backbone) and trains the PCT Compositional
Token Mixer + Heteroscedastic Uncertainty Head in ~15-20 minutes on RTX 5060.
"""
_base_ = ['./ua_simcc_vipnas_mbv3_coco_256x192.py']

load_from = (
    'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/'
    'coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth'
)

train_cfg = dict(max_epochs=20, val_interval=2)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-4, weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'deconv_head': dict(lr_mult=0.2),
            'token_mixer': dict(lr_mult=1.0),
            'uncertainty_head': dict(lr_mult=1.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=200, start_factor=0.01, by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        begin=1,
        end=20,
        T_max=19,
        eta_min=1e-6,
        by_epoch=True,
    ),
]
