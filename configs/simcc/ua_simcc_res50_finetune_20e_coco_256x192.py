"""Fast Fine-Tuning Config (20 Epochs) for UA-SimCC + ResNet-50.

Loads the official COCO-pretrained MMPose SimCC ResNet-50 checkpoint so the
backbone, deconv layers, and 1D coordinate classifiers start at 72.1 AP, and
rapidly trains the PCT Compositional Token Mixer + Heteroscedastic Uncertainty
Head on a single RTX 5060 GPU (~30-45 minutes for 20 epochs).
"""
_base_ = ['./ua_simcc_res50_coco_256x192.py']

load_from = (
    'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/'
    'coco/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth'
)

train_cfg = dict(max_epochs=20, val_interval=2)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4),
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
