"""Fast Fine-Tuning Config (20 Epochs) for UA-SimCC + ResNet-50.

Loads the official COCO-pretrained MMPose SimCC ResNet-50 checkpoint so the
backbone, deconv layers, and 1D coordinate classifiers start from the published
weights, then trains the Compositional Token Mixer + Heteroscedastic Uncertainty
Head on a single GPU.
"""
_base_ = ['./ua_simcc_res50_coco_256x192.py']

load_from = 'checkpoints/simcc_res50_256x192.pth'

train_cfg = dict(max_epochs=20, val_interval=5)

# The paper reports the final epoch: picking a best epoch would select on val2017.
default_hooks = dict(
    checkpoint=dict(interval=5, save_best=None, max_keep_ckpts=1),
)

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
