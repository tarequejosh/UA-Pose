"""Uncertainty-Aware SimCC + SOCA-PRNet-34 + PCT Compositional Token Mixer.

Hardware profile: NVIDIA RTX 5060 (8GB VRAM) + Intel Core Ultra i7 (Windows 11).
Uses AMP (fp16) and batch_size=64 with 4 workers for fast single-GPU training.
"""
_base_ = ['../../mmpose/configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['ua_pose'],
    allow_failed_imports=False,
)

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer with AMP (Mixed Precision) for RTX 5060 Tensor Cores
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

auto_scale_lr = dict(base_batch_size=512)

default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50),
)

# SimCC coordinate classification codec (arXiv:2107.03332)
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=6.0,
    simcc_split_ratio=2.0,
)

# Model: SOCA-PRNet-34 backbone + UA-SimCC Head with PCT Compositional Tokens
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='SOCAPRNet',
        depth=34,
        soca_stages=(1, 2, 3),
        gcb_reduction=4,
        soca_reduction=8,
        dropout_prob=0.05,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34'),
    ),
    head=dict(
        type='UASimCCHead',
        in_channels=512,  # ResNet-34 final stage channels
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        use_compositional_mixer=True,  # PCT-inspired structural token mixer
        token_dim=128,
        num_tokens=8,
        dropout_rate=0.1,
        uncertainty_mode='combined',
        mc_samples=20,
        loss=dict(
            type='HeteroscedasticKLLoss',
            use_target_weight=True,
            adaptive_target_sigma=True,
            base_sigma=6.0,
            het_weight=0.5,
            cal_weight=0.2,
            vis_weight=0.2,
        ),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        uncertainty_mode='combined',
        calibrate_scores=True,
        calibration_alpha=0.2,
    ),
)

# Dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

# DataLoaders tuned for Windows + Core Ultra i7 + RTX 5060
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# Evaluators: both standard COCO OKS AP and Uncertainty-Aware Pose Metric
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    ),
    dict(
        type='UncertaintyPoseMetric',
        pck_threshold=0.1,
    ),
]
test_evaluator = val_evaluator
