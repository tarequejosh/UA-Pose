"""Uncertainty-Aware Heatmap + SOCA-PRNet-34 Config.

Supports both single-pass heteroscedastic + spatial spread uncertainty AND
20-pass MC-Dropout in the deconvolutional head.
"""
_base_ = ['../../mmpose/configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['ua_pose'],
    allow_failed_imports=False,
)

train_cfg = dict(max_epochs=210, val_interval=10)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

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

codec = dict(
    type='MSRAHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2,
)

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
        type='UAHeatmapHead',
        in_channels=512,
        out_channels=17,
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        dropout_rate=0.1,
        uncertainty_mode='combined',
        mc_samples=20,
        loss=dict(
            type='HeteroscedasticHeatmapLoss',
            use_target_weight=True,
            het_weight=0.2,
            cal_weight=0.2,
        ),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
        uncertainty_mode='combined',
        calibrate_scores=False,
    ),
)

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'
# Explicit, because MMPose's own lookup needs links made by its legacy installer.
metainfo = dict(from_file='mmpose/configs/_base_/datasets/coco.py')

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

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=metainfo,
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
        metainfo=metainfo,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

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
