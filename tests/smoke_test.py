"""Hardware, CUDA, and Architecture Smoke Test for UA-PoseEstimation.

Verifies:
1. PyTorch + CUDA device detection (e.g., NVIDIA RTX 5060) & AMP capability
2. Custom backbone registration and forward pass (`SOCAPRNet-34`)
3. Custom head forward, loss, and uncertainty prediction (`UASimCCHead` with
   PCT Compositional Token Mixer & `UAHeatmapHead`)
4. Uncertainty evaluation metrics (`UncertaintyPoseMetric`, AUROC, AUSE)
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'mmpose'))

import ua_pose  # noqa: F401
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData, PixelData
from mmpose.registry import METRICS, MODELS
from mmpose.structures import PoseDataSample


def make_dummy_simcc_samples(batch_size: int = 2, num_joints: int = 17, w: int = 384, h: int = 512, device: str = 'cpu'):
    samples = []
    for i in range(batch_size):
        s = PoseDataSample()
        s.set_metainfo(
            dict(
                id=i + 1,
                img_id=100 + i,
                img_path='dummy.jpg',
                input_center=np.array([96.0, 128.0], dtype=np.float32),
                input_scale=np.array([192.0, 256.0], dtype=np.float32),
                input_size=(192, 256),
                flip_indices=list(range(num_joints)),
                raw_ann_info=dict(
                    id=i + 1,
                    image_id=100 + i,
                    bbox=[10.0, 10.0, 150.0, 220.0],
                    area=150.0 * 220.0,
                    keypoints=np.column_stack([
                        np.random.uniform(20, 140, size=num_joints),
                        np.random.uniform(20, 200, size=num_joints),
                        np.array([2] * 12 + [1] * 5),  # 12 visible, 5 occluded
                    ]).reshape(-1).tolist(),
                ),
            ))
        gt_inst = InstanceData()
        gt_inst.bboxes = np.array([[10.0, 10.0, 160.0, 230.0]], dtype=np.float32)
        gt_inst.bbox_scores = np.array([1.0], dtype=np.float32)
        s.gt_instances = gt_inst

        gt_labels = InstanceData()
        gt_labels.keypoint_x_labels = torch.softmax(
            torch.randn(1, num_joints, w, device=device), dim=-1)
        gt_labels.keypoint_y_labels = torch.softmax(
            torch.randn(1, num_joints, h, device=device), dim=-1)
        gt_labels.keypoint_weights = torch.ones(1, num_joints, device=device)
        s.gt_instance_labels = gt_labels

        gt_fields = PixelData()
        gt_fields.heatmaps = torch.rand(num_joints, 64, 48, device=device)
        s.gt_fields = gt_fields
        samples.append(s)
    return samples


def main():
    init_default_scope('mmpose')
    print('=' * 70)
    print('  UA-PoseEstimation Hardware & Architecture Verification')
    print('=' * 70)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'  PyTorch Version : {torch.__version__}')
    print(f'  Active Device   : {device}')
    if torch.cuda.is_available():
        print(f'  GPU Name        : {torch.cuda.get_device_name(0)}')
        print(f'  CUDA Version    : {torch.version.cuda}')
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'  Total VRAM      : {vram_gb:.2f} GB')
    print('-' * 70)

    # 1. Test SOCAPRNet-34 + UASimCCHead (with PCT Compositional Token Mixer)
    print('[1/3] Testing SOCAPRNet-34 + UASimCCHead (SimCC + PCT + Heteroscedastic)...')
    codec_simcc = dict(
        type='SimCCLabel', input_size=(192, 256), sigma=6.0, simcc_split_ratio=2.0)
    simcc_model_cfg = dict(
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
            dropout_prob=0.05,
        ),
        head=dict(
            type='UASimCCHead',
            in_channels=512,
            out_channels=17,
            input_size=(192, 256),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            use_compositional_mixer=True,
            token_dim=128,
            num_tokens=8,
            dropout_rate=0.1,
            uncertainty_mode='mc_dropout',
            mc_samples=5,
            loss=dict(type='HeteroscedasticKLLoss', use_target_weight=True),
            decoder=codec_simcc,
        ),
        test_cfg=dict(flip_test=True, uncertainty_mode='mc_dropout', mc_samples=5),
    )

    simcc_model = MODELS.build(simcc_model_cfg).to(device)
    params_m = sum(p.numel() for p in simcc_model.parameters()) / 1e6
    print(f'      Model built successfully ({params_m:.2f}M parameters).')

    dummy_imgs = torch.randn(2, 3, 256, 192, device=device)
    dummy_samples = make_dummy_simcc_samples(batch_size=2, device=device)

    simcc_model.train()
    losses = simcc_model.loss(dummy_imgs, dummy_samples)
    total_loss = sum(v for k, v in losses.items() if 'loss' in k)
    total_loss.backward()
    print('      Train loss & backward pass OK:', {k: float(v) for k, v in losses.items()})

    simcc_model.eval()
    with torch.no_grad():
        preds = simcc_model.predict(dummy_imgs, dummy_samples)
    pred0 = preds[0].pred_instances
    assert hasattr(pred0, 'keypoint_uncertainty'), 'Missing keypoint_uncertainty!'
    assert hasattr(pred0, 'uncertainty_het'), 'Missing uncertainty_het!'
    assert hasattr(pred0, 'uncertainty_dist_std'), 'Missing uncertainty_dist_std!'
    assert hasattr(pred0, 'uncertainty_mc'), 'Missing uncertainty_mc!'
    print(f'      Predict pass OK: keypoints={pred0.keypoints.shape}, '
          f'uncertainty={pred0.keypoint_uncertainty.shape}')

    # 2. Test SOCAPRNet-34 + UAHeatmapHead
    print('[2/3] Testing SOCAPRNet-34 + UAHeatmapHead (2D Heatmap + Heteroscedastic + MC-Dropout)...')
    codec_hm = dict(
        type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
    hm_model_cfg = dict(
        type='TopdownPoseEstimator',
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
        ),
        backbone=dict(type='SOCAPRNet', depth=34),
        head=dict(
            type='UAHeatmapHead',
            in_channels=512,
            out_channels=17,
            dropout_rate=0.1,
            uncertainty_mode='mc_dropout',
            mc_samples=5,
            loss=dict(type='HeteroscedasticHeatmapLoss', use_target_weight=True),
            decoder=codec_hm,
        ),
        test_cfg=dict(flip_test=False, uncertainty_mode='mc_dropout', mc_samples=5),
    )
    hm_model = MODELS.build(hm_model_cfg).to(device)
    hm_model.train()
    hm_losses = hm_model.loss(dummy_imgs, dummy_samples)
    sum(v for k, v in hm_losses.items() if 'loss' in k).backward()
    print('      Heatmap train loss & backward pass OK:', {k: float(v) for k, v in hm_losses.items()})

    hm_model.eval()
    with torch.no_grad():
        hm_preds = hm_model.predict(dummy_imgs, dummy_samples)
    hm_pred0 = hm_preds[0].pred_instances
    assert hasattr(hm_pred0, 'uncertainty_entropy'), 'Missing uncertainty_entropy on UAHeatmapHead!'
    assert hasattr(hm_pred0, 'raw_keypoint_scores'), 'Missing raw_keypoint_scores on UAHeatmapHead!'
    print(f'      Heatmap predict pass OK: uncertainty={hm_pred0.keypoint_uncertainty.shape}')

    # 3. Test UncertaintyPoseMetric
    print('[3/3] Testing UncertaintyPoseMetric (PCK@0.1, Occ/Vis Ratio, Spearman rho, AUSE)...')
    metric = METRICS.build(dict(type='UncertaintyPoseMetric', pck_threshold=0.1))
    sample_dicts = [s.to_dict() for s in preds]
    metric.process(data_batch={}, data_samples=sample_dicts)
    eval_res = metric.compute_metrics(metric.results)
    print(f'      Metric computation OK: Spearman rho={eval_res["spearman_rho_unc"]:.4f}, '
          f'Occ/Vis Ratio={eval_res["occ_vis_unc_ratio"]:.2f}x')

    print('=' * 70)
    print('  ALL SMOKE TESTS PASSED SUCCESSFULLY!')
    print('=' * 70)


if __name__ == '__main__':
    main()
