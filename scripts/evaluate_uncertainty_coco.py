"""End-to-End Uncertainty-Aware Pose Estimation Benchmark & Figure Generator.

Evaluates any MMPose config + checkpoint on COCO val2017 (either a 500-image
random sample matching the preliminary study or the full 5,000-image val2017
set), compares:
  1. Model Confidence Baseline (1 - score)
  2. Intrinsic Distribution Spread / Entropy (Single-Pass, Zero-Cost)
  3. Learned Heteroscedastic Uncertainty (Single-Pass)
  4. Combined Uncertainty (Single-Pass)
  5. MC-Dropout (20 Passes on Keypoint Head)
and automatically generates publication-ready figures matching Figures 1, 2, 3.
"""
import argparse
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

# Ensure local project root and bundled mmpose are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'mmpose'))

import ua_pose  # noqa: F401 - registers SOCAPRNet, UASimCCHead, UAHeatmapHead, etc.
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose, default_collate
from mmengine.runner import load_checkpoint
from mmpose.datasets.datasets.body import CocoDataset
from mmpose.registry import MODELS
from ua_pose.evaluation.uncertainty_metric import (
    compute_auroc,
    compute_ece,
    compute_risk_coverage_and_ause,
)

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Uncertainty-Aware Pose Estimation & Plot Figures')
    parser.add_argument('config', help='Path to MMPose config file')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Path or URL to checkpoint file (optional; can use pretrained URL)',
    )
    parser.add_argument(
        '--data-root',
        default='data/coco/',
        help='COCO dataset root directory (containing val2017/ and annotations/)',
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=500,
        help='Number of COCO val2017 images to sample (default 500 to match '
        'preliminary study; set to -1 for all 5000 images)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling 500 COCO val2017 images',
    )
    parser.add_argument(
        '--mc-samples',
        type=int,
        default=20,
        help='Number of forward passes of the head for MC-Dropout comparison',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Inference batch size (default 64 for RTX 5060)',
    )
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device used for inference',
    )
    parser.add_argument(
        '--out-dir',
        default='work_dirs/uncertainty_eval',
        help='Directory to save JSON metrics and Figures 1, 2, 3',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings',
    )
    return parser.parse_args()


def plot_figure2_and_figure3(
    errors: np.ndarray,
    vis: np.ndarray,
    scores: np.ndarray,
    unc_dict: dict,
    primary_key: str,
    out_dir: Path,
) -> dict:
    """Generate Figure 2 (Visible vs Occluded + PCK@0.1 Sparsification) and
    Figure 3 (Quintile Error + Scatter Uncertainty vs Error)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_mask = vis == 2
    occ_mask = vis == 1
    primary_unc = unc_dict[primary_key]

    # ---------------- Figure 2 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)

    # Fig 2(a): Visible vs Occluded Uncertainty distribution (Boxplot + Violin/Means)
    ax = axes[0]
    u_vis = primary_unc[vis_mask]
    u_occ = primary_unc[occ_mask]
    ratio = float(np.mean(u_occ)) / max(float(np.mean(u_vis)), 1e-8)

    parts = ax.violinplot(
        [u_vis, u_occ],
        positions=[1, 2],
        showmeans=True,
        showextrema=False,
        widths=0.6,
    )
    colors = ['#2ecc71', '#e74c3c']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.65)

    ax.boxplot(
        [u_vis, u_occ],
        positions=[1, 2],
        widths=0.18,
        showfliers=False,
        boxprops=dict(linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f'Visible (v=2)\n(n={len(u_vis):,})', f'Occluded (v=1)\n(n={len(u_occ):,})'],
        fontsize=11,
    )
    ax.set_ylabel(f'Uncertainty Score ({primary_key})', fontsize=11)
    ax.set_title(
        f'(a) Uncertainty: Visible vs. Occluded ({ratio:.2f}x higher on occluded)',
        fontsize=12,
        fontweight='bold',
    )
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Fig 2(b): PCK@0.1 Accuracy curves when filtering keypoints by confidence/uncertainty
    ax2 = axes[1]
    rc_base = compute_risk_coverage_and_ause(errors, -scores, pck_threshold=0.1)
    ax2.plot(
        rc_base['fractions'] * 100,
        rc_base['pck_curve'] * 100,
        '--',
        color='#7f8c8d',
        linewidth=2,
        label=f'Peak Confidence Baseline (AUSE={rc_base["ause_pck"]:.4f})',
    )

    palette = {
        'dist_std': ('#3498db', '1D/2D Distribution Spread (Single-Pass)'),
        'entropy': ('#9b59b6', 'Distributional Entropy (Single-Pass)'),
        'heteroscedastic': ('#e67e22', 'Learned Heteroscedastic Head (Single-Pass)'),
        'combined': ('#e74c3c', 'Combined UA Head (Single-Pass)'),
        'mc_dropout': ('#16a085', 'MC-Dropout (20 Passes)'),
    }

    for k_name, u_vals in unc_dict.items():
        if np.std(u_vals) < 1e-12:
            continue
        rc = compute_risk_coverage_and_ause(errors, u_vals, pck_threshold=0.1)
        color, label_str = palette.get(k_name, ('#2c3e50', k_name))
        lw = 2.6 if k_name == primary_key else 1.8
        ax2.plot(
            rc['fractions'] * 100,
            rc['pck_curve'] * 100,
            color=color,
            linewidth=lw,
            label=f'{label_str} (AUSE={rc["ause_pck"]:.4f})',
        )

    ax2.plot(
        rc_base['fractions'] * 100,
        rc_base['oracle_pck_curve'] * 100,
        ':',
        color='black',
        linewidth=1.5,
        label='Oracle Upper Bound',
    )
    ax2.set_xlabel('Retained Keypoints (%, sorted most confident first)', fontsize=11)
    ax2.set_ylabel('PCK@0.1 Accuracy (%)', fontsize=11)
    ax2.set_title('(b) PCK@0.1 Risk-Coverage Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig2_path = out_dir / 'figure2_occlusion_and_pck_curves.png'
    plt.savefig(fig2_path)
    plt.close(fig)

    # ---------------- Figure 3 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)

    # Fig 3 Left: Mean Localization Error per Uncertainty Quintile
    ax = axes[0]
    order = np.argsort(primary_unc)
    splits = np.array_split(order, 5)
    q_means = [float(np.mean(errors[s])) for s in splits]
    q_stds = [float(np.std(errors[s]) / np.sqrt(len(s))) for s in splits]
    q_labels = ['Q1\n(Lowest Unc)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest Unc)']
    bar_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    bars = ax.bar(
        range(5),
        q_means,
        yerr=q_stds,
        capsize=5,
        color=bar_colors,
        edgecolor='black',
        alpha=0.85,
    )
    for bar, m in zip(bars, q_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f'{m:.3f}',
            ha='center',
            va='bottom',
            fontsize=9.5,
            fontweight='bold',
        )
    ax.set_xticks(range(5))
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_ylabel('Normalized Localization Error (d / sqrt(Area))', fontsize=11)
    ax.set_title('Left: Localization Error per Uncertainty Quintile', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Fig 3 Right: Per-keypoint Uncertainty vs. Localization Error
    ax2 = axes[1]
    rho, _ = spearmanr(primary_unc, errors)
    r_val, _ = pearsonr(primary_unc, errors)

    # Subsample up to 4000 points for clean scatter rendering
    rng = np.random.default_rng(42)
    idx_plot = rng.choice(len(errors), size=min(4000, len(errors)), replace=False)
    vis_sub = vis[idx_plot] == 2
    occ_sub = vis[idx_plot] == 1

    ax2.scatter(
        primary_unc[idx_plot][vis_sub],
        errors[idx_plot][vis_sub],
        s=10,
        alpha=0.35,
        c='#2ecc71',
        label='Visible (v=2)',
    )
    ax2.scatter(
        primary_unc[idx_plot][occ_sub],
        errors[idx_plot][occ_sub],
        s=14,
        alpha=0.55,
        c='#e74c3c',
        marker='x',
        label='Occluded (v=1)',
    )

    # Binned trend line
    bins = np.percentile(primary_unc, np.linspace(2, 98, 12))
    bin_centers, bin_errs = [], []
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        m = (primary_unc >= b_low) & (primary_unc <= b_high)
        if np.sum(m) > 10:
            bin_centers.append(float(np.mean(primary_unc[m])))
            bin_errs.append(float(np.mean(errors[m])))
    if bin_centers:
        ax2.plot(
            bin_centers,
            bin_errs,
            color='black',
            linewidth=2.5,
            marker='o',
            label='Binned Mean Trend',
        )

    ax2.set_ylim(0, min(float(np.percentile(errors, 99)) * 1.25, 1.0))
    ax2.set_xlabel(f'Per-Keypoint Uncertainty ({primary_key})', fontsize=11)
    ax2.set_ylabel('Normalized Localization Error', fontsize=11)
    ax2.set_title(
        f'Right: Uncertainty vs. Error (Spearman ρ = {rho:.3f}, r = {r_val:.3f})',
        fontsize=12,
        fontweight='bold',
    )
    ax2.legend(loc='upper left', fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig3_path = out_dir / 'figure3_quintiles_and_error_correlation.png'
    plt.savefig(fig3_path)
    plt.close(fig)

    return dict(
        fig2_path=str(fig2_path),
        fig3_path=str(fig3_path),
        quintile_means=q_means,
    )


def plot_figure1_skeletons(vis_samples: list, out_dir: Path, unc_min: float, unc_max: float):
    """Generate Figure 1: Predicted skeletons colored by per-joint uncertainty
    (green = confident, red = uncertain; cyan crosses = occluded joints)."""
    if not vis_samples:
        return None

    n_cols = min(len(vis_samples), 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.8), dpi=200)
    if n_cols == 1:
        axes = [axes]

    cmap = plt.get_cmap('RdYlGn_r')  # 0.0 -> Green (confident), 1.0 -> Red (uncertain)

    for ax, sample in zip(axes, vis_samples[:n_cols]):
        img_path = sample['img_path']
        if osp.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.full((384, 288, 3), 240, dtype=np.uint8)

        kpts = sample['pred_kpts']  # [K, 2]
        gt_vis = sample['gt_vis']  # [K]
        unc = sample['unc']  # [K]
        bbox = sample['bbox']  # [x, y, w, h]

        # Crop around person bbox with padding for clear visualization
        x, y, w, h = bbox
        pad = 0.2 * max(w, h)
        x1 = int(max(0, x - pad))
        y1 = int(max(0, y - pad))
        x2 = int(min(img.shape[1], x + w + pad))
        y2 = int(min(img.shape[0], y + h + pad))

        crop = img[y1:y2, x1:x2]
        ax.imshow(crop)

        norm_u = np.clip((unc - unc_min) / max(unc_max - unc_min, 1e-6), 0.0, 1.0)

        # Draw skeleton links
        for u_idx, v_idx in COCO_SKELETON:
            if gt_vis[u_idx] > 0 and gt_vis[v_idx] > 0:
                edge_u = 0.5 * (norm_u[u_idx] + norm_u[v_idx])
                color = cmap(edge_u)
                ax.plot(
                    [kpts[u_idx, 0] - x1, kpts[v_idx, 0] - x1],
                    [kpts[u_idx, 1] - y1, kpts[v_idx, 1] - y1],
                    color=color,
                    linewidth=2.2,
                    alpha=0.85,
                )

        # Draw joints
        for j in range(len(kpts)):
            if gt_vis[j] == 0:
                continue
            jx, jy = kpts[j, 0] - x1, kpts[j, 1] - y1
            color = cmap(norm_u[j])
            ax.scatter([jx], [jy], c=[color], s=42, edgecolors='black', linewidths=0.8, zorder=4)
            if gt_vis[j] == 1:  # Occluded joint -> cyan cross
                ax.scatter(
                    [jx], [jy], c='cyan', marker='x', s=70, linewidths=2.0, zorder=5)

        ax.set_axis_off()
        ax.set_title(
            f'Img {sample["img_id"]} | Err={sample["mean_err"]:.3f}',
            fontsize=10,
            fontweight='bold',
        )

    fig.suptitle(
        'Figure 1. Predicted Skeletons with Per-Joint Uncertainty '
        '(Green = Confident, Red = Uncertain; Cyan × = Occluded)',
        fontsize=12,
        fontweight='bold',
    )
    plt.tight_layout()
    fig1_path = out_dir / 'figure1_predicted_skeletons_uncertainty.png'
    plt.savefig(fig1_path)
    plt.close(fig)
    return str(fig1_path)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Enable MC-Dropout evaluation inside the head so all uncertainty modes are
    # collected simultaneously for a comprehensive comparison table, while
    # keeping raw keypoint_scores uncalibrated as the baseline comparator.
    cfg.model.test_cfg['uncertainty_mode'] = 'mc_dropout'
    cfg.model.test_cfg['mc_samples'] = args.mc_samples
    cfg.model.test_cfg['calibrate_scores'] = False

    # Build model
    model = MODELS.build(cfg.model)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(args.device)
    model.eval()

    # Load COCO val2017 ground-truth dataset
    ann_file = osp.join(args.data_root, 'annotations/person_keypoints_val2017.json')
    if not osp.exists(ann_file):
        raise FileNotFoundError(
            f'COCO annotation file not found at {ann_file}. '
            f'Run `python scripts/download_coco_and_weights.py` first.')

    val_pipeline = cfg.val_dataloader.dataset.pipeline
    dataset = CocoDataset(
        data_root=args.data_root,
        data_mode='topdown',
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    )

    # Subsample 500 unique images (matching preliminary experiment protocol)
    all_img_ids = sorted(list({dataset.get_data_info(i)['img_id'] for i in range(len(dataset))}))
    if 0 < args.num_images < len(all_img_ids):
        rng = np.random.default_rng(args.seed)
        sampled_img_ids = set(
            rng.choice(all_img_ids, size=args.num_images, replace=False).tolist())
        selected_indices = [
            i for i in range(len(dataset))
            if dataset.get_data_info(i)['img_id'] in sampled_img_ids
        ]
    else:
        sampled_img_ids = set(all_img_ids)
        selected_indices = list(range(len(dataset)))

    print(
        f'Evaluating {len(selected_indices)} person instances across '
        f'{len(sampled_img_ids)} COCO val2017 images on {args.device}...'
    )

    all_errors = []
    all_vis = []
    all_scores = []
    unc_het_list = []
    unc_dist_list = []
    unc_ent_list = []
    unc_mc_list = []
    vis_samples = []

    t_start = time.perf_counter()

    for start_idx in range(0, len(selected_indices), args.batch_size):
        batch_indices = selected_indices[start_idx:start_idx + args.batch_size]
        batch_items = [dataset[idx] for idx in batch_indices]
        batch_data = default_collate(batch_items)

        with torch.no_grad():
            outputs = model.test_step(batch_data)

        for data_sample in outputs:
            pred_inst = data_sample.pred_instances
            pred_kpts = np.asarray(pred_inst.keypoints)[0]  # [K, 2]
            pred_scores = np.asarray(pred_inst.keypoint_scores)[0]  # [K]

            raw_ann = data_sample.metainfo.get('raw_ann_info', None)
            if isinstance(raw_ann, list):
                raw_ann = raw_ann[0]
            gt_raw = np.asarray(raw_ann['keypoints'], dtype=np.float64).reshape(-1, 3)
            gt_kpts = gt_raw[:, :2]
            gt_vis = gt_raw[:, 2].astype(np.int64)
            bbox = raw_ann['bbox']
            box_area = float(max(bbox[2] * bbox[3], 1.0))

            norm_err = np.linalg.norm(pred_kpts - gt_kpts, axis=-1) / np.sqrt(box_area)

            u_het = np.asarray(getattr(pred_inst, 'uncertainty_het', [1.0 - pred_scores]))[0]
            u_dist = np.asarray(getattr(pred_inst, 'uncertainty_dist_std', [1.0 - pred_scores]))[0]
            u_ent = np.asarray(getattr(pred_inst, 'uncertainty_entropy', u_dist[None]))[0]
            u_mc = np.asarray(getattr(pred_inst, 'uncertainty_mc', np.zeros_like(u_dist)[None]))[0]

            all_errors.append(norm_err)
            all_vis.append(gt_vis)
            all_scores.append(pred_scores)
            unc_het_list.append(u_het)
            unc_dist_list.append(u_dist)
            unc_ent_list.append(u_ent)
            unc_mc_list.append(u_mc)

            # Save qualitative examples that contain both visible and occluded joints
            if len(vis_samples) < 8 and np.sum(gt_vis == 1) >= 3 and np.sum(gt_vis == 2) >= 6:
                vis_samples.append(
                    dict(
                        img_id=data_sample.metainfo['img_id'],
                        img_path=data_sample.metainfo['img_path'],
                        pred_kpts=pred_kpts,
                        gt_vis=gt_vis,
                        unc=0.5 * u_het + 0.5 * u_dist,
                        bbox=bbox,
                        mean_err=float(np.mean(norm_err[gt_vis > 0])),
                    ))

    elapsed = time.perf_counter() - t_start

    all_errors = np.concatenate(all_errors, axis=0)
    all_vis = np.concatenate(all_vis, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    u_het_all = np.concatenate(unc_het_list, axis=0)
    u_dist_all = np.concatenate(unc_dist_list, axis=0)
    u_ent_all = np.concatenate(unc_ent_list, axis=0)
    u_mc_all = np.concatenate(unc_mc_list, axis=0)

    labeled = all_vis > 0
    errors = all_errors[labeled]
    vis = all_vis[labeled]
    scores = all_scores[labeled]

    # Normalize each component to unit mean before combining
    u_het_l = u_het_all[labeled] / max(float(np.mean(u_het_all[labeled])), 1e-8)
    u_dist_l = u_dist_all[labeled] / max(float(np.mean(u_dist_all[labeled])), 1e-8)
    u_ent_l = u_ent_all[labeled]
    u_mc_l = u_mc_all[labeled]
    u_comb_l = 0.5 * u_het_l + 0.5 * u_dist_l

    unc_dict = {
        'dist_std': u_dist_all[labeled],
        'entropy': u_ent_l,
        'heteroscedastic': u_het_all[labeled],
        'combined': u_comb_l,
        'mc_dropout': u_mc_l,
    }

    # Summarize comparison table across all uncertainty methods
    summary = {
        'num_images': len(sampled_img_ids),
        'num_people': len(selected_indices),
        'num_keypoints_total': int(np.sum(labeled)),
        'num_keypoints_visible': int(np.sum(vis == 2)),
        'num_keypoints_occluded': int(np.sum(vis == 1)),
        'PCK@0.1_all': float(np.mean(errors < 0.1)),
        'PCK@0.1_visible': float(np.mean(errors[vis == 2] < 0.1)),
        'PCK@0.1_occluded': float(np.mean(errors[vis == 1] < 0.1)),
        'compute_time_sec': elapsed,
        'sec_per_image': elapsed / max(len(sampled_img_ids), 1),
        'methods': {},
    }

    method_candidates = {'baseline_neg_conf': -scores, **unc_dict}
    is_correct = (errors < 0.1).astype(np.float64)
    ece_baseline = compute_ece(scores, is_correct)
    summary['ECE_baseline'] = float(ece_baseline)

    for m_name, u_vec in method_candidates.items():
        if np.std(u_vec) < 1e-12:
            continue
        rho, _ = spearmanr(u_vec, errors)
        r_val, _ = pearsonr(u_vec, errors)
        u_v = float(np.mean(u_vec[vis == 2]))
        u_o = float(np.mean(u_vec[vis == 1]))
        ratio = u_o / max(u_v, 1e-8) if u_v > 0 else np.nan
        auroc_occ = compute_auroc(u_vec, (vis == 1).astype(np.int64))
        rc = compute_risk_coverage_and_ause(errors, u_vec, pck_threshold=0.1)
        if m_name == 'baseline_neg_conf':
            ece_val = ece_baseline
        else:
            norm_u = np.maximum(u_vec, 0.0) / max(float(np.mean(np.maximum(u_vec, 0.0))), 1e-8)
            cal_conf = np.clip(scores * np.exp(-0.3 * norm_u), 0.0, 1.0)
            ece_val = compute_ece(cal_conf, is_correct)
        summary['methods'][m_name] = {
            'spearman_rho': float(rho),
            'pearson_r': float(r_val),
            'occ_vis_ratio': float(ratio),
            'occlusion_auroc': float(auroc_occ),
            'AUSE_pck01': float(rc['ause_pck']),
            'AUSE_err': float(rc['ause_err']),
            'ECE': float(ece_val),
        }

    # Plot figures
    fig_info = plot_figure2_and_figure3(
        errors=errors,
        vis=vis,
        scores=scores,
        unc_dict=unc_dict,
        primary_key='combined',
        out_dir=out_dir,
    )
    fig1_path = plot_figure1_skeletons(
        vis_samples=vis_samples,
        out_dir=out_dir,
        unc_min=float(np.percentile(u_comb_l, 5)),
        unc_max=float(np.percentile(u_comb_l, 95)),
    )
    summary['figures'] = {**fig_info, 'fig1_path': fig1_path}

    json_path = out_dir / 'uncertainty_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Export LaTeX table snippet for direct inclusion in Q1 journal manuscript
    latex_lines = [
        '% Auto-generated LaTeX Table for Q1 Journal Manuscript',
        '\\begin{tabular}{lccccc}',
        '\\toprule',
        'Uncertainty Method & Spearman $\\rho \\uparrow$ & Occ/Vis Ratio $\\uparrow$ & Occ AUROC $\\uparrow$ & AUSE (PCK) $\\downarrow$ & ECE $\\downarrow$ \\\\',
        '\\midrule',
    ]
    for m_name, m_stats in summary['methods'].items():
        ratio_str = f'{m_stats["occ_vis_ratio"]:.2f}$\\times$' if not np.isnan(m_stats['occ_vis_ratio']) else '--'
        latex_lines.append(
            f'{m_name.replace("_", "\\_")} & {m_stats["spearman_rho"]:.3f} & '
            f'{ratio_str} & {m_stats["occlusion_auroc"]:.3f} & '
            f'{m_stats["AUSE_pck01"]:.4f} & {m_stats["ECE"]:.4f} \\\\'
        )
    latex_lines.extend(['\\bottomrule', '\\end{tabular}'])
    latex_path = out_dir / 'q1_paper_table.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines) + '\n')

    print(f'\nSaved full metrics to {json_path} and LaTeX table to {latex_path}')
    print(f'PCK@0.1 (All): {summary["PCK@0.1_all"]*100:.2f}% | '
          f'Vis: {summary["PCK@0.1_visible"]*100:.2f}% | '
          f'Occ: {summary["PCK@0.1_occluded"]*100:.2f}%')
    print('-' * 90)
    print(f'{"Method":<22} | {"Spearman ρ":>10} | {"Occ/Vis Ratio":>13} | '
          f'{"Occ AUROC":>9} | {"AUSE (PCK)":>10} | {"ECE":>8}')
    print('-' * 90)
    for m_name, m_stats in summary['methods'].items():
        print(f'{m_name:<22} | {m_stats["spearman_rho"]:>10.4f} | '
              f'{m_stats["occ_vis_ratio"]:>12.2f}x | '
              f'{m_stats["occlusion_auroc"]:>9.4f} | '
              f'{m_stats["AUSE_pck01"]:>10.4f} | '
              f'{m_stats["ECE"]:>8.4f}')
    print('-' * 90)


if __name__ == '__main__':
    main()
