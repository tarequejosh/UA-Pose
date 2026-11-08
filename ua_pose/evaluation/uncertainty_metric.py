"""Uncertainty-Aware Evaluation Metric for COCO Keypoints.

Computes all metrics from the preliminary MC-Dropout experiment plus
publication-standard uncertainty calibration metrics:
- PCK@0.1 (distance normalized by sqrt(bbox area) < 0.1)
- Mean Normalized Localization Error (overall, visible v=2, occluded v=1)
- Occluded vs. Visible Uncertainty Ratio (e.g., 2.55x in preliminary work)
- Occlusion Detection AUROC (using uncertainty to classify v=1 vs v=2)
- Spearman Rank Correlation rho (uncertainty vs. localization error)
- Pearson Correlation r (uncertainty vs. localization error)
- Area Under Sparsification Error (AUSE) & PCK@0.1 Risk-Coverage Curves
- Error per Uncertainty Quintile (Q1..Q5)
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from scipy.stats import pearsonr, rankdata, spearmanr

from mmpose.registry import METRICS


def compute_auroc(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """Compute Area Under the ROC Curve without external sklearn dependency.

    Args:
        scores (np.ndarray): Uncertainty scores [N] (higher = more likely positive).
        binary_labels (np.ndarray): Binary targets [N] in {0, 1} (1 = occluded).
    """
    pos_mask = binary_labels == 1
    neg_mask = binary_labels == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Mann-Whitney U statistic with average ranks, so tied scores count as 0.5. Ordinal
    # ranks would let the sort order decide ties, which matters for discrete signals
    # such as argmax MC-Dropout where many keypoints have exactly zero spread.
    ranks = rankdata(scores, method='average')
    auc = (np.sum(ranks[pos_mask]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE, Gu et al., ICML 2024).

    Args:
        confidences (np.ndarray): Predicted confidence scores in [0, 1].
        accuracies (np.ndarray): Empirical accuracy / OKS / binary PCK in [0, 1].
        num_bins (int): Number of uniform confidence bins. Defaults to 10.
    """
    n = len(confidences)
    if n == 0:
        return 0.0
    # Inputs must already be probabilities. Raw SimCC scores are logits and need a
    # fitted mapping first (see scripts/analyze_results.py); clipping them is not one.
    conf = np.clip(confidences, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        mask = (conf >= b_low) & (conf < b_high if b_high < 1.0 else conf <= b_high)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(conf[mask]))
            avg_acc = float(np.mean(accuracies[mask]))
            ece += (count / n) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_risk_coverage_and_ause(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    pck_threshold: float = 0.1,
    num_steps: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute PCK@0.1 retention curve and Area Under Sparsification Error (AUSE).

    Args:
        errors (np.ndarray): Normalized localization errors [N].
        uncertainties (np.ndarray): Uncertainty scores [N] (higher = more uncertain).
        pck_threshold (float): Threshold for PCK (default 0.1).
        num_steps (int): Number of coverage steps from 10% to 100%.
    """
    n = len(errors)
    if n == 0:
        return dict(
            fractions=np.linspace(0.1, 1.0, num_steps),
            pck_curve=np.zeros(num_steps),
            oracle_pck_curve=np.zeros(num_steps),
            err_curve=np.zeros(num_steps),
            oracle_err_curve=np.zeros(num_steps),
            ause_err=0.0,
            ause_pck=0.0,
        )

    # Sort ascending by uncertainty (most confident first). Ties are broken by a fixed
    # random key: stable order would leak dataset order into the curve.
    tie_break = np.random.default_rng(0).random(n)
    unc_order = np.lexsort((tie_break, uncertainties))
    # Oracle sorts ascending by true error
    oracle_order = np.argsort(errors)

    fractions = np.linspace(0.1, 1.0, num_steps)
    pck_curve = []
    oracle_pck_curve = []
    err_curve = []
    oracle_err_curve = []

    is_correct = (errors < pck_threshold).astype(np.float64)

    for frac in fractions:
        k = max(1, int(round(frac * n)))
        idx_u = unc_order[:k]
        idx_o = oracle_order[:k]

        pck_curve.append(float(np.mean(is_correct[idx_u])))
        oracle_pck_curve.append(float(np.mean(is_correct[idx_o])))

        err_curve.append(float(np.mean(errors[idx_u])))
        oracle_err_curve.append(float(np.mean(errors[idx_o])))

    pck_curve = np.array(pck_curve)
    oracle_pck_curve = np.array(oracle_pck_curve)
    err_curve = np.array(err_curve)
    oracle_err_curve = np.array(oracle_err_curve)

    # AUSE: area between model sparsification error and oracle sparsification error
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    ause_err = float(trapz_fn(err_curve - oracle_err_curve, fractions))
    ause_pck = float(trapz_fn(oracle_pck_curve - pck_curve, fractions))

    return dict(
        fractions=fractions,
        pck_curve=pck_curve,
        oracle_pck_curve=oracle_pck_curve,
        err_curve=err_curve,
        oracle_err_curve=oracle_err_curve,
        ause_err=ause_err,
        ause_pck=ause_pck,
    )


@METRICS.register_module()
class UncertaintyPoseMetric(BaseMetric):
    """MMPose Evaluator Metric for Keypoint Uncertainty & Reliability.

    Matches the exact evaluation protocol from the preliminary COCO val2017
    study:
    - Accuracy metric: Euclidean distance to ground truth normalized by
      ``sqrt(box_area)``, and ``PCK@0.1`` (fraction of keypoints with
      normalized error < 0.1).
    - Visibility split: Uses raw COCO visibility flags ``v=2`` (visible) vs.
      ``v=1`` (occluded).
    - Compares both the model's primary ``keypoint_uncertainty`` (plus sub-types
      ``uncertainty_het``, ``uncertainty_dist_std``, ``uncertainty_mc``) AND
      the baseline confidence score ``1.0 - keypoint_scores``.

    Args:
        pck_threshold (float): Normalized distance threshold for PCK.
            Defaults to 0.1.
        collect_device (str): Device for distributed metric collection.
            Defaults to 'cpu'.
        prefix (str, optional): Metric name prefix. Defaults to 'ua'.
    """

    default_prefix: Optional[str] = 'ua'

    def __init__(
        self,
        pck_threshold: float = 0.1,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.pck_threshold = pck_threshold

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of predictions and ground-truth annotations."""
        for data_sample in data_samples:
            pred_inst = data_sample['pred_instances']
            # Predicted keypoints in image space: [1, K, 2]
            pred_kpts = np.asarray(pred_inst['keypoints'])[0]  # [K, 2]
            pred_scores = np.asarray(pred_inst['keypoint_scores'])[0]  # [K]

            # Extract raw COCO ground truth from metainfo['raw_ann_info']
            raw_ann = data_sample.get('raw_ann_info', None)
            if isinstance(raw_ann, list):
                raw_ann = raw_ann[0]

            if raw_ann is not None and 'keypoints' in raw_ann:
                gt_raw = np.asarray(
                    raw_ann['keypoints'], dtype=np.float64).reshape(-1, 3)
                gt_kpts = gt_raw[:, :2]
                gt_vis = gt_raw[:, 2].astype(np.int64)  # 0=unlabeled, 1=occluded, 2=visible
                bbox = raw_ann.get('bbox', None)
                if bbox is not None:
                    box_area = float(max(bbox[2] * bbox[3], 1.0))
                else:
                    box_area = float(max(raw_ann.get('area', 1.0), 1.0))
            else:
                # Fallback if evaluated on a dataset without raw_ann_info
                gt_inst = data_sample['gt_instances']
                gt_kpts = np.asarray(gt_inst['keypoints'])[0]
                gt_vis_raw = np.asarray(gt_inst['keypoints_visible'])[0]
                # Approximate v=2 for visible
                gt_vis = np.where(gt_vis_raw > 0, 2, 0).astype(np.int64)
                bboxes = np.asarray(gt_inst['bboxes'])[0]
                box_area = float(
                    max((bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]), 1.0))

            norm_factor = np.sqrt(box_area)
            # Euclidean distance normalized by sqrt(box_area)
            norm_errors = np.linalg.norm(pred_kpts - gt_kpts, axis=-1) / norm_factor

            # Extract uncertainty arrays if present, else fallback to (1 - score)
            unc_primary = np.asarray(
                pred_inst.get('keypoint_uncertainty', 1.0 - pred_scores[None])
            )[0]
            unc_het = np.asarray(
                pred_inst.get('uncertainty_het', unc_primary[None])
            )[0]
            unc_dist = np.asarray(
                pred_inst.get('uncertainty_dist_std', unc_primary[None])
            )[0]
            unc_mc = np.asarray(
                pred_inst.get('uncertainty_mc', np.zeros_like(unc_primary)[None])
            )[0]

            self.results.append(
                dict(
                    norm_errors=norm_errors,
                    gt_vis=gt_vis,
                    scores=pred_scores,
                    unc_primary=unc_primary,
                    unc_het=unc_het,
                    unc_dist=unc_dist,
                    unc_mc=unc_mc,
                ))

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Aggregate keypoint-level statistics across all evaluated people."""
        logger: MMLogger = MMLogger.get_current_instance()

        all_errors = np.concatenate([r['norm_errors'] for r in results], axis=0)
        all_vis = np.concatenate([r['gt_vis'] for r in results], axis=0)
        all_scores = np.concatenate([r['scores'] for r in results], axis=0)
        all_unc = np.concatenate([r['unc_primary'] for r in results], axis=0)
        all_het = np.concatenate([r['unc_het'] for r in results], axis=0)
        all_dist = np.concatenate([r['unc_dist'] for r in results], axis=0)

        # Filter to labeled keypoints (v == 1 occluded or v == 2 visible)
        labeled_mask = all_vis > 0
        errors = all_errors[labeled_mask]
        vis = all_vis[labeled_mask]
        scores = all_scores[labeled_mask]
        baseline_unc = -scores  # Higher = less confident
        unc = all_unc[labeled_mask]
        het = all_het[labeled_mask]
        dist = all_dist[labeled_mask]

        vis_mask = vis == 2
        occ_mask = vis == 1

        n_total = int(np.sum(labeled_mask))
        n_vis = int(np.sum(vis_mask))
        n_occ = int(np.sum(occ_mask))

        # 1. Pose Accuracy (PCK@0.1 & Mean Normalized Error)
        pck_all = float(np.mean(errors < self.pck_threshold)) if n_total > 0 else 0.0
        pck_vis = float(np.mean(errors[vis_mask] < self.pck_threshold)) if n_vis > 0 else 0.0
        pck_occ = float(np.mean(errors[occ_mask] < self.pck_threshold)) if n_occ > 0 else 0.0

        mean_err_all = float(np.mean(errors)) if n_total > 0 else 0.0
        mean_err_vis = float(np.mean(errors[vis_mask])) if n_vis > 0 else 0.0
        mean_err_occ = float(np.mean(errors[occ_mask])) if n_occ > 0 else 0.0

        # 2. Occlusion vs. Visibility Uncertainty Ratio & AUROC
        unc_vis_mean = float(np.mean(unc[vis_mask])) if n_vis > 0 else 1e-8
        unc_occ_mean = float(np.mean(unc[occ_mask])) if n_occ > 0 else 0.0
        occ_vis_ratio = unc_occ_mean / max(unc_vis_mean, 1e-8)

        occ_binary = (vis == 1).astype(np.int64)
        occ_auroc_unc = compute_auroc(unc, occ_binary)
        occ_auroc_base = compute_auroc(baseline_unc, occ_binary)

        # 3. Error Correlation (Spearman rho & Pearson r)
        if n_total > 2 and np.std(unc) > 1e-12:
            rho_unc, _ = spearmanr(unc, errors)
            r_unc, _ = pearsonr(unc, errors)
        else:
            rho_unc, r_unc = 0.0, 0.0

        if n_total > 2 and np.std(het) > 1e-12:
            rho_het, _ = spearmanr(het, errors)
        else:
            rho_het = 0.0

        if n_total > 2 and np.std(dist) > 1e-12:
            rho_dist, _ = spearmanr(dist, errors)
        else:
            rho_dist = 0.0

        if n_total > 2 and np.std(baseline_unc) > 1e-12:
            rho_base, _ = spearmanr(baseline_unc, errors)
        else:
            rho_base = 0.0

        # 4. Risk-Coverage & AUSE
        rc_unc = compute_risk_coverage_and_ause(
            errors, unc, pck_threshold=self.pck_threshold)
        rc_base = compute_risk_coverage_and_ause(
            errors, baseline_unc, pck_threshold=self.pck_threshold)

        # 5. Error per Uncertainty Quintile (Q1 = 20% most confident, Q5 = 20% most uncertain)
        quintile_errors = []
        if n_total >= 5:
            order = np.argsort(unc)
            splits = np.array_split(order, 5)
            for sp in splits:
                quintile_errors.append(float(np.mean(errors[sp])))
        else:
            quintile_errors = [0.0] * 5

        # ECE is deliberately not logged here: SimCC scores are logits, so it needs a
        # mapping fitted on held-out data. See scripts/analyze_results.py.

        metrics = dict(
            num_keypoints=n_total,
            num_visible=n_vis,
            num_occluded=n_occ,
            PCK_01_all=pck_all,
            PCK_01_vis=pck_vis,
            PCK_01_occ=pck_occ,
            mean_norm_err_all=mean_err_all,
            mean_norm_err_vis=mean_err_vis,
            mean_norm_err_occ=mean_err_occ,
            occ_vis_unc_ratio=float(occ_vis_ratio),
            occ_auroc_unc=float(occ_auroc_unc),
            occ_auroc_baseline=float(occ_auroc_base),
            spearman_rho_unc=float(rho_unc),
            spearman_rho_het=float(rho_het),
            spearman_rho_dist=float(rho_dist),
            spearman_rho_baseline=float(rho_base),
            pearson_r_unc=float(r_unc),
            AUSE_err_unc=rc_unc['ause_err'],
            AUSE_err_baseline=rc_base['ause_err'],
            AUSE_pck_unc=rc_unc['ause_pck'],
            AUSE_pck_baseline=rc_base['ause_pck'],
            quintile_1_err=quintile_errors[0],
            quintile_3_err=quintile_errors[2],
            quintile_5_err=quintile_errors[4],
        )

        logger.info(
            f'\n================ Uncertainty-Aware Pose Evaluation ================\n'
            f'  Keypoints Evaluated : {n_total} (Visible: {n_vis}, Occluded: {n_occ})\n'
            f'  PCK@0.1             : All={pck_all*100:.2f}% | Vis={pck_vis*100:.2f}% | Occ={pck_occ*100:.2f}%\n'
            f'  Mean Norm Error     : All={mean_err_all:.4f} | Vis={mean_err_vis:.4f} | Occ={mean_err_occ:.4f}\n'
            f'  Occ/Vis Unc Ratio   : {occ_vis_ratio:.2f}x (Occ AUROC: {occ_auroc_unc:.3f} vs Base {occ_auroc_base:.3f})\n'
            f'  Spearman rho (Err)  : Combined={rho_unc:.4f} | Het={rho_het:.4f} | DistStd={rho_dist:.4f} | BaseConf={rho_base:.4f}\n'
            f'  AUSE (PCK@0.1)      : Unc={rc_unc["ause_pck"]:.4f} vs BaseConf={rc_base["ause_pck"]:.4f} (lower is better)\n'
            f'  Quintile Error      : Q1(confident)={quintile_errors[0]:.4f} -> Q5(uncertain)={quintile_errors[4]:.4f}\n'
            f'==================================================================='
        )

        return metrics
