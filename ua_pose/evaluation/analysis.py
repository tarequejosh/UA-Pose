"""Statistics for the uncertainty study: held-out fitting and weighted bootstrap.

Bootstrap resamples are represented as integer image weights rather than copied arrays.
With the sort order computed once, each replicate of AUROC or AUSE is then a cumulative
sum, which makes a 1000-resample paired bootstrap over many signals cheap.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr

COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
])

RETAIN_FRACTIONS = np.linspace(0.1, 1.0, 19)


def split_images(img_ids: Sequence[int], cal_fraction: float = 0.3,
                 seed: int = 2026) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic image-level split into (cal, test) ids."""
    ids = np.array(sorted(set(int(i) for i in img_ids)))
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_cal = int(round(cal_fraction * len(ids)))
    return np.sort(ids[perm[:n_cal]]), np.sort(ids[perm[n_cal:]])


# ---------------------------------------------------------------- weighted metrics


class RankedAUROC:
    """AUROC with average-rank tie handling, re-weightable without re-sorting."""

    def __init__(self, scores: np.ndarray, labels: np.ndarray) -> None:
        order = np.argsort(scores, kind='mergesort')
        s = scores[order]
        self.labels = labels[order].astype(bool)
        self.order = order
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])

    def __call__(self, weights: np.ndarray) -> float:
        w = weights[self.order]
        pos = np.add.reduceat(w * self.labels, self.starts).astype(np.float64)
        neg = np.add.reduceat(w * ~self.labels, self.starts).astype(np.float64)
        total_pos, total_neg = pos.sum(), neg.sum()
        if total_pos == 0 or total_neg == 0:
            return float('nan')
        neg_before = np.cumsum(neg) - neg
        return float(np.sum(pos * (neg_before + 0.5 * neg)) / (total_pos * total_neg))


class RankedAUSE:
    """Area between the model and oracle sparsification curves of mean error.

    Curves are mean error over the most-confident retained fraction, for retention in
    [0.1, 1]. Ties in the uncertainty are broken by a fixed random key.
    """

    def __init__(self, uncertainty: np.ndarray, errors: np.ndarray,
                 seed: int = 0) -> None:
        tie = np.random.default_rng(seed).random(len(errors))
        self.order_u = np.lexsort((tie, uncertainty))
        self.order_o = np.argsort(errors, kind='mergesort')
        self.errors = errors

    def _curve(self, order: np.ndarray, weights: np.ndarray) -> np.ndarray:
        w = weights[order].astype(np.float64)
        cum_w = np.cumsum(w)
        cum_e = np.cumsum(w * self.errors[order])
        targets = RETAIN_FRACTIONS * cum_w[-1]
        idx = np.minimum(np.searchsorted(cum_w, targets), len(cum_w) - 1)
        return cum_e[idx] / np.maximum(cum_w[idx], 1.0)

    def curves(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._curve(self.order_u, weights), self._curve(self.order_o, weights)

    def __call__(self, weights: np.ndarray) -> float:
        model, oracle = self.curves(weights)
        gap = model - oracle
        return float(np.sum(0.5 * (gap[1:] + gap[:-1]) * np.diff(RETAIN_FRACTIONS)))


def bootstrap_image_weights(image_index: np.ndarray, num_images: int,
                            num_resamples: int, seed: int = 2026):
    """Yield per-keypoint integer weights for an image-level bootstrap."""
    rng = np.random.default_rng(seed)
    for _ in range(num_resamples):
        counts = np.bincount(rng.integers(0, num_images, num_images),
                             minlength=num_images)
        yield counts[image_index]


def percentile_ci(values: np.ndarray) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float('nan'), float('nan')
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


# ---------------------------------------------------------------- calibration


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-6,
                 iters: int = 60) -> np.ndarray:
    """Newton/IRLS logistic regression. ``x`` [N, D], returns [D + 1] with intercept."""
    design = np.c_[np.ones(len(x)), x]
    beta = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(iters):
        p = np.clip(_sigmoid(design @ beta), 1e-6, 1 - 1e-6)
        grad = design.T @ (y - p) - penalty @ beta
        hess = (design * (p * (1 - p))[:, None]).T @ design + penalty
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _sigmoid(np.c_[np.ones(len(x)), x] @ beta)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def expected_calibration_error(prob: np.ndarray, outcome: np.ndarray,
                               num_bins: int = 15) -> float:
    """Equal-width ECE of predicted probability against a binary outcome."""
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, num_bins - 1)
    ece = 0.0
    for b in range(num_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - outcome[mask].mean())
    return float(ece)


def signal_feature(values: np.ndarray, positive_scale: bool) -> np.ndarray:
    """Feature for Platt scaling: log for spreads, identity for scores."""
    return np.log(values + 1e-4) if positive_scale else values


def conformal_radius(errors_cal: np.ndarray, u_cal: np.ndarray, level: float = 0.9):
    """Split-conformal multiplier q so that error <= q * u has marginal coverage."""
    ratio = errors_cal / u_cal
    n = len(ratio)
    rank = min(int(np.ceil((n + 1) * level)), n)
    return float(np.sort(ratio)[rank - 1])


# ---------------------------------------------------------------- full signal report


def evaluate_signal(unc_cal: np.ndarray, unc_test: np.ndarray,
                    err_cal: np.ndarray, err_test: np.ndarray,
                    vis_test: np.ndarray, positive_scale: bool,
                    threshold: float = 0.1) -> Dict[str, float]:
    """Point metrics for one signal. Fits on cal, reports on test."""
    fail_test = err_test >= threshold
    correct_cal = (err_cal < threshold).astype(np.float64)
    correct_test = (~fail_test).astype(np.float64)

    weights = np.ones(len(err_test), dtype=np.int64)
    out = dict(
        failure_auroc=RankedAUROC(unc_test, fail_test)(weights),
        ause=RankedAUSE(unc_test, err_test)(weights),
        spearman=float(spearmanr(unc_test, err_test).statistic),
    )
    occ_mask = vis_test > 0
    out['occlusion_auroc'] = RankedAUROC(
        unc_test[occ_mask], vis_test[occ_mask] == 1)(weights[occ_mask])

    beta = fit_logistic(signal_feature(unc_cal, positive_scale)[:, None], correct_cal)
    prob = predict_logistic(beta, signal_feature(unc_test, positive_scale)[:, None])
    out['ece'] = expected_calibration_error(prob, correct_test)
    out['brier'] = float(np.mean((prob - correct_test)**2))
    return out


def conformal_report(u_cal: np.ndarray, u_test: np.ndarray, err_cal: np.ndarray,
                     err_test: np.ndarray, vis_test: np.ndarray,
                     level: float = 0.9) -> Dict[str, float]:
    """Coverage and mean radius of a split-conformal region, split by visibility."""
    floor = max(float(np.percentile(u_cal, 1)), 1e-6)
    u_cal = np.maximum(u_cal, floor)
    u_test = np.maximum(u_test, floor)
    q = conformal_radius(err_cal, u_cal, level)
    radius = q * u_test
    covered = err_test <= radius
    occ, vis = vis_test == 1, vis_test == 2
    return dict(
        q=q,
        coverage=float(covered.mean()),
        coverage_visible=float(covered[vis].mean()),
        coverage_occluded=float(covered[occ].mean()),
        mean_radius=float(radius.mean()),
        median_radius=float(np.median(radius)),
        mean_radius_visible=float(radius[vis].mean()),
        mean_radius_occluded=float(radius[occ].mean()),
    )


def expected_oks(sigma_px: np.ndarray, area_px: np.ndarray,
                 kappa: Optional[np.ndarray] = None) -> np.ndarray:
    """Closed-form E[OKS_k] for zero-mean isotropic Gaussian error.

    With total variance sigma^2 split evenly over two axes and c = A * (2 kappa)^2,
    E[exp(-d^2 / (2c))] = (1 + sigma^2 / (2c))^-1.
    """
    if kappa is None:
        kappa = COCO_SIGMAS
    c = area_px[..., None] * (2.0 * kappa)**2
    return 1.0 / (1.0 + sigma_px**2 / (2.0 * c))


__all__ = [
    'COCO_SIGMAS', 'RETAIN_FRACTIONS', 'RankedAUROC', 'RankedAUSE',
    'bootstrap_image_weights', 'conformal_report', 'conformal_radius',
    'evaluate_signal', 'expected_calibration_error', 'expected_oks', 'fit_logistic',
    'percentile_ci', 'predict_logistic', 'rankdata', 'signal_feature', 'split_images',
]
