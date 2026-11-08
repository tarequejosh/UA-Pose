"""Regression tests for the defects found while auditing the project.

Each test fails against the original code. Run: python -m pytest tests/test_core.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'mmpose'))

import ua_pose  # noqa: E402,F401
from mmpose.registry import MODELS  # noqa: E402
from ua_pose.evaluation.analysis import (RankedAUROC, RankedAUSE,  # noqa: E402
                                         bootstrap_image_weights, conformal_report,
                                         expected_calibration_error, expected_oks,
                                         fit_logistic, predict_logistic, split_images)
from ua_pose.evaluation.uncertainty_metric import (compute_auroc,  # noqa: E402
                                                   compute_risk_coverage_and_ause)
from ua_pose.heads.ua_simcc_head import CompositionalTokenMixer  # noqa: E402
from ua_pose.losses.uncertainty_losses import HeteroscedasticKLLoss  # noqa: E402

CODEC = dict(type='SimCCLabel', input_size=(192, 256), sigma=6.0, simcc_split_ratio=2.0)


def simcc_head(**overrides):
    cfg = dict(
        type='UASimCCHead', in_channels=512, out_channels=17, input_size=(192, 256),
        in_featuremap_size=(6, 8), simcc_split_ratio=2.0,
        loss=dict(type='HeteroscedasticKLLoss'), decoder=CODEC)
    cfg.update(overrides)
    return MODELS.build(cfg)


def gaussian_labels(centers, bins, sigma=6.0):
    grid = torch.arange(bins, dtype=torch.float32)
    labels = torch.exp(-(grid[None, None] - centers[..., None])**2 / (2 * sigma**2))
    return labels / (sigma * np.sqrt(2 * np.pi))


# ---- metrics ----

def test_auroc_ties_count_half():
    scores = np.zeros(100)
    labels = np.r_[np.ones(50), np.zeros(50)].astype(np.int64)
    assert compute_auroc(scores, labels) == pytest.approx(0.5)


def test_ranked_auroc_matches_reference_and_weights():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 20, 5000).astype(float)
    labels = rng.random(5000) < 0.2 + 0.03 * scores
    ranked = RankedAUROC(scores, labels)
    assert ranked(np.ones(5000, dtype=np.int64)) == pytest.approx(compute_auroc(scores, labels.astype(int)))
    # Integer weights must equal literally repeating the rows.
    weights = rng.integers(0, 4, 5000)
    repeated = compute_auroc(np.repeat(scores, weights), np.repeat(labels, weights).astype(int))
    assert ranked(weights) == pytest.approx(repeated)


def test_ause_is_zero_for_oracle_and_positive_for_random():
    rng = np.random.default_rng(0)
    errors = rng.exponential(size=5000)
    unit = np.ones(5000, dtype=np.int64)
    assert RankedAUSE(errors, errors)(unit) == pytest.approx(0.0, abs=1e-12)
    assert RankedAUSE(rng.random(5000), errors)(unit) > 0.1


def test_risk_coverage_tie_break_does_not_leak_dataset_order():
    errors = np.r_[np.zeros(500), np.ones(500)]
    result = compute_risk_coverage_and_ause(errors, np.zeros(1000))
    # All-tied uncertainty carries no information: retained mean stays near 0.5.
    assert abs(result['err_curve'][0] - 0.5) < 0.1


def test_bootstrap_resamples_whole_images():
    image_index = np.repeat(np.arange(10), 3)
    for weights in bootstrap_image_weights(image_index, 10, 5):
        assert all(len(set(weights[image_index == i])) == 1 for i in range(10))
        assert weights.sum() == 30


def test_split_is_deterministic_and_disjoint():
    cal, test = split_images(range(5000))
    cal2, _ = split_images(reversed(range(5000)))
    assert np.array_equal(cal, cal2)
    assert len(np.intersect1d(cal, test)) == 0 and len(cal) + len(test) == 5000
    assert len(cal) == 1500


def test_logistic_recovers_probabilities_and_lowers_ece():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 1))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(2 * x[:, 0] + 1)))).astype(float)
    beta = fit_logistic(x, y)
    assert beta[0] == pytest.approx(1.0, abs=0.1) and beta[1] == pytest.approx(2.0, abs=0.1)
    assert expected_calibration_error(predict_logistic(beta, x), y) < 0.02


def test_conformal_achieves_target_coverage():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.01, 0.2, 40000)
    err = np.abs(rng.normal(size=40000)) * u
    vis = np.where(rng.random(40000) < 0.1, 1, 2)
    report = conformal_report(u[:20000], u[20000:], err[:20000], err[20000:], vis[20000:])
    assert report['coverage'] == pytest.approx(0.9, abs=0.01)


def test_expected_oks_matches_monte_carlo():
    rng = np.random.default_rng(0)
    sigma, area = 8.0, 12000.0
    for k, kappa in enumerate([0.026, 0.079, 0.107]):
        c = area * (2 * kappa)**2
        err = rng.normal(0, sigma / np.sqrt(2), size=(2000000, 2))
        mc = np.exp(-(err**2).sum(axis=1) / (2 * c)).mean()
        closed = expected_oks(np.array([[sigma]]), np.array([area]), np.array([kappa]))
        assert closed.item() == pytest.approx(mc, abs=1e-3), f'kappa index {k}'


# ---- loss ----

def make_batch(offset_bins):
    n, k = 4, 17
    cx = torch.full((n, k), 192.0)
    cy = torch.full((n, k), 256.0)
    gt = (gaussian_labels(cx, 384), gaussian_labels(cy, 512))
    # Peaked logits whose argmax is offset by a known number of bins.
    pred = (torch.log(gaussian_labels(cx + offset_bins, 384) + 1e-9),
            torch.log(gaussian_labels(cy, 512) + 1e-9))
    return pred, gt, torch.ones(n, k)


def test_oks_target_matches_coco_definition():
    loss = HeteroscedasticKLLoss(vis_weight=1.0, oks_area_px=12000.0)
    pred, gt, weight = make_batch(offset_bins=20)  # 20 bins = 10 input pixels
    kappa = torch.tensor(ua_pose.losses.uncertainty_losses.COCO_OKS_SIGMAS)
    target = torch.exp(-100.0 / (2 * 12000.0 * (2 * kappa)**2))
    # The optimum of BCE is at logit(target); the loss there is the entropy of the target.
    logits = torch.logit(target).expand(4, 17)
    out = loss(pred, gt, weight, log_var=torch.zeros(4, 17, 2), vis_logits=logits)
    entropy = -(target * target.log() + (1 - target) * (1 - target).log()).mean()
    assert out['loss_oks_cal'].item() == pytest.approx(entropy.item(), rel=1e-4)


def test_unlabelled_joints_get_no_oks_loss():
    loss = HeteroscedasticKLLoss(vis_weight=1.0)
    pred, gt, weight = make_batch(offset_bins=0)
    weight[:, 5] = 0.0
    logits = torch.zeros(4, 17)
    base = loss(pred, gt, weight, log_var=torch.zeros(4, 17, 2), vis_logits=logits)['loss_oks_cal']
    logits[:, 5] = 50.0  # A wild output on an unlabelled joint must not matter.
    other = loss(pred, gt, weight, log_var=torch.zeros(4, 17, 2), vis_logits=logits)['loss_oks_cal']
    assert base.item() == pytest.approx(other.item())


def test_loss_terms_share_units():
    """The attenuation and Laplace optima must agree for a realistic residual."""
    loss = HeteroscedasticKLLoss(adaptive_target_sigma=False)
    offset = 12  # bins = 2 sigma_0
    pred, gt, weight = make_batch(offset_bins=offset)
    grid = torch.linspace(-8, 6, 281)
    het, cal = [], []
    for s in grid:
        log_var = torch.zeros(4, 17, 2)
        log_var[..., 0] = s
        out = loss(pred, gt, weight, log_var=log_var)
        het.append(out['loss_het'].item())
        cal.append(out['loss_cal'].item())
    s_het = grid[int(np.argmin(het))].item()
    s_cal = grid[int(np.argmin(cal))].item()
    # Gaussian KL = e^2 / 2 = 2 -> log 2 = 0.69. Laplace optimum = 2 log e = 1.39.
    assert s_het == pytest.approx(np.log(2.0), abs=0.15)
    assert s_cal == pytest.approx(2 * np.log(2.0), abs=0.15)


def test_adaptive_target_widens_only_when_uncertain():
    loss = HeteroscedasticKLLoss(base_sigma=6.0)
    gt = gaussian_labels(torch.full((1, 1), 100.0), 384)
    def width(s):
        target = loss._build_adaptive_gaussian_target(gt, torch.full((1, 1), s))
        grid = torch.arange(384.0)
        mass = target / target.sum(-1, keepdim=True)
        mean = (mass * grid).sum(-1, keepdim=True)
        return ((mass * (grid - mean)**2).sum(-1).sqrt()).item()
    assert width(0.0) == pytest.approx(6.0, abs=0.05)
    assert width(np.log(4.0)) == pytest.approx(12.0, abs=0.1)
    assert width(-8.0) == pytest.approx(0.7 * 6.0, abs=0.05)


# ---- modules ----

def test_mixer_prior_survives_init_weights():
    head = simcc_head()
    head.init_weights()
    routing = head.token_mixer.to_tokens.weight
    # Token 0 routes the 5 face joints: 0.2 each plus small noise.
    assert routing[0, :5].mean().item() == pytest.approx(0.2, abs=0.03)
    assert routing[0, 5:].abs().mean().item() < 0.05


def test_mixer_is_identity_at_init():
    mixer = CompositionalTokenMixer(num_joints=17, feat_dim=3072).eval()
    x = torch.randn(2, 17, 3072)
    with torch.no_grad():
        enhanced, _ = mixer(x)
    assert torch.equal(enhanced, x)


def test_zero_init_mixer_still_learns():
    mixer = CompositionalTokenMixer(num_joints=17, feat_dim=64, token_dim=32)
    enhanced, _ = mixer(torch.randn(2, 17, 64))
    enhanced.sum().backward()
    assert mixer.hidden_to_joint[0].weight.grad.abs().sum() > 0


def test_heatmap_head_keeps_pretrained_key_layout():
    from mmpose.models.heads.heatmap_heads.heatmap_head import HeatmapHead
    ua = MODELS.build(dict(type='UAHeatmapHead', in_channels=2048, out_channels=17,
                           loss=dict(type='HeteroscedasticHeatmapLoss')))
    plain = HeatmapHead(in_channels=2048, out_channels=17)
    plain_keys = set(plain.state_dict())
    ua_keys = set(ua.state_dict())
    assert plain_keys <= ua_keys, sorted(plain_keys - ua_keys)
    assert all(k.startswith('uncertainty_branch') for k in ua_keys - plain_keys)
    assert any(isinstance(m, torch.nn.Dropout2d) for m in ua.modules())


def test_mc_dropout_reaches_attention():
    head = simcc_head().eval()
    head._enable_mc_dropout()
    assert head.token_mixer.token_attn.training
    assert head.feat_dropout.training
    assert not head.deconv_head.training


def test_dist_std_is_in_input_pixels():
    head_cls = type(simcc_head())
    sigma_bins = 12.0
    cx, cy = torch.full((1, 1), 192.0), torch.full((1, 1), 256.0)
    logits_x = torch.log(gaussian_labels(cx, 384, sigma_bins) + 1e-12)
    logits_y = torch.log(gaussian_labels(cy, 512, sigma_bins) + 1e-12)
    std, entropy, peak = head_cls.compute_simcc_distribution_metrics(logits_x, logits_y, 2.0)
    # sigma 12 bins = 6 px per axis, sqrt(36 + 36) = 8.49 px.
    assert std.item() == pytest.approx(np.sqrt(72.0), abs=0.05)
    assert 0.0 < entropy.item() < 1.0 and 0.0 < peak.item() < 1.0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


def test_detach_kl_in_het_stops_only_the_pose_gradient():
    def het_grads(detach):
        torch.manual_seed(0)
        loss = HeteroscedasticKLLoss(detach_kl_in_het=detach, cal_weight=0.0, vis_weight=0.0,
                                     adaptive_target_sigma=False)
        pred_x = torch.randn(2, 17, 384, requires_grad=True)
        pred_y = torch.randn(2, 17, 512, requires_grad=True)
        gt_x = torch.softmax(torch.randn(2, 17, 384), dim=-1)
        gt_y = torch.softmax(torch.randn(2, 17, 512), dim=-1)
        log_var = torch.full((2, 17, 2), -4.0, requires_grad=True)
        out = loss((pred_x, pred_y), (gt_x, gt_y), torch.ones(2, 17), log_var=log_var)
        return torch.autograd.grad(out['loss_het'], [pred_x, log_var], allow_unused=True)

    pose_on, s_on = het_grads(False)
    pose_off, s_off = het_grads(True)
    assert pose_on is not None and pose_on.abs().sum() > 0
    assert pose_off is None or pose_off.abs().sum() == 0
    # The sigma head must still learn, with an unchanged gradient.
    assert s_off.abs().sum() > 0
    assert torch.allclose(s_on, s_off)
