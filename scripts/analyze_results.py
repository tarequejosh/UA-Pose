"""Compute every keypoint-level result in the paper from dumps. No GPU needed.

Usage:
    python scripts/analyze_results.py --tags zs_simcc_r50 zs_simcc_mbv3 ...
Writes work_dirs/results/keypoint_metrics.json and curves.npz.

Splits come from ua_pose.evaluation.analysis.split_images, and everything fitted (Platt
parameters, fusion weights, conformal quantile) only sees val-cal.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the statistics module directly: this script needs neither torch nor mmpose.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'analysis', PROJECT_ROOT / 'ua_pose' / 'evaluation' / 'analysis.py')
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

# name -> (dump key, sign, is a non-negative spread)
SIGNALS = {
    'conf': ('scores', -1.0, False),
    'peak': ('peak_prob', -1.0, False),
    'dist': ('u_dist', 1.0, True),
    'entropy': ('u_entropy', 1.0, True),
    'flip': ('u_flip', 1.0, True),
    'mc': ('u_mc', 1.0, True),
    'mc_soft': ('u_mc_soft', 1.0, True),
    'het': ('u_het', 1.0, True),
    'oks': ('predicted_oks', -1.0, False),
}


def load_dump(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def keypoint_table(dump: dict, signal_names):
    """Flatten labelled keypoints: returns img_id, joint, vis, err, signals."""
    instances = np.arange(len(dump['img_id']))
    keep_inst = (dump['iscrowd'] == 0) & (dump['bbox_xywh'][:, 2] > 0) \
        & (dump['bbox_xywh'][:, 3] > 0)
    instances = instances[keep_inst]
    vis = dump['gt_vis'][instances]
    scale = np.sqrt(dump['bbox_xywh'][instances, 2] * dump['bbox_xywh'][instances, 3])
    err = np.linalg.norm(
        dump['keypoints'][instances] - dump['gt_keypoints'][instances], axis=-1
    ) / scale[:, None]

    mask = vis > 0
    img = np.repeat(dump['img_id'][instances][:, None], vis.shape[1], axis=1)[mask]
    joint = np.tile(np.arange(vis.shape[1]), (len(instances), 1))[mask]
    signals = {}
    for name in signal_names:
        key, sign, _ = SIGNALS[name]
        if key in dump:
            signals[name] = sign * dump[key][instances][mask].astype(np.float64)

    # Per-joint OKS with the COCO definition, for the sensitivity analysis.
    area = dump['area'][instances]
    d2 = np.sum((dump['keypoints'][instances] - dump['gt_keypoints'][instances])**2, axis=-1)
    oks = np.exp(-d2 / (2.0 * np.maximum(area[:, None], 1.0)
                        * (2.0 * A.COCO_SIGMAS)**2))[mask]
    return dict(img=img, joint=joint, vis=vis[mask], err=err[mask], oks=oks, signals=signals)


def fusion_signal(table, cal_mask, trained):
    """Logistic fusion of single-pass signals, fitted on val-cal, adds no inference cost.

    Uses only signals that need one forward pass (with flip-test). MC-Dropout is excluded.
    """
    names = ['conf', 'dist', 'entropy', 'flip']
    if 'het' in trained:
        names.append('het')
    if 'oks' in trained:
        names.append('oks')
    names = [n for n in names if n in table['signals']]
    feats = []
    for name in names:
        positive = SIGNALS[name][2]
        feats.append(A.signal_feature(table['signals'][name], positive))
    x = np.stack(feats, axis=1)
    mean, std = x[cal_mask].mean(0), x[cal_mask].std(0) + 1e-8
    x = (x - mean) / std
    fail = (table['err'] >= 0.1).astype(np.float64)
    beta = A.fit_logistic(x[cal_mask], fail[cal_mask])
    return A.predict_logistic(beta, x), dict(zip(names, beta[1:].tolist())), names


def analyze_dump(tag, dump, trained, num_boot, cal_ids):
    usable = [n for n in SIGNALS
              if n in ('conf', 'peak', 'dist', 'entropy', 'flip', 'mc', 'mc_soft')
              or n in trained]
    table = keypoint_table(dump, usable)
    # The heatmap head has no peak_prob and an untouched OKS logit.
    signals = table['signals']

    cal_mask = np.isin(table['img'], cal_ids)
    test_mask = ~cal_mask
    fused, fusion_weights, fusion_names = fusion_signal(table, cal_mask, trained)
    # Keep it on the logit scale, where Platt scaling (linear in its input) is well posed.
    # Rank-based metrics are unaffected by this monotone transform.
    fused = np.clip(fused, 1e-7, 1 - 1e-7)
    signals['fusion'] = np.log(fused / (1 - fused))

    if 'het' in trained and 'dist' in signals:
        # The repo's "combined" score: equal-weight mean of unit-mean signals.
        het_mean = signals['het'][cal_mask].mean()
        dist_mean = signals['dist'][cal_mask].mean()
        signals['combined'] = 0.5 * signals['het'] / het_mean \
            + 0.5 * signals['dist'] / dist_mean

    err_test, err_cal = table['err'][test_mask], table['err'][cal_mask]
    vis_test = table['vis'][test_mask]

    report = dict(
        tag=tag,
        trained_signals=sorted(trained),
        fusion_inputs=fusion_names,
        fusion_weights=fusion_weights,
        num_keypoints_test=int(test_mask.sum()),
        num_keypoints_cal=int(cal_mask.sum()),
        num_images_test=int(len(np.unique(table['img'][test_mask]))),
        num_visible_test=int((vis_test == 2).sum()),
        num_occluded_test=int((vis_test == 1).sum()),
        pck01_all=float(np.mean(err_test < 0.1)),
        pck01_visible=float(np.mean(err_test[vis_test == 2] < 0.1)),
        pck01_occluded=float(np.mean(err_test[vis_test == 1] < 0.1)),
        mean_err_all=float(err_test.mean()),
        mean_err_visible=float(err_test[vis_test == 2].mean()),
        mean_err_occluded=float(err_test[vis_test == 1].mean()),
        signals={},
    )

    # Bootstrap over val-test images. Same resamples for every signal, so paired.
    images, img_index = np.unique(table['img'][test_mask], return_inverse=True)
    occ_mask = vis_test > 0
    fail_test = err_test >= 0.1
    oks_fail = table['oks'][test_mask] < 0.5

    evaluators = {}
    for name, values in signals.items():
        if np.std(values) < 1e-12:
            continue
        u_test = values[test_mask]
        evaluators[name] = dict(
            failure_auroc=A.RankedAUROC(u_test, fail_test),
            ause=A.RankedAUSE(u_test, err_test),
            occlusion_auroc=A.RankedAUROC(u_test[occ_mask], vis_test[occ_mask] == 1),
        )

    replicates = {n: {m: [] for m in e} for n, e in evaluators.items()}
    for weights in A.bootstrap_image_weights(img_index, len(images), num_boot):
        for name, metric_fns in evaluators.items():
            for metric, fn in metric_fns.items():
                replicates[name][metric].append(fn(weights))
    replicates = {n: {m: np.array(v) for m, v in d.items()}
                  for n, d in replicates.items()}

    curves = {}
    unit = np.ones(len(err_test), dtype=np.int64)
    for name, values in signals.items():
        if name not in evaluators:
            continue
        positive = SIGNALS[name][2] if name in SIGNALS else name == 'combined'
        if name == 'fusion':
            positive = False
        metrics = A.evaluate_signal(values[cal_mask], values[test_mask], err_cal, err_test,
                                    vis_test, positive)
        for metric in ('failure_auroc', 'ause', 'occlusion_auroc'):
            low, high = A.percentile_ci(replicates[name][metric])
            metrics[f'{metric}_ci'] = [low, high]
        # Sensitivity: failure defined by per-joint OKS rather than PCK@0.1.
        metrics['failure_auroc_oks'] = A.RankedAUROC(values[test_mask], oks_fail)(unit)
        # Pure-means ratio is only meaningful for spreads, not for negated scores.
        if positive:
            u_test = values[test_mask]
            metrics['occ_vis_ratio'] = float(
                u_test[vis_test == 1].mean() / max(u_test[vis_test == 2].mean(), 1e-12))
            metrics['conformal'] = A.conformal_report(
                values[cal_mask], u_test, err_cal, err_test, vis_test)
        report['signals'][name] = metrics
        model_c, oracle_c = evaluators[name]['ause'].curves(unit)
        curves[f'{tag}/{name}/model'] = model_c
        curves[f'{tag}/{name}/oracle'] = oracle_c

        order = np.argsort(values[test_mask], kind='mergesort')
        report['signals'][name]['quintile_err'] = [
            float(err_test[s].mean()) for s in np.array_split(order, 5)]

    # Constant-radius reference for conformal efficiency.
    report['conformal_constant'] = A.conformal_report(
        np.ones_like(err_cal), np.ones_like(err_test), err_cal, err_test, vis_test)

    # Paired differences: AUROC gain and AUSE reduction against reference signals.
    report['paired'] = {}
    for ref in ('conf', 'dist', 'mc', 'mc_soft'):
        if ref not in replicates:
            continue
        report['paired'][ref] = {}
        for name in replicates:
            if name == ref:
                continue
            entry = {}
            for metric in ('failure_auroc', 'ause', 'occlusion_auroc'):
                diff = replicates[name][metric] - replicates[ref][metric]
                low, high = A.percentile_ci(diff)
                entry[metric] = dict(mean=float(np.nanmean(diff)), ci=[low, high])
            report['paired'][ref][name] = entry

    return report, curves, replicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', default='work_dirs/dumps')
    parser.add_argument('--out-dir', default='work_dirs/results')
    parser.add_argument('--tags', nargs='+', required=True,
                        help='tag or tag:het,oks to mark learned signals as trained')
    parser.add_argument('--num-boot', type=int, default=1000)
    parser.add_argument('--ann',
                        default='data/coco/annotations/person_keypoints_val2017.json')
    parser.add_argument('--split-file', default=None,
                        help='JSON with "cal" and "test" image ids, e.g. a dataset\'s official '
                             'split. Default: the seeded 30/70 split of --ann images')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file:
        with open(args.split_file, encoding='utf-8') as f:
            split = json.load(f)
        cal_ids, test_ids = np.array(split['cal']), np.array(split['test'])
        assert not set(cal_ids.tolist()) & set(test_ids.tolist()), 'cal and test overlap'
    else:
        with open(args.ann, encoding='utf-8') as f:
            image_ids = [img['id'] for img in json.load(f)['images']]
        cal_ids, test_ids = A.split_images(image_ids)

    results, all_curves = {}, {}
    for spec in args.tags:
        tag, _, trained = spec.partition(':')
        trained = {t for t in trained.split(',') if t}
        path = Path(args.dump_dir) / f'{tag}.npz'
        if not path.exists():
            print(f'[skip] missing {path}')
            continue
        print(f'[analyze] {tag} trained={sorted(trained)}', flush=True)
        report, curves, _ = analyze_dump(tag, load_dump(path), trained, args.num_boot, cal_ids)
        results[tag] = report
        all_curves.update(curves)
        for name, m in report['signals'].items():
            print(f'  {name:9s} failAUROC {m["failure_auroc"]:.4f} '
                  f'AUSE {m["ause"]:.5f} occAUROC {m["occlusion_auroc"]:.4f} '
                  f'rho {m["spearman"]:.3f} ECE {m["ece"]:.4f}', flush=True)

    meta = dict(
        split_file=args.split_file,
        split_seed=None if args.split_file else 2026,
        cal_fraction=None if args.split_file else 0.3,
        num_cal_images=len(cal_ids),
        num_test_images=len(test_ids),
        num_bootstrap=args.num_boot,
    )
    with open(out_dir / 'keypoint_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(dict(meta=meta, results=results), f, indent=2)
    np.savez_compressed(out_dir / 'curves.npz', fractions=A.RETAIN_FRACTIONS, **all_curves)
    print(f'Saved {out_dir / "keypoint_metrics.json"}')


if __name__ == '__main__':
    main()
