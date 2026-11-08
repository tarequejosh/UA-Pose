"""COCO AP under different instance-scoring rules, on val-test images only (RQ4).

Scoring rules:
  default     MMPose: bbox_score * mean(conf_k for conf_k > 0.2)
  exp_alpha   the repo's rule: conf_k * exp(-alpha * u_k), alpha chosen on val-cal
  eoks_dist   closed-form expected OKS from the SimCC distribution spread
  eoks_het    closed-form expected OKS from the learned sigma (trained arms)
  pred_oks    bbox_score * mean(predicted OKS_k) (trained arms)

Instance scores change, keypoint locations do not. Instances then go through the same
OKS-NMS and COCOeval as MMPose's CocoMetric. `default` on all images must reproduce the
CocoMetric number, which test_core.py and the sanity table check.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'mmpose'))

import ua_pose  # noqa: E402,F401
from mmpose.evaluation.functional import oks_nms  # noqa: E402
from xtcocotools.coco import COCO  # noqa: E402
from xtcocotools.cocoeval import COCOeval  # noqa: E402

from ua_pose.evaluation.analysis import COCO_SIGMAS, expected_oks, split_images  # noqa: E402

STAT_NAMES = ['AP', 'AP50', 'AP75', 'APM', 'APL', 'AR', 'AR50', 'AR75', 'ARM', 'ARL']
INPUT_AREA = 192 * 256
ALPHA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
GAMMA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def default_instance_scores(scores, bbox_scores, thr=0.2):
    valid = scores > thr
    count = valid.sum(axis=1)
    mean = np.where(count > 0, (scores * valid).sum(axis=1) / np.maximum(count, 1), 0.0)
    return bbox_scores * mean


def build_predictions(dump, instance_scores):
    """Apply per-image OKS-NMS with CocoMetric's settings and return COCO results."""
    by_image = {}
    for i, img_id in enumerate(dump['img_id'].tolist()):
        by_image.setdefault(img_id, []).append(i)

    results = []
    area = np.prod(dump['bbox_scale'], axis=1)
    for img_id, idx in by_image.items():
        # CocoMetric de-duplicates by instance id; ids are unique per box in these dumps.
        instances = [
            dict(
                keypoints=np.concatenate(
                    [dump['keypoints'][i], dump['scores'][i][:, None]], axis=-1),
                score=float(instance_scores[i]),
                area=float(area[i]),
                img_id=img_id,
            ) for i in idx
        ]
        keep = oks_nms(instances, 0.9, sigmas=COCO_SIGMAS)
        for k in keep:
            inst = instances[k]
            results.append(dict(
                image_id=int(img_id),
                category_id=1,
                keypoints=inst['keypoints'].reshape(-1).tolist(),
                score=inst['score'],
            ))
    return results


def coco_stats(coco_gt, results, img_ids=None):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, 'keypoints', COCO_SIGMAS, True)
        evaluator.params.useSegm = None
        if img_ids is not None:
            evaluator.params.imgIds = [int(i) for i in img_ids]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return dict(zip(STAT_NAMES, [float(v) for v in evaluator.stats]))


def score_variants(dump):
    """Return name -> function(param) -> instance scores, with its parameter grid."""
    scores = dump['scores'].astype(np.float64)
    bbox = dump['bbox_score'].astype(np.float64)
    area_crop = np.full(len(scores), fill_value=np.nan)

    variants = {'default': ([None], lambda _: default_instance_scores(scores, bbox))}

    if 'u_dist' in dump:
        u = dump['u_dist'].astype(np.float64)
        unit = u / max(u.mean(), 1e-8)
        variants['exp_alpha_dist'] = (
            ALPHA_GRID,
            lambda a: default_instance_scores(scores * np.exp(-a * unit), bbox))
    return variants, scores, bbox, area_crop


def expected_oks_variant(dump, sigma, rho):
    """bbox_score * mean_k E[OKS_k], with A = rho * crop area in input pixels."""
    area = np.full(len(sigma), rho * INPUT_AREA, dtype=np.float64)
    bbox = dump['bbox_score'].astype(np.float64)
    def fn(gamma):
        return bbox * expected_oks(gamma * sigma, area).mean(axis=1)
    return fn


def select_on_cal(fn, grid, dump, coco_gt, cal_ids):
    best_param, best_ap, curve = None, -1.0, {}
    for param in grid:
        ap = coco_stats(coco_gt, build_predictions(dump, fn(param)), cal_ids)['AP']
        curve[str(param)] = ap
        if ap > best_ap:
            best_ap, best_param = ap, param
    return best_param, curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', default='work_dirs/dumps')
    parser.add_argument('--out', default='work_dirs/results/ap_rescoring.json')
    parser.add_argument('--tags', nargs='+', required=True,
                        help='tag or tag:het,oks to mark learned signals as trained')
    parser.add_argument('--rho', type=float, default=9850.0 / INPUT_AREA,
                        help='Segment area / crop area, from scripts/compute_oks_area.py')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--ann',
                        default='data/coco/annotations/person_keypoints_val2017.json')
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.ann)
    cal_ids, test_ids = split_images(coco_gt.getImgIds())
    rng = np.random.default_rng(2026)
    folds = np.array_split(rng.permutation(test_ids), args.folds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if out_path.exists():
        with open(out_path, encoding='utf-8') as f:
            results = json.load(f)

    for spec in args.tags:
        tag, _, trained = spec.partition(':')
        trained = {t for t in trained.split(',') if t}
        path = Path(args.dump_dir) / f'{tag}.npz'
        if not path.exists():
            print(f'[skip] missing {path}')
            continue
        with np.load(path) as data:
            dump = {k: data[k] for k in data.files}

        rules = {}
        variants, *_ = score_variants(dump)
        rules.update(variants)
        # E[OKS] needs a spread in input pixels. The heatmap head's spread is in normalised
        # coordinates of a temperature-sharpened map, so the formula does not apply to it.
        if 'u_dist' in dump and not tag.startswith(('zs_hm', 'sweep_hm', 'noflip_hm')):
            rules['eoks_dist'] = (GAMMA_GRID, expected_oks_variant(
                dump, dump['u_dist'].astype(np.float64), args.rho))
        if 'het' in trained:
            rules['eoks_het'] = (GAMMA_GRID, expected_oks_variant(
                dump, dump['u_het'].astype(np.float64), args.rho))
        if 'oks' in trained:
            bbox = dump['bbox_score'].astype(np.float64)
            pred = dump['predicted_oks'].astype(np.float64)
            rules['pred_oks'] = ([None], lambda _, b=bbox, p=pred: b * p.mean(axis=1))

        report = dict(all_images={}, test={}, fold_diff={}, selected={})
        baseline_folds = None
        for name, (grid, fn) in rules.items():
            param = grid[0]
            if len(grid) > 1:
                param, curve = select_on_cal(fn, grid, dump, coco_gt, cal_ids)
                report['selected'][name] = dict(param=param, cal_ap=curve)
            preds = build_predictions(dump, fn(param))
            report['test'][name] = coco_stats(coco_gt, preds, test_ids)
            per_fold = np.array([coco_stats(coco_gt, preds, f)['AP'] for f in folds])
            if name == 'default':
                baseline_folds = per_fold
                report['all_images'] = coco_stats(coco_gt, preds)
            diff = per_fold - baseline_folds
            report['fold_diff'][name] = dict(
                mean=float(diff.mean()),
                sd=float(diff.std(ddof=1)),
                num_positive=int((diff > 0).sum()),
                folds=len(diff),
            )
            print(f'{tag:28s} {name:16s} testAP {report["test"][name]["AP"]:.4f} '
                  f'diff {diff.mean():+.4f} +- {diff.std(ddof=1):.4f} '
                  f'param {param}', flush=True)
        results[tag] = report
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
