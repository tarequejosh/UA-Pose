"""Val-test COCO AP for the fine-tuned arms, with paired fold differences from a control.

The analysis plan reports AP on val-test images only. ``ap_sanity.py`` scores all of val2017
because published numbers do, so it is kept for the reproduction check and this script serves
the comparison between arms. Folds match ``rescore_ap.py``.

Usage: arm_ap.py TAG[=CONTROL] ...   A tag without a control only gets its AP.
The detector-box dump ``<tag>_det.npz`` is scored too when it exists.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_ap import (COCO, build_predictions, coco_stats,  # noqa: E402
                        default_instance_scores, split_images)


def predictions(path):
    with np.load(path) as data:
        dump = {k: data[k] for k in data.files}
    scores = default_instance_scores(dump['scores'].astype(np.float64),
                                     dump['bbox_score'].astype(np.float64))
    return build_predictions(dump, scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specs', nargs='+', help='TAG or TAG=CONTROL')
    parser.add_argument('--dump-dir', default='work_dirs/dumps')
    parser.add_argument('--out', default='work_dirs/results/arm_ap.json')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--ann',
                        default='data/coco/annotations/person_keypoints_val2017.json')
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.ann)
    _, test_ids = split_images(coco_gt.getImgIds())
    folds = np.array_split(np.random.default_rng(2026).permutation(test_ids), args.folds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding='utf-8')) if out.exists() else {}
    cache = {}

    def score(name):
        """Whole val-test AP and per-fold AP for one dump, or None if it is missing."""
        if name not in cache:
            path = Path(args.dump_dir) / f'{name}.npz'
            if not path.exists():
                print(f'[skip] {path}')
                cache[name] = None
            else:
                preds = predictions(path)
                cache[name] = dict(
                    AP=coco_stats(coco_gt, preds, test_ids)['AP'],
                    folds=np.array([coco_stats(coco_gt, preds, f)['AP'] for f in folds]))
        return cache[name]

    for spec in args.specs:
        tag, _, control = spec.partition('=')
        report = {}
        for box, suffix in (('gt', ''), ('det', '_det')):
            current = score(tag + suffix)
            if current is None:
                continue
            entry = dict(AP=current['AP'])
            reference = score(control + suffix) if control and control != tag else None
            if reference is not None:
                diff = current['folds'] - reference['folds']
                entry.update(control=control + suffix,
                             diff_mean=float(diff.mean()),
                             diff_sd=float(diff.std(ddof=1)),
                             num_positive=int((diff > 0).sum()),
                             folds=len(diff))
            report[box] = entry
            shown = f'{entry["diff_mean"]:+.4f} +- {entry["diff_sd"]:.4f}' if reference else '--'
            print(f'{tag:24s} {box:4s} AP {entry["AP"]:.4f} diff {shown}', flush=True)
        results[tag] = report
        out.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
