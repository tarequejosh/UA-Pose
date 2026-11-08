"""Default-score COCO AP on all val2017 images, for the reproduction check (Table 1)."""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_ap import COCO, build_predictions, coco_stats, default_instance_scores  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags', nargs='+')
    parser.add_argument('--dump-dir', default='work_dirs/dumps')
    parser.add_argument('--out', default='work_dirs/results/ap_sanity.json')
    parser.add_argument('--ann',
                        default='data/coco/annotations/person_keypoints_val2017.json')
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.ann)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding='utf-8')) if out.exists() else {}
    for tag in args.tags:
        path = Path(args.dump_dir) / f'{tag}.npz'
        if not path.exists():
            print(f'[skip] {path}')
            continue
        with np.load(path) as data:
            dump = {k: data[k] for k in data.files}
        scores = default_instance_scores(dump['scores'].astype(np.float64),
                                         dump['bbox_score'].astype(np.float64))
        stats = coco_stats(coco_gt, build_predictions(dump, scores))
        results[tag] = stats
        print(tag, {k: round(v, 4) for k, v in stats.items()}, flush=True)
        out.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
