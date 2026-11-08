"""Generate person boxes for val2017 with a public torchvision detector, and score them.

The file most MMPose results use (COCO_val2017_detections_AP_H_56_person.json) is hosted
behind OneDrive/Google Drive folders. This gives a reproducible alternative. Its AP is
not the same as the published 56.4 box AP, so pose AP here is not directly comparable
with model-zoo numbers. The paper reports the detector's box AP next to pose AP.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import torch
import torchvision
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/coco')
    parser.add_argument('--out', default='data/coco/person_detection_results/'
                                          'val2017_fasterrcnn_r50_fpn_v2.json')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    root = Path(args.data_root)
    coco = COCO(str(root / 'annotations/person_keypoints_val2017.json'))
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=args.score_thr)
    model.to(args.device).eval()
    print(f'torchvision {torchvision.__version__}, weights {weights.name}')

    results = []
    img_ids = sorted(coco.getImgIds())
    with torch.no_grad():
        for n, img_id in enumerate(img_ids):
            info = coco.loadImgs(img_id)[0]
            img = read_image(str(root / 'val2017' / info['file_name']),
                             mode=ImageReadMode.RGB).float().to(args.device) / 255.0
            pred = model([img])[0]
            keep = pred['labels'] == 1  # COCO person
            for box, score in zip(pred['boxes'][keep].tolist(),
                                  pred['scores'][keep].tolist()):
                x1, y1, x2, y2 = box
                results.append(dict(
                    image_id=img_id, category_id=1, score=float(score),
                    bbox=[x1, y1, x2 - x1, y2 - y1]))
            if (n + 1) % 500 == 0:
                print(f'  {n + 1}/{len(img_ids)}', flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f)

    # Box AP against the same person annotations.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        evaluator = COCOeval(coco, coco.loadRes(results), 'bbox')
        evaluator.params.catIds = [1]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    summary = dict(
        file=str(out),
        detector='torchvision fasterrcnn_resnet50_fpn_v2 COCO_V1',
        score_thr=args.score_thr,
        num_boxes=len(results),
        person_box_AP=float(evaluator.stats[0]),
        person_box_AP50=float(evaluator.stats[1]),
    )
    with open(out.with_suffix('.summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
