"""Measure the mean COCO segment area in input-crop pixels, for the OKS loss target.

GetBBoxCenterScale pads the box by 1.25 and fixes the aspect ratio to the input, so a
crop of (W, H) pixels covers scale_w * scale_h image pixels. The segment area in crop
pixels is therefore area * (W * H) / (scale_w * scale_h). The loss uses the median, and
it also reports the segment-to-box ratio that the rescoring baseline starts from.
"""
import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann',
                        default='data/coco/annotations/person_keypoints_train2017.json')
    parser.add_argument('--input-w', type=int, default=192)
    parser.add_argument('--input-h', type=int, default=256)
    args = parser.parse_args()

    with open(args.ann, encoding='utf-8') as f:
        anns = json.load(f)['annotations']

    aspect = args.input_w / args.input_h
    crop_area, seg_box = [], []
    for ann in anns:
        if ann.get('iscrowd', 0) or ann.get('num_keypoints', 0) == 0:
            continue
        w, h = ann['bbox'][2], ann['bbox'][3]
        if w <= 0 or h <= 0:
            continue
        # Same aspect fix as TopdownAffine, after the 1.25 padding.
        sw, sh = 1.25 * w, 1.25 * h
        if sw > sh * aspect:
            sh = sw / aspect
        else:
            sw = sh * aspect
        crop_area.append(ann['area'] * args.input_w * args.input_h / (sw * sh))
        seg_box.append(ann['area'] / (w * h))

    crop_area = np.array(crop_area)
    print(f'instances: {len(crop_area)}')
    print(f'segment area in crop px^2: mean={crop_area.mean():.0f} '
          f'median={np.median(crop_area):.0f}')
    print(f'segment/box area ratio: mean={np.mean(seg_box):.3f} '
          f'median={np.median(seg_box):.3f}')


if __name__ == '__main__':
    main()
