"""Merge OCHuman val and test into one COCO file, with the official split kept as a split file.

OCHuman ships val and test as separate COCO-format files whose image ids both start at 1. The dump and
analysis scripts work on one annotation file, so test image and annotation ids are offset to keep them
unique. The split file lists val image ids as calibration and test image ids as test, so every fitted
quantity uses OCHuman val and every reported value OCHuman test, the dataset's own protocol.
The original files are not modified.

Usage: python scripts/prepare_ochuman.py [--root data/OCHuman]
"""
import argparse
import json
from pathlib import Path

OFFSET = 100_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/OCHuman')
    args = parser.parse_args()
    root = Path(args.root)

    def load(split):
        with open(root / f'ochuman_coco_format_{split}_range_0.00_1.00.json', encoding='utf-8') as f:
            return json.load(f)

    val, test = load('val'), load('test')
    assert max(i['id'] for i in val['images']) < OFFSET and max(a['id'] for a in val['annotations']) < OFFSET
    for img in test['images']:
        img['id'] += OFFSET
    for ann in test['annotations']:
        ann['id'] += OFFSET
        ann['image_id'] += OFFSET

    merged = dict(images=val['images'] + test['images'],
                  annotations=val['annotations'] + test['annotations'],
                  categories=val['categories'])
    out = root / 'annotations'
    out.mkdir(exist_ok=True)
    with open(out / 'ochuman_val_test_merged.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f)
    split = dict(source='OCHuman official val (calibration) and test (test)', id_offset_test=OFFSET,
                 cal=sorted(i['id'] for i in val['images']),
                 test=sorted(i['id'] for i in test['images']))
    with open(out / 'ochuman_split.json', 'w', encoding='utf-8') as f:
        json.dump(split, f)
    print(f'{len(val["images"])} val + {len(test["images"])} test images, '
          f'{len(merged["annotations"])} annotations -> {out}')


if __name__ == '__main__':
    main()
