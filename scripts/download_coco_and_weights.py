"""Windows / Cross-Platform Downloader for COCO 2017 & Pretrained MMPose Weights.

Downloads and extracts:
1. Pretrained MMPose checkpoints for SimCC (ResNet-50, ViPNAS-MobileNetV3) and
   SimpleBaseline Heatmap (PoseResNet-50) into `checkpoints/`.
2. COCO 2017 annotations (`annotations_trainval2017.zip`) and validation images
   (`val2017.zip`), plus optionally `train2017.zip` into `data/coco/`.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

PRETRAINED_WEIGHTS = {
    'simcc_res50_256x192.pth': (
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/'
        'coco/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth'
    ),
    'simcc_vipnas_mbv3_256x192.pth': (
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/'
        'coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth'
    ),
    'td_hm_res50_256x192.pth': (
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/'
        'topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192-04af38ce_20220923.pth'
    ),
}

COCO_URLS = {
    'annotations_trainval2017.zip': (
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    ),
    'val2017.zip': 'http://images.cocodataset.org/zips/val2017.zip',
    'train2017.zip': 'http://images.cocodataset.org/zips/train2017.zip',
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f'[Skip] {dest} already exists.')
        return

    print(f'[Download] {url} -> {dest}')

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, block_num * block_size * 100.0 / total_size)
            sys.stdout.write(f'\r  Progress: {pct:5.1f}%')
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print('\n  Done.')


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f'[Extract] {zip_path} -> {extract_to}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print('  Extracted successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Download Pretrained MMPose Weights & COCO 2017 Dataset')
    parser.add_argument(
        '--data-root',
        default='data/coco',
        help='Target directory for COCO dataset (default: data/coco)',
    )
    parser.add_argument(
        '--ckpt-dir',
        default='checkpoints',
        help='Target directory for pretrained checkpoints (default: checkpoints)',
    )
    parser.add_argument(
        '--weights-only',
        action='store_true',
        help='Only download pretrained MMPose checkpoints',
    )
    parser.add_argument(
        '--include-train2017',
        action='store_true',
        help='Also download COCO train2017.zip (~18 GB) for full training',
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    for filename, url in PRETRAINED_WEIGHTS.items():
        download_file(url, ckpt_dir / filename)

    if args.weights_only:
        return

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # 1. Annotations
    ann_file = data_root / 'annotations' / 'person_keypoints_val2017.json'
    if not ann_file.exists():
        ann_zip = data_root / 'annotations_trainval2017.zip'
        download_file(COCO_URLS['annotations_trainval2017.zip'], ann_zip)
        extract_zip(ann_zip, data_root)

    # 2. val2017 images
    val_dir = data_root / 'val2017'
    if not val_dir.exists():
        val_zip = data_root / 'val2017.zip'
        download_file(COCO_URLS['val2017.zip'], val_zip)
        extract_zip(val_zip, data_root)

    # 3. Optional train2017 images
    if args.include_train2017:
        train_dir = data_root / 'train2017'
        if not train_dir.exists():
            train_zip = data_root / 'train2017.zip'
            download_file(COCO_URLS['train2017.zip'], train_zip)
            extract_zip(train_zip, data_root)

    print('\nAll requested weights and datasets are ready!')


if __name__ == '__main__':
    main()
