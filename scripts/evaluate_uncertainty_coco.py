"""Run a pose model over COCO val2017 and dump every per-keypoint signal to disk.

Inference and analysis are separate on purpose: a dump holds raw predictions, so tables
and figures (scripts/analyze_results.py) can be regenerated without touching the GPU, and
the calibration/test split is applied in exactly one place.

Dump contents, for N person instances and K=17 joints:
    keypoints [N,K,2], scores [N,K], signals u_* / peak_prob / predicted_oks [N,K],
    img_id, ann_id, bbox_score [N], and with ground-truth boxes also
    gt_keypoints [N,K,2], gt_vis [N,K], bbox_xywh [N,4], area [N].

With --bbox-file the same model runs on detector boxes, which only supports AP.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'mmpose'))

import ua_pose  # noqa: E402,F401  registers custom modules, applies compat shims
from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import load_checkpoint  # noqa: E402
from mmpose.datasets.datasets.body import CocoDataset  # noqa: E402
from mmpose.registry import MODELS  # noqa: E402

SIGNAL_FIELDS = {
    'u_het': 'uncertainty_het',
    'u_dist': 'uncertainty_dist_std',
    'u_entropy': 'uncertainty_entropy',
    'u_mc': 'uncertainty_mc',
    'u_mc_soft': 'uncertainty_mc_soft',
    'u_flip': 'uncertainty_flip',
    'peak_prob': 'peak_prob',
    'predicted_oks': 'predicted_oks',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tag', required=True, help='Name of the output dump')
    parser.add_argument('--out-dir', default='work_dirs/dumps')
    parser.add_argument('--data-root', default='data/coco/')
    parser.add_argument('--ann-file', default='annotations/person_keypoints_val2017.json',
                        help='Relative to --data-root. Any COCO-format 17-keypoint file, '
                             'e.g. OCHuman annotations/ochuman_coco_format_test_range_0.00_1.00.json')
    parser.add_argument('--img-prefix', default='val2017/', help='Relative to --data-root')
    parser.add_argument('--bbox-file', default=None,
                        help='COCO-format detections. Default: ground-truth boxes')
    parser.add_argument('--num-images', type=int, default=-1,
                        help='Random image subset for quick checks. -1 uses all images')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--mc-samples', type=int, default=20,
                        help='MC-Dropout passes. 0 skips MC-Dropout')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device',
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def report_checkpoint_keys(model, checkpoint_path: str) -> dict:
    """Load a checkpoint and say exactly which model weights it did not provide.

    ``strict=False`` is needed because the uncertainty modules are new, but on its own it
    would hide a mismatch in the pretrained path, so anything outside those modules is
    treated as an error.
    """
    ckpt = load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False,
                           logger='silent')
    state = ckpt.get('state_dict', ckpt)
    model_keys = set(model.state_dict().keys())
    missing = sorted(model_keys - set(state.keys()))
    # Normalisation constants: non-persistent buffers here, and set from the config.
    unexpected = sorted(
        k for k in set(state.keys()) - model_keys
        if k not in ('data_preprocessor.mean', 'data_preprocessor.std'))
    new_modules = ('head.token_mixer', 'head.uncertainty_head', 'head.unc_proj',
                   'head.uncertainty_branch')
    bad = [k for k in missing if not k.startswith(new_modules)]
    if bad or unexpected:
        raise RuntimeError(
            f'Checkpoint mismatch outside the new modules.\n'
            f'missing: {bad[:20]}\nunexpected: {unexpected[:20]}')
    return dict(
        num_loaded=len(model_keys) - len(missing),
        num_missing_new_modules=len(missing),
    )


def collect(outputs, store, has_gt):
    for sample in outputs:
        pred = sample.pred_instances
        meta = sample.metainfo
        store['keypoints'].append(np.asarray(pred.keypoints)[0])
        store['scores'].append(np.asarray(pred.keypoint_scores)[0])
        for key, field in SIGNAL_FIELDS.items():
            if field in pred:
                store[key].append(np.asarray(pred.get(field))[0])
        store['img_id'].append(int(meta['img_id']))
        store['ann_id'].append(int(meta.get('id', -1)))
        store['bbox_score'].append(float(np.asarray(pred.bbox_scores).reshape(-1)[0]))
        store['bbox_scale'].append(np.asarray(meta['input_scale'], dtype=np.float32))

        if has_gt:
            raw = meta['raw_ann_info']
            raw = raw[0] if isinstance(raw, list) else raw
            gt = np.asarray(raw['keypoints'], dtype=np.float32).reshape(-1, 3)
            store['gt_keypoints'].append(gt[:, :2])
            store['gt_vis'].append(gt[:, 2].astype(np.int8))
            store['bbox_xywh'].append(np.asarray(raw['bbox'], dtype=np.float32))
            store['area'].append(float(raw.get('area', 0.0)))
            store['iscrowd'].append(int(raw.get('iscrowd', 0)))


def main():
    init_default_scope('mmpose')
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    test_cfg = cfg.model.test_cfg
    test_cfg['uncertainty_mode'] = 'mc_dropout' if args.mc_samples > 0 else 'dist_std'
    test_cfg['mc_samples'] = args.mc_samples
    test_cfg['calibrate_scores'] = False

    # Pretrained weights come from the checkpoint, so skip the ImageNet download.
    cfg.model.backbone.pop('init_cfg', None)

    model = MODELS.build(cfg.model)
    key_report = report_checkpoint_keys(model, args.checkpoint)
    model.to(args.device).eval()
    print(f'[checkpoint] {key_report}')

    has_gt = args.bbox_file is None
    dataset = CocoDataset(
        data_root=args.data_root,
        data_mode='topdown',
        metainfo=dict(from_file=str(
            PROJECT_ROOT / 'mmpose/configs/_base_/datasets/coco.py')),
        ann_file=args.ann_file,
        data_prefix=dict(img=args.img_prefix),
        bbox_file=args.bbox_file,
        test_mode=True,
        pipeline=cfg.val_dataloader.dataset.pipeline,
    )

    indices = list(range(len(dataset)))
    if args.num_images > 0:
        img_ids = sorted({dataset.get_data_info(i)['img_id'] for i in indices})
        rng = np.random.default_rng(args.seed)
        keep = set(rng.choice(img_ids, size=min(args.num_images, len(img_ids)),
                              replace=False).tolist())
        indices = [i for i in indices if dataset.get_data_info(i)['img_id'] in keep]

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=pseudo_collate,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    print(f'{len(indices)} instances, device {args.device}, T={args.mc_samples}')
    torch.manual_seed(args.seed)
    store = {k: [] for k in ('keypoints', 'scores', 'img_id', 'ann_id', 'bbox_score',
                             'bbox_scale', 'gt_keypoints', 'gt_vis', 'bbox_xywh',
                             'area', 'iscrowd', *SIGNAL_FIELDS)}

    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        with torch.no_grad():
            outputs = model.test_step(batch)
        collect(outputs, store, has_gt)
        if (i + 1) % 25 == 0:
            done = (i + 1) * args.batch_size
            rate = done / (time.perf_counter() - t0)
            print(f'  {done}/{len(indices)} ({rate:.0f} inst/s)', flush=True)
    elapsed = time.perf_counter() - t0

    arrays = {k: np.asarray(v) for k, v in store.items() if len(v) > 0}
    np.savez_compressed(out_dir / f'{args.tag}.npz', **arrays)
    meta = dict(
        tag=args.tag,
        config=args.config,
        checkpoint=args.checkpoint,
        bbox_file=args.bbox_file,
        data_root=args.data_root,
        ann_file=args.ann_file,
        mc_samples=args.mc_samples,
        num_instances=len(indices),
        checkpoint_keys=key_report,
        wall_seconds=elapsed,
        note='wall_seconds includes data loading and is not a latency measurement. '
             'Use scripts/measure_cost.py for cost.',
    )
    with open(out_dir / f'{args.tag}.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved {out_dir / (args.tag + ".npz")} in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
