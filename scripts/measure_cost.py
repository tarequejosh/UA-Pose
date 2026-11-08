"""Parameters, FLOPs and latency for the cost table (RQ3).

Latency: batch size 1, 50 warm-up runs, 300 timed runs, median and p90, with CUDA
synchronised around the timer. Timed section is backbone + head.predict without data
loading, with flip-test off so "one pass" means exactly one forward through the model.
Run this with no training job on the GPU. Other desktop apps still share it, so absolute
numbers carry that noise; the ratio between modes is the robust quantity.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'mmpose'))

import ua_pose  # noqa: E402,F401
from mmengine.analysis import get_model_complexity_info  # noqa: E402
from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.structures import InstanceData  # noqa: E402
from mmpose.registry import MODELS  # noqa: E402
from mmpose.structures import PoseDataSample  # noqa: E402


def make_sample():
    sample = PoseDataSample()
    sample.set_metainfo(dict(
        input_center=np.array([96.0, 128.0], dtype=np.float32),
        input_scale=np.array([192.0, 256.0], dtype=np.float32),
        input_size=(192, 256),
        flip_indices=list(range(17)),
    ))
    gt = InstanceData()
    gt.bboxes = np.array([[0.0, 0.0, 192.0, 256.0]], dtype=np.float32)
    gt.bbox_scores = np.array([1.0], dtype=np.float32)
    sample.gt_instances = gt
    return sample


def time_mode(model, device, mode, mc_samples, warmup, runs):
    model.test_cfg['uncertainty_mode'] = mode
    model.test_cfg['mc_samples'] = mc_samples
    inputs = torch.randn(1, 3, 256, 192, device=device)
    samples = [make_sample()]
    times = []
    with torch.no_grad():
        for i in range(warmup + runs):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(inputs, samples)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - start)
    times = np.array(times) * 1000.0
    return dict(median_ms=float(np.median(times)), p90_ms=float(np.percentile(times, 90)))


def count(module):
    return int(sum(p.numel() for p in module.parameters())) if module is not None else 0


def main():
    init_default_scope('mmpose')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--out', default='work_dirs/results/cost.json')
    parser.add_argument('--mc-samples', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--runs', type=int, default=300)
    parser.add_argument('--cpu-runs', type=int, default=100)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.backbone.pop('init_cfg', None)
    cfg.model.test_cfg['flip_test'] = False
    cfg.model.test_cfg['calibrate_scores'] = False

    model = MODELS.build(cfg.model).eval()
    head = model.head
    report = dict(
        tag=args.tag,
        params_total=count(model),
        params_backbone=count(model.backbone),
        params_head=count(head),
        params_mixer=count(getattr(head, 'token_mixer', None)),
        params_uncertainty=count(getattr(head, 'uncertainty_head', None))
        + count(getattr(head, 'unc_proj', None))
        + count(getattr(head, 'uncertainty_branch', None)),
    )

    # FLOPs of one forward. The deterministic `forward` runs the mixer and uncertainty
    # branch through forward_with_uncertainty, so their cost is included.
    original = model.forward
    def tensor_forward(inputs):
        return head.forward(model.extract_feat(inputs))
    model.forward = tensor_forward
    info = get_model_complexity_info(model, (3, 256, 192), show_table=False, show_arch=False)
    model.forward = original
    report['flops'] = float(info['flops'])
    report['params_checked'] = float(info['params'])

    modes = dict(single_pass='dist_std', mc_dropout='mc_dropout')
    report['latency'] = {}
    if torch.cuda.is_available():
        model.to('cuda:0')
        report['gpu'] = torch.cuda.get_device_name(0)
        report['latency']['gpu'] = {
            name: time_mode(model, 'cuda:0', mode, args.mc_samples, args.warmup, args.runs)
            for name, mode in modes.items()}
        model.to('cpu')

    torch.set_num_threads(1)
    report['latency']['cpu_1thread'] = {
        name: time_mode(model, 'cpu', mode, args.mc_samples, 10, args.cpu_runs)
        for name, mode in modes.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if out_path.exists():
        with open(out_path, encoding='utf-8') as f:
            results = json.load(f)
    results[args.tag] = report
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
