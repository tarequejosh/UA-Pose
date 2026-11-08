"""Overhead of the training-free method on top of a SimCC forward pass.

Times exactly what the method adds: softmax over each axis, the distribution spread
(ua_pose/heads/ua_simcc_head.py, compute_simcc_distribution_metrics), the Platt failure probability
and the conformal radius, at batch size 1. Same protocol as scripts/measure_cost.py: 50 warm-up and
300 timed runs, median reported, on the GPU and on one CPU thread.

Default shapes are COCO top-down SimCC at input 192x256 with split ratio 2: 17 joints, 384 x-bins and
512 y-bins. The logits are random, which does not change the cost of these dense operations.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


def method(pred_x, pred_y, grid_x, grid_y, platt_a, platt_b, q_hat):
    px = F.softmax(pred_x, dim=-1)
    py = F.softmax(pred_y, dim=-1)
    mu_x = (px * grid_x).sum(-1, keepdim=True)
    mu_y = (py * grid_y).sum(-1, keepdim=True)
    spread = torch.sqrt((px * (grid_x - mu_x) ** 2).sum(-1) + (py * (grid_y - mu_y) ** 2).sum(-1) + 1e-8)
    p_fail = torch.sigmoid(platt_a * torch.log(spread) + platt_b)
    radius = q_hat * spread
    return spread, p_fail, radius


def time_it(device, args):
    gen = torch.Generator().manual_seed(2026)
    pred_x = torch.randn(1, args.joints, args.x_bins, generator=gen).to(device)
    pred_y = torch.randn(1, args.joints, args.y_bins, generator=gen).to(device)
    grid_x = (torch.arange(args.x_bins, dtype=torch.float32) / args.split_ratio).to(device)
    grid_y = (torch.arange(args.y_bins, dtype=torch.float32) / args.split_ratio).to(device)
    params = (1.3, -4.0, 2.5)  # Values do not affect timing.
    sync = torch.cuda.synchronize if device == 'cuda' else (lambda: None)
    times = []
    with torch.no_grad():
        for i in range(args.warmup + args.runs):
            sync()
            start = time.perf_counter()
            method(pred_x, pred_y, grid_x, grid_y, *params)
            sync()
            if i >= args.warmup:
                times.append((time.perf_counter() - start) * 1e3)
    return dict(median_ms=float(np.median(times)), p90_ms=float(np.percentile(times, 90)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joints', type=int, default=17)
    parser.add_argument('--x-bins', type=int, default=384)
    parser.add_argument('--y-bins', type=int, default=512)
    parser.add_argument('--split-ratio', type=float, default=2.0)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--runs', type=int, default=300)
    parser.add_argument('--out', default='work_dirs/results/method_overhead.json')
    args = parser.parse_args()

    result = dict(shapes=dict(joints=args.joints, x_bins=args.x_bins, y_bins=args.y_bins))
    torch.set_num_threads(1)
    result['cpu_1thread'] = time_it('cpu', args)
    if torch.cuda.is_available():
        result['gpu'] = time_it('cuda', args)
        result['gpu_name'] = torch.cuda.get_device_name(0)
    print(json.dumps(result, indent=2))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
