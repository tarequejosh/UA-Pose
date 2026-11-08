"""Where the large gradient norm comes from: per loss term and per module, on training batches.

Exploratory diagnostic. It loads a final checkpoint, draws fixed training images and reports
the L2 gradient norm of each loss term over each parameter group, without clipping or AMP.
No evaluation data is used.

Run this only when no training is active. It holds about 14 GB of commit, and running it beside
a training job on Windows killed that job's data-loader worker (WinError 1455).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ua_pose  # noqa: E402,F401  Registers the plugin modules.
from mmengine.config import Config  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import load_checkpoint  # noqa: E402
from mmpose.registry import DATASETS, MODELS  # noqa: E402


def group_of(name):
    parts = name.split('.')
    if parts[0] == 'backbone':
        return 'backbone'
    return '.'.join(parts[:2]) if len(parts) > 1 else parts[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags', nargs='+', help='work_dirs/<tag> holding epoch_20.pth')
    parser.add_argument('--batches', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', default='work_dirs/results/grad_diagnostic.json')
    args = parser.parse_args()

    results = {}
    for tag in args.tags:
        work_dir = ROOT / 'work_dirs' / tag
        cfg = Config.fromfile(next(work_dir.glob('*.py')))
        init_default_scope('mmpose')
        torch.manual_seed(2026)
        model = MODELS.build(cfg.model)
        load_checkpoint(model, str(work_dir / 'epoch_20.pth'), map_location='cpu', logger='silent')
        model.train()

        dataset_cfg = cfg.train_dataloader.dataset.copy()
        rng = np.random.default_rng(2026)
        size = args.batches * args.batch_size
        dataset_cfg['indices'] = rng.choice(100_000, size=size, replace=False).tolist()
        dataset = DATASETS.build(dataset_cfg)

        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        norms = defaultdict(lambda: defaultdict(list))
        for b in range(args.batches):
            samples = [dataset[i] for i in range(b * args.batch_size, (b + 1) * args.batch_size)]
            data = model.data_preprocessor(pseudo_collate(samples), training=True)
            losses = model.loss(data['inputs'], data['data_samples'])
            terms = {k: v for k, v in losses.items()
                     if 'loss' in k and isinstance(v, torch.Tensor) and v.requires_grad}
            terms['total'] = sum(terms.values())
            for term, value in terms.items():
                grads = torch.autograd.grad(value, [p for _, p in params],
                                            retain_graph=True, allow_unused=True)
                squared = defaultdict(float)
                for (name, _), grad in zip(params, grads):
                    if grad is not None:
                        squared[group_of(name)] += float(grad.pow(2).sum())
                norms[term]['ALL'].append(float(np.sqrt(sum(squared.values()))))
                norms[term]['value'].append(float(value.detach()))
                for group, s in squared.items():
                    norms[term][group].append(float(np.sqrt(s)))
        results[tag] = {term: {g: float(np.median(v)) for g, v in groups.items()}
                        for term, groups in norms.items()}
        # mmengine logs a 50-iteration running mean of grad_norm, so tails matter for comparison.
        total = np.array(norms['total']['ALL'])
        results[tag]['total_ALL_summary'] = dict(
            batches=len(total), mean=float(total.mean()), median=float(np.median(total)),
            p10=float(np.percentile(total, 10)), p90=float(np.percentile(total, 90)),
            max=float(total.max()))
        print(tag, results[tag]['total_ALL_summary'])
        for term, groups in results[tag].items():
            shown = {g: round(v, 3) for g, v in groups.items()}
            print(f'  {term}: {shown}', flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
