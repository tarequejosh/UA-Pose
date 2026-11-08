"""Run analyze_results.py over every GT dump on disk, with learned signals flagged correctly.

analyze_results.py only scores a learned signal (sigma, OKS) when its tag is written as
`tag:het,oks`. Typing that list by hand once dropped the learned signals from every fine-tuned
run, so this script derives the flags from the arm definitions in run_experiments.py.

Usage: python scripts/score_all.py [--num-boot 1000] [extra analyze_results.py args]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_experiments import ARMS, trained_signals  # noqa: E402

FT_TAG = re.compile(r'^ft_(?P<backbone>[a-z0-9]+)_(?P<arm>.+)_s(?P<seed>\d+)$')
ORDER = ('zs_', 'noflip_', 'sweep_', 'ft_', 'im_')


def flagged(tag):
    match = FT_TAG.match(tag)
    if match and match['arm'] in ARMS:
        signals = trained_signals(match['arm'])
        return f'{tag}:{",".join(signals)}' if signals else tag
    if tag.startswith('ft_'):
        raise SystemExit(f'unknown arm in {tag}; add it to ARMS in run_experiments.py')
    return tag  # Zero-shot, no-flip, sweep and SOCA runs train no uncertainty head.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', default='work_dirs/dumps')
    parser.add_argument('--dry-run', action='store_true')
    args, rest = parser.parse_known_args()

    tags = sorted((p.stem for p in (ROOT / args.dump_dir).glob('*.npz')
                   if not p.stem.endswith('_det') and p.stem.startswith(ORDER)),
                  key=lambda t: (next(i for i, p in enumerate(ORDER) if t.startswith(p)), t))
    specs = [flagged(t) for t in tags]
    print('scoring:', ' '.join(specs), flush=True)
    if args.dry_run:
        return
    cmd = [sys.executable, str(ROOT / 'scripts' / 'analyze_results.py'),
           '--dump-dir', args.dump_dir, '--tags', *specs, *rest]
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == '__main__':
    main()
