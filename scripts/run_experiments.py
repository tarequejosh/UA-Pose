"""Resumable experiment queue for the paper.

Each step is one subprocess with its own log. A step is skipped when its output exists,
so the queue can be stopped and restarted at any time. A failed or stalled step is retried
from its last checkpoint, then reported, and the queue moves on, so one crash overnight
does not waste the rest of the run.

    python scripts/run_experiments.py --list
    python scripts/run_experiments.py --stages zeroshot
    python scripts/run_experiments.py --stages finetune_core dumps
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
MAX_ATTEMPTS = 3
STALL_SECONDS = 25 * 60
GPU_FREE_READINGS = 3
GPU_POLL_SECONDS = 60
DET ='data/coco/person_detection_results/val2017_fasterrcnn_r50_fpn_v2.json'

ZEROSHOT = {
    'zs_simcc_r50': ('configs/simcc/ua_simcc_res50_coco_256x192.py',
                     'checkpoints/simcc_res50_256x192.pth'),
    'zs_simcc_mbv3': ('configs/simcc/ua_simcc_vipnas_mbv3_coco_256x192.py',
                      'checkpoints/simcc_vipnas_mbv3_256x192.pth'),
    'zs_hm_r50': ('configs/heatmap/ua_heatmap_res50_coco_256x192.py',
                  'checkpoints/td_hm_res50_256x192.pth'),
}

FINETUNE = {
    'mbv3': 'configs/simcc/ua_simcc_vipnas_mbv3_finetune_20e_coco_256x192.py',
    'r50': 'configs/simcc/ua_simcc_res50_finetune_20e_coco_256x192.py',
}

# Ablation. `base` is the control: plain KL through the same head and schedule.
ARMS = {
    'base': dict(mixer=False, adaptive=False, het=0.0, cal=0.0, oks=0.0),
    'het': dict(mixer=False, adaptive=False, het=0.5, cal=0.2, oks=0.0),
    'het_oks': dict(mixer=False, adaptive=False, het=0.5, cal=0.2, oks=0.2),
    'mix': dict(mixer=True, adaptive=False, het=0.5, cal=0.2, oks=0.2),
    'full': dict(mixer=True, adaptive=True, het=0.5, cal=0.2, oks=0.2),
    # Exploratory, added after an interim result. See paper/ANALYSIS_PLAN.md change log.
    'full_sg': dict(mixer=True, adaptive=True, het=0.5, cal=0.2, oks=0.2, detach_kl=True),
}

# (backbone, arm, seed), in priority order.
RUNS_CORE = [
    ('mbv3', 'base', 0), ('mbv3', 'full', 0), ('mbv3', 'full_sg', 0),
    ('r50', 'base', 0), ('r50', 'full', 0),
]
RUNS_ABLATION = [('mbv3', 'het', 0), ('mbv3', 'het_oks', 0), ('mbv3', 'mix', 0)]
RUNS_SEEDS = [
    ('mbv3', 'base', 1), ('mbv3', 'full', 1),
    ('mbv3', 'base', 2), ('mbv3', 'full', 2),
]

# SOCA-PRNet comparison from ImageNet weights, same head and shortened schedule.
SOCA_EPOCHS = 60
SOCA_RUNS = {
    'r34_base': 'configs/simcc/ua_simcc_res34_coco_256x192.py',
    'soca_base': 'configs/simcc/ua_simcc_soca_prnet34_coco_256x192.py',
}


def arm_options(arm):
    a = ARMS[arm]
    return [
        f'model.head.use_compositional_mixer={a["mixer"]}',
        f'model.head.loss.adaptive_target_sigma={a["adaptive"]}',
        f'model.head.loss.het_weight={a["het"]}',
        f'model.head.loss.cal_weight={a["cal"]}',
        f'model.head.loss.vis_weight={a["oks"]}',
        f'model.head.loss.detach_kl_in_het={a.get("detach_kl", False)}',
    ]


def trained_signals(arm):
    a = ARMS[arm]
    signals = []
    if a['het'] > 0:
        signals.append('het')
    if a['oks'] > 0:
        signals.append('oks')
    return signals


def eval_steps(tag, config, checkpoint, options=()):
    opts = ['--cfg-options', *options] if options else []
    return [
        (f'dump {tag} gt', f'work_dirs/dumps/{tag}.npz',
         [PY, 'scripts/evaluate_uncertainty_coco.py', config, '--checkpoint', checkpoint,
          '--tag', tag, '--mc-samples', '20', *opts]),
        (f'dump {tag} det', f'work_dirs/dumps/{tag}_det.npz',
         [PY, 'scripts/evaluate_uncertainty_coco.py', config, '--checkpoint', checkpoint,
          '--tag', f'{tag}_det', '--bbox-file', DET, '--mc-samples', '0', *opts]),
    ]


def run_steps(runs):
    steps = []
    for backbone, arm, seed in runs:
        tag = f'ft_{backbone}_{arm}_s{seed}'
        work_dir = f'work_dirs/{tag}'
        checkpoint = f'{work_dir}/epoch_20.pth'
        steps.append((
            f'train {tag}', checkpoint,
            [PY, 'scripts/train.py', FINETUNE[backbone], '--amp', '--resume', 'auto',
             '--work-dir', work_dir, '--cfg-options', f'randomness.seed={seed}',
             *arm_options(arm)]))
        steps.extend(eval_steps(tag, FINETUNE[backbone], checkpoint, arm_options(arm)))
    return steps


def soca_steps():
    steps = []
    schedule = [
        f'train_cfg.max_epochs={SOCA_EPOCHS}', 'train_cfg.val_interval=10',
        'optim_wrapper.optimizer.lr=0.0005',
        f'param_scheduler.1.end={SOCA_EPOCHS}',
        f'param_scheduler.1.milestones=[{int(SOCA_EPOCHS * 0.8)},{int(SOCA_EPOCHS * 0.95)}]',
        'default_hooks.checkpoint.interval=10',
        'default_hooks.checkpoint.save_best=None',
        'default_hooks.checkpoint.max_keep_ckpts=1',
    ]
    for name, config in SOCA_RUNS.items():
        tag = f'im_{name}_s0'
        work_dir = f'work_dirs/{tag}'
        checkpoint = f'{work_dir}/epoch_{SOCA_EPOCHS}.pth'
        steps.append((
            f'train {tag}', checkpoint,
            [PY, 'scripts/train.py', config, '--amp', '--resume', 'auto',
             '--work-dir', work_dir, '--cfg-options', 'randomness.seed=0',
             *schedule, *arm_options('base')]))
        steps.extend(eval_steps(tag, config, checkpoint, arm_options('base')))
    return steps


def zeroshot_steps():
    steps = [('detector boxes', DET, [PY, 'scripts/generate_detections.py'])]
    for tag, (config, checkpoint) in ZEROSHOT.items():
        steps.extend(eval_steps(tag, config, checkpoint))
    return steps


def cost_steps():
    """FLOPs and latency, to be run with no training job on the GPU."""
    steps = []
    for backbone, config in FINETUNE.items():
        for arm in ('base', 'het_oks', 'full'):
            tag = f'{backbone}_{arm}'
            steps.append((f'cost {tag}', f'work_dirs/results/.cost_{tag}.done',
                          [PY, 'scripts/measure_cost.py', config, '--tag', tag,
                           '--cfg-options', *arm_options(arm)]))
    for name, config in SOCA_RUNS.items():
        steps.append((f'cost {name}', f'work_dirs/results/.cost_{name}.done',
                      [PY, 'scripts/measure_cost.py', config, '--tag', name,
                       '--cfg-options', *arm_options('base')]))
    steps.append(('cost hm_r50', 'work_dirs/results/.cost_hm_r50.done',
                  [PY, 'scripts/measure_cost.py',
                   'configs/heatmap/ua_heatmap_res50_coco_256x192.py', '--tag', 'hm_r50']))
    # Unmodified MMPose models: the true reference, with no unused uncertainty modules.
    zoo = 'mmpose/configs/body_2d_keypoint/simcc/coco'
    for tag, config in (('mbv3_plain', f'{zoo}/simcc_vipnas-mbv3_8xb64-210e_coco-256x192.py'),
                        ('r50_plain', f'{zoo}/simcc_res50_8xb64-210e_coco-256x192.py')):
        steps.append((f'cost {tag}', f'work_dirs/results/.cost_{tag}.done',
                      [PY, 'scripts/measure_cost.py', config, '--tag', tag]))
    return steps


STAGES = {
    'zeroshot': zeroshot_steps,
    'cost': cost_steps,
    'finetune_core': lambda: run_steps(RUNS_CORE),
    'ablation': lambda: run_steps(RUNS_ABLATION),
    'seeds': lambda: run_steps(RUNS_SEEDS),
    'soca': soca_steps,
}


def analysis_tags():
    """Dump tags for analyze_results.py / rescore_ap.py, with their trained signals."""
    tags = list(ZEROSHOT)
    for backbone, arm, seed in RUNS_CORE + RUNS_ABLATION + RUNS_SEEDS:
        trained = ','.join(trained_signals(arm))
        tags.append(f'ft_{backbone}_{arm}_s{seed}:{trained}' if trained
                    else f'ft_{backbone}_{arm}_s{seed}')
    tags.extend(f'im_{name}_s0' for name in SOCA_RUNS)
    return tags


def gpu_memory_used_mib():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    return int(output.split()[0])


def wait_for_gpu(limit_mib):
    """Block until no other training job holds the GPU.

    Sharing is far worse than taking turns: when two jobs exceed the 8 GB of VRAM, Windows
    pages it to system RAM. Measured on this machine, a ViPNAS step went from 0.25 s to
    about 2 s and a concurrent YOLO epoch from 5 min to over 120 min. Several readings in
    a row are required so a brief dip does not start a run on top of another job.
    """
    if limit_mib <= 0:
        return
    announced = False
    free_readings = 0
    while free_readings < GPU_FREE_READINGS:
        used = gpu_memory_used_mib()
        free_readings = free_readings + 1 if used < limit_mib else 0
        if free_readings == 0 and not announced:
            print(f'[gpu] {time.strftime("%H:%M:%S")} {used} MiB in use, waiting for '
                  f'less than {limit_mib} MiB', flush=True)
            announced = True
        if free_readings < GPU_FREE_READINGS:
            time.sleep(GPU_POLL_SECONDS)


def run_with_watchdog(command, log_path):
    """Run one step, killing its whole process tree if its log stops growing.

    A crashed DataLoader worker can leave the parent waiting forever with no exit code,
    so a return code alone cannot detect failure. Training resumes from its last
    checkpoint on the next attempt, so a kill loses at most a few epochs.
    """
    with open(log_path, 'a', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                return process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                log.flush()
                idle = time.time() - os.path.getmtime(log_path)
                if idle > STALL_SECONDS:
                    print(f'[watchdog] no log output for {idle / 60:.0f} min, '
                          f'killing pid {process.pid}', flush=True)
                    subprocess.call(['taskkill', '/T', '/F', '/PID', str(process.pid)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    process.wait()
                    return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', nargs='+', default=list(STAGES), choices=list(STAGES))
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--print-tags', action='store_true')
    parser.add_argument('--gpu-free-mib', type=int, default=4500,
                        help='Start a step only when GPU memory in use is below this. '
                             '0 disables the check')
    args = parser.parse_args()

    if args.print_tags:
        print(' '.join(analysis_tags()))
        return

    log_dir = Path('logs/queue')
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for stage in args.stages:
        for name, output, command in STAGES[stage]():
            if Path(output).exists():
                print(f'[done] {name}', flush=True)
                continue
            if args.list:
                print(f'[todo] {stage}: {name}')
                continue
            log_path = log_dir / (name.replace(' ', '_') + '.log')
            start = time.time()
            for attempt in range(1, MAX_ATTEMPTS + 1):
                wait_for_gpu(args.gpu_free_mib)
                print(f'[start] {time.strftime("%H:%M:%S")} {name} attempt {attempt}',
                      flush=True)
                code = run_with_watchdog(command, log_path)
                if code == 0:
                    break
                print(f'[retry] {name} exit={code} on attempt {attempt}', flush=True)
                time.sleep(30)
            minutes = (time.time() - start) / 60
            if code == 0 and output.endswith('.done'):
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).touch()
            if code != 0:
                failures.append(name)
                print(f'[FAILED] {name} exit={code} after {minutes:.1f} min, '
                      f'see {log_path}', flush=True)
            else:
                print(f'[ok] {name} in {minutes:.1f} min', flush=True)
    print(f'QUEUE FINISHED. failures={failures}', flush=True)


if __name__ == '__main__':
    main()
