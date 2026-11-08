# SpreadCP: Training-Free Keypoint Uncertainty for Lightweight Pose Estimation

Code, configs and manuscript for **SpreadCP**, a training-free uncertainty module for
coordinate-classification (SimCC-style) pose estimators, built on
[OpenMMLab MMPose](https://github.com/open-mmlab/mmpose).

SpreadCP reads the spread of the 1D coordinate distributions a SimCC head already outputs, maps it to a
calibrated keypoint failure probability (Platt scaling), and uses it to scale split-conformal error
bounds per keypoint. It adds no parameters and no training, and about 0.06 ms per person on one CPU
thread.

The manuscript is in [`paper/main.pdf`](paper/main.pdf); every number in it is generated from result
files by [`scripts/make_paper_assets.py`](scripts/make_paper_assets.py). The evaluation plan and every
later change to it are logged in [`paper/ANALYSIS_PLAN.md`](paper/ANALYSIS_PLAN.md).

## What the experiments show

All metrics are on held-out test images; every fitted quantity (Platt, fusion, conformal quantile,
rescoring parameters) uses a separate calibration split.

| Finding | Where it holds |
|---|---|
| Spread score detects keypoint failures better than 20-pass head MC-Dropout | All 5 SimCC-style models, COCO and OCHuman, every dropout rate tested |
| Spread-scaled conformal bounds keep 90% coverage with a smaller radius than a constant radius | All 5 SimCC-style models, COCO and OCHuman |
| Beats the model's default confidence at failure detection | Original SimCC models; **only matches** it on RTMPose |
| Flags occluded joints better than the default confidence | COCO; **reverses** on OCHuman |
| A trained uncertainty head (heteroscedastic + calibration + OKS terms, token mixer, adaptive target) beats the spread score | **No**: worse on two backbones, three seeds, and about 1 AP point lower on ViPNAS-MobileNetV3 |
| Uncertainty-based instance rescoring raises AP meaningfully | **No** |
| Works on heatmap models | **No**: the default confidence is as good, and conformal radius grows |

Models: SimCC with ViPNAS-MobileNetV3 and ResNet-50, RTMPose-t/s/m, a SimpleBaseline ResNet-50 heatmap
model, plus ResNet-34 and SOCA-PRNet-34 trained from ImageNet weights. Datasets: COCO val2017 (seeded
30/70 calibration/test split) and OCHuman (official val/test).

## Repository layout

```text
ua_pose/                 MMPose plugin
  heads/ua_simcc_head.py   SimCC head with all per-keypoint signals and the trained uncertainty head
  heads/ua_rtmcc_head.py   RTMPose head wrapper with the same signals (no trained parameters)
  heads/ua_heatmap_head.py Heatmap head with spatial spread and head MC-Dropout
  backbones/soca_prnet.py  SOCA-PRNet backbone
  losses/                  Heteroscedastic, calibration and OKS losses
  evaluation/analysis.py   Split, AUROC, AUSE, ECE, conformal and bootstrap utilities
configs/                 SimCC, heatmap and RTMPose configs
scripts/
  run_experiments.py       Resumable queue: zero-shot dumps, fine-tuning arms, seeds, SOCA, cost
  evaluate_uncertainty_coco.py  Dump predictions and every signal for a model on a COCO-format set
  analyze_results.py       Keypoint-level metrics with paired bootstrap intervals
  score_all.py             Runs analyze_results.py on every dump with the right learned-signal flags
  ap_sanity.py, arm_ap.py, rescore_ap.py  AP reproduction, paired fold AP differences, rescoring
  measure_cost.py, measure_method_overhead.py  Parameters, FLOPs, latency
  prepare_ochuman.py       Merges OCHuman val/test and writes the official split file
  make_paper_assets.py     Tables, figures and LaTeX macros for the paper
paper/                   Manuscript, analysis plan, generated tables and figures
tests/test_core.py       Regression tests for defects found during the study
```

## Reproducing

Datasets, checkpoints and results are not in the repository (`data/`, `checkpoints/`, `work_dirs/`
are ignored).

```bash
# Environment (Windows + CUDA 12.8 helper; see requirements.txt for packages)
powershell -ExecutionPolicy Bypass -File .\setup_windows_rtx5060.ps1

# COCO 2017 and the published SimCC / heatmap checkpoints
python scripts/download_coco_and_weights.py --include-train2017

# Experiments (resumable; each step is skipped once its output exists)
python scripts/run_experiments.py --stages zeroshot cost finetune_core ablation seeds soca

# Keypoint metrics (1000 bootstrap resamples), AP checks and rescoring
python scripts/score_all.py --num-boot 1000
python scripts/ap_sanity.py <tags>
python scripts/rescore_ap.py --tags <tags>

# OCHuman (download from the official OCHuman site into data/OCHuman first)
python scripts/prepare_ochuman.py
python scripts/analyze_results.py --dump-dir work_dirs/dumps_ochuman --out-dir work_dirs/results_ochuman \
    --ann data/OCHuman/annotations/ochuman_val_test_merged.json \
    --split-file data/OCHuman/annotations/ochuman_split.json --tags och_simcc_mbv3 ...

# Paper assets and PDF
python scripts/make_paper_assets.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

RTMPose checkpoints are the published MMPose COCO weights (`rtmpose-{tiny,s,m}_simcc-coco_pt-aic-coco_420e-256x192`);
the paper reports that they reproduce the published AP to within 0.2 points.

## Tests

```bash
python -m pytest tests/test_core.py -q
```
