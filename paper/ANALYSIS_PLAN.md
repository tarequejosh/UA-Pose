# Analysis plan (fixed before any result was inspected)

Written 2026-11-05, before the first benchmark run. Any later change is logged at
the bottom with its reason, so the paper can separate planned from exploratory work.

## Research questions

- **RQ1.** On pretrained top-down pose models, do signals available in one forward
  pass (peak confidence, 1D/2D distribution spread, entropy, flip disagreement) match
  20-pass MC-Dropout at detecting keypoint failure and occlusion?
- **RQ2.** Does fine-tuning a learned uncertainty head on SimCC improve on those free
  signals, and what does each component add (heteroscedastic + calibration terms,
  OKS head, compositional token mixer, adaptive target)?
- **RQ3.** What does the added machinery cost in AP, parameters, FLOPs and latency?
- **RQ4.** Does uncertainty-based rescoring raise COCO AP on held-out images?

## Data and splits

- COCO val2017, all images with labelled person keypoints, ground-truth boxes for
  keypoint-level metrics (occlusion flags need GT correspondence).
- Keypoints with v=0 are excluded. v=2 is "visible", v=1 is "occluded".
- Images are split once with seed 2026: 30% **val-cal**, 70% **val-test**. Every fitted
  or selected quantity (Platt parameters, fusion weights, rescoring alpha, conformal
  quantile, area ratio) uses val-cal only. Every reported metric uses val-test only.
- Training uses COCO train2017 only. val2017 AP is logged but is not used to select
  arms. Reported checkpoints are the final epoch, not the best epoch, to avoid
  selecting on the evaluation set.

## Definitions

- Normalised error: e = ||p - g|| / sqrt(bbox_w * bbox_h). This matches the pilot.
- Failure: e >= 0.1 (complement of PCK@0.1). Sensitivity check: per-keypoint OKS < 0.5.

## Metrics

Primary (keypoint-level, on val-test):
1. Failure AUROC.
2. AUSE on normalised error (area between model and oracle sparsification curves,
   retained fraction 0.1 to 1.0).
3. Occlusion AUROC (v=1 positive vs v=2).

Secondary: Spearman rho with error, ECE and Brier after Platt scaling fitted on val-cal,
split-conformal 90% coverage and mean radius, overall and for v=1.

Uncertainty intervals: image-level bootstrap with 1000 resamples, percentile 95% CI.
Paired comparisons resample the same images for both signals. A difference is called
supported only if its 95% CI excludes zero. Seeds are aggregated as mean and SD.

AP: COCO keypoint AP from xtcocotools with the MMPose default score mode and OKS-NMS.
Reported on val-test images only, for GT boxes and for detector boxes. Rescoring
differences use the mean paired difference across 10 disjoint val-test folds.

## Arms

Zero-shot, no training: SimCC-R50, SimCC-ViPNAS-MobileNetV3, Heatmap-R50 with official
checkpoints. Learned head outputs are excluded because they are untrained.

Fine-tune, 20 epochs from the official checkpoint, same schedule per backbone:
- `base`: KL only (control, also gives trained MC-Dropout).
- `het`: + heteroscedastic and calibration terms, no mixer.
- `het_oks`: + OKS head.
- `mix`: + compositional token mixer.
- `full`: + adaptive target smoothing.

Primary backbone: ViPNAS-MobileNetV3 (the lightweight case). R50 runs `base` and `full`
at minimum. Seeds 0, 1, 2 for `base` and `full` on the primary backbone if time permits;
fewer seeds are reported as such, never pooled silently.

SOCA-PRNet-34: plain-backbone vs SOCA comparison from ImageNet weights. The 210-epoch
schedule may be shortened for one GPU; any shortening is stated, and results are then
not comparable with published 210-epoch numbers.

## Decision rules and honest reporting

- Any null or negative result is reported as such. The paper's conclusions follow the
  numbers, whichever way they go.
- "Lightweight" and "SOTA" are not claimed beyond what the measured cost table supports.
- Published baselines are only quoted for context and are labelled as not re-measured.
- Numbers in the paper are generated from result files by script, not typed by hand.

## Change log

- 2026-11-05, after seeing the zero-shot table. MC-Dropout scored far below the free
  signals at p=0.1, the only rate in the plan. A dropout-rate sweep on the zero-shot
  models (p in 0.05, 0.2, 0.3, 0.5) is added so the comparison is not an artefact of one
  rate. It is labelled exploratory in the paper and every rate is reported, not the best.
- 2026-11-05, same point. All planned dumps use flip-test, so their "free" signals come
  from two backbone passes. No-flip dumps of the three zero-shot models are added so the
  one-pass claim is measured directly rather than inferred.
- 2026-11-05, after the first rescoring run. Selected alpha and gamma sat on the upper edge
  of their grids, so both grids were extended upward. Selection still uses val-cal only.
  E[OKS] was removed for the heatmap model because its spread is not in pixels; the -1.1
  AP seen in that first run was a unit error, not a result.
- 2026-11-05, same point. Fusion's Platt step was fed a probability and is now fed its
  logit. This is a bug fix to a secondary metric (ECE/Brier) and changes no rank metric.
- 2026-11-06, after an interim check of the `full` arm at epoch 5 (all val2017 images, GT
  boxes): AP 0.698 against 0.711 for `base` at the same epoch, with clipped gradient norms
  near 100. The suspected cause is `exp(-s)` multiplying the pose gradient in the
  attenuation term. All planned arms stay unchanged and are reported as planned. One arm is
  added, `full_sg`: `full` with that gradient stopped. It is exploratory, was motivated by
  an evaluation-set number, and is labelled as such in the paper. It was not chosen over any
  other arm, and it does not replace `full`.
- 2026-11-06, while comparing `full` with the control. The arm table was reading AP on all
  val2017 images, against this plan. Arm comparisons now use val-test AP with paired fold
  differences (`scripts/arm_ap.py`). The reproduction table stays on all val2017, a stated
  deviation, because the published numbers it is checked against use all of val2017.
- 2026-11-08, after all runs finished. Seeds 0, 1 and 2 were completed for `base` and `full` on
  ViPNAS-MobileNetV3; every other arm has one seed. The SOCA comparison used 60 epochs (steps at
  48 and 57), as the plan allowed, and is stated as not comparable with 210-epoch results. Final
  keypoint metrics use 1000 bootstrap resamples as planned (interim tables used 200; no claim in
  the text changed between them). One interim scoring run omitted the learned signals because of
  a missing per-tag flag; it was discarded, never used in the paper, and `scripts/score_all.py`
  now derives the flags from the arm definitions. Rescoring was also run on the seed-0 `base` and
  `full` heads of both backbones, adding rules on learned sigma and predicted OKS. Several rules
  were compared on single runs, so the one gain above its fold SD (E[OKS] from sigma on ResNet-50)
  is reported as a lead, not a result.
- 2026-11-08, after all results. The manuscript was reframed from a comparison study into a
  method paper (SpreadCP: spread score, Platt failure probability and spread-scaled conformal
  bounds, all of which were planned signals and metrics). No metric, split, arm or comparison
  changed. One measurement was added: the latency of the method's own operations
  (`scripts/measure_method_overhead.py`). The previous framing is kept as
  `paper/main_analysis_version.tex`.
- 2026-11-08, extension requested for journal submission. RTMPose-t/s/m (published COCO weights) are
  added, evaluation only, with the same split, signals, metrics and MC-Dropout placement (p=0.1 on
  the features before the coordinate classifiers). RTMPose is trained with KLDiscretLoss at
  temperature beta=10, so its trained distribution is softmax(10 * logits); a 30-image smoke test
  showed a plain softmax was near uniform (spread about 92 px for every joint). All signals
  therefore use softmax(beta * logits) with beta read from each model's training loss. The SimCC
  models were trained with beta=1, so their results are unchanged. No performance metric was looked
  at before this rule was fixed. Checkpoint SHA-256 prefixes did not match their filenames, so the
  weights are verified by reproducing published AP instead.
- 2026-11-09, same extension, written before any OCHuman result was computed. OCHuman (official
  COCO-format val and test files) is added as a cross-dataset, heavy-occlusion test of the six
  published models (three SimCC-style families incl. RTMPose t/s/m, plus the heatmap model), with
  COCO-trained weights and no retraining. Its own split is used: every fitted quantity (Platt,
  fusion, conformal quantile) on OCHuman val, every reported value on OCHuman test
  (`scripts/prepare_ochuman.py`, `analyze_results.py --split-file`). Same signals, metrics, failure
  threshold and 1000 bootstrap resamples as COCO. Ground-truth boxes only; no AP is reported on
  OCHuman.
