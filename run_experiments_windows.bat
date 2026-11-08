@echo off
REM ============================================================================
REM Windows entry point for the experiment queue (scripts\run_experiments.py).
REM Every step skips itself when its output exists, so any stage can be re-run.
REM
REM Usage, from the activated environment:
REM   run_experiments_windows.bat download   -> Weights, COCO annotations, val2017
REM   run_experiments_windows.bat eval       -> Pretrained models on all val2017 images
REM   run_experiments_windows.bat finetune   -> Fine-tuning ablation and seeds
REM   run_experiments_windows.bat train_soca -> SOCA-PRNet-34 vs ResNet-34 from ImageNet
REM   run_experiments_windows.bat cost       -> Params, FLOPs, latency (idle GPU only)
REM ============================================================================

SET MODE=%1
IF "%MODE%"=="" SET MODE=eval

IF "%MODE%"=="download" (
    python scripts\download_coco_and_weights.py
    python scripts\verify_checkpoints.py
    GOTO :EOF
)
IF "%MODE%"=="eval" (
    python scripts\run_experiments.py --stages zeroshot
    GOTO :EOF
)
IF "%MODE%"=="finetune" (
    python scripts\run_experiments.py --stages finetune_core ablation seeds
    GOTO :EOF
)
IF "%MODE%"=="train_soca" (
    python scripts\run_experiments.py --stages soca
    GOTO :EOF
)
IF "%MODE%"=="cost" (
    python scripts\run_experiments.py --stages cost
    GOTO :EOF
)

echo Unknown command "%MODE%". Use one of: download, eval, finetune, train_soca, cost
