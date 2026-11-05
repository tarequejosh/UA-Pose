@echo off
REM ============================================================================
REM Master Experiment Runner for Windows PC (NVIDIA RTX 5060 + Core Ultra i7)
REM Usage:
REM   run_experiments_windows.bat download   -> Download weights & COCO val2017
REM   run_experiments_windows.bat eval       -> Evaluate pretrained SimCC & Heatmap on 500 COCO images & plot Figs 1-3
REM   run_experiments_windows.bat finetune   -> Fine-tune UA-SimCC & UA-Heatmap (20 epochs with AMP on RTX 5060)
REM   run_experiments_windows.bat train_soca -> Train SOCA-PRNet-34 + UA-SimCC from scratch
REM ============================================================================

SET MODE=%1
IF "%MODE%"=="" SET MODE=eval

IF "%MODE%"=="download" (
    echo [Download] Downloading pretrained MMPose checkpoints and COCO 2017 val dataset...
    python scripts\download_coco_and_weights.py
    GOTO :EOF
)

IF "%MODE%"=="eval" (
    echo [Exp 1/3] Evaluating UA-SimCC + ResNet-50 on 500 COCO val2017 images...
    python scripts\evaluate_uncertainty_coco.py ^
        configs\simcc\ua_simcc_res50_coco_256x192.py ^
        --checkpoint checkpoints\simcc_res50_256x192.pth ^
        --num-images 500 ^
        --mc-samples 20 ^
        --out-dir work_dirs\eval_simcc_res50_500imgs

    echo [Exp 2/3] Evaluating Ultra-Lightweight UA-SimCC + ViPNAS-MobileNetV3 on 500 COCO val2017 images...
    python scripts\evaluate_uncertainty_coco.py ^
        configs\simcc\ua_simcc_vipnas_mbv3_coco_256x192.py ^
        --checkpoint checkpoints\simcc_vipnas_mbv3_256x192.pth ^
        --num-images 500 ^
        --mc-samples 20 ^
        --out-dir work_dirs\eval_simcc_vipnas_mbv3_500imgs

    echo [Exp 3/3] Evaluating UA-Heatmap + PoseResNet-50 on 500 COCO val2017 images...
    python scripts\evaluate_uncertainty_coco.py ^
        configs\heatmap\ua_heatmap_res50_coco_256x192.py ^
        --checkpoint checkpoints\td_hm_res50_256x192.pth ^
        --num-images 500 ^
        --mc-samples 20 ^
        --out-dir work_dirs\eval_heatmap_res50_500imgs

    echo All evaluations and Figures 1, 2, 3 saved under work_dirs\ !
    GOTO :EOF
)

IF "%MODE%"=="finetune" (
    echo [Fine-Tune 1/2] Fine-tuning UA-SimCC + PCT Compositional Mixer + Heteroscedastic Head (ResNet-50)...
    python scripts\train.py configs\simcc\ua_simcc_res50_finetune_20e_coco_256x192.py --amp

    echo [Fine-Tune 2/2] Fine-tuning Ultra-Lightweight UA-SimCC (ViPNAS-MobileNetV3)...
    python scripts\train.py configs\simcc\ua_simcc_vipnas_mbv3_finetune_20e_coco_256x192.py --amp

    echo [Evaluate Fine-Tuned] Running 500-image uncertainty evaluation on fine-tuned UA-SimCC ResNet-50...
    python scripts\evaluate_uncertainty_coco.py ^
        configs\simcc\ua_simcc_res50_finetune_20e_coco_256x192.py ^
        --checkpoint work_dirs\ua_simcc_res50_finetune_20e_coco_256x192\epoch_20.pth ^
        --num-images 500 ^
        --mc-samples 20 ^
        --out-dir work_dirs\eval_finetuned_ua_simcc_res50
    GOTO :EOF
)

IF "%MODE%"=="train_soca" (
    echo [Train] Training SOCA-PRNet-34 + UA-SimCCHead + PCT Compositional Tokens on COCO...
    python scripts\train.py configs\simcc\ua_simcc_soca_prnet34_coco_256x192.py --amp
    GOTO :EOF
)

echo Unknown command "%MODE%". Use one of: download, eval, finetune, train_soca
