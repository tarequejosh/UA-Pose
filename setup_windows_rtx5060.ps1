# PowerShell Setup Script for Windows 11 + NVIDIA RTX 5060 (Blackwell) + Intel Core Ultra i7
# Usage (in PowerShell or Anaconda Prompt):
#   powershell -ExecutionPolicy Bypass -File .\setup_windows_rtx5060.ps1

$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Setting up UA-PoseEstimation Environment (Windows + RTX 5060)      " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Check if Conda is available; if so, create/activate `ua-pose` environment
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "[1/6] Creating/Activating Conda environment 'ua-pose' (Python 3.10)..." -ForegroundColor Green
    $envExists = conda env list | Select-String -Pattern "\bua-pose\b"
    if (-not $envExists) {
        conda create -n ua-pose python=3.10 -y
    }
    conda activate ua-pose
} else {
    Write-Host "[1/6] Conda not detected; using Python venv '.venv'..." -ForegroundColor Yellow
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    .\.venv\Scripts\Activate.ps1
}

python -m pip install --upgrade pip setuptools wheel

# 2. Install PyTorch with CUDA 12.8 support (required for NVIDIA RTX 5060 Blackwell sm_120)
Write-Host "[2/6] Installing PyTorch + TorchVision with CUDA 12.8 (RTX 5060 Blackwell)..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install OpenMMLab Dependencies (MMEngine, MMCV, MMDetection)
Write-Host "[3/6] Installing OpenMMLab foundation libraries (openmim, mmengine, mmcv)..." -ForegroundColor Green
pip install -U openmim
pip install "mmengine>=0.10.0"

# Try prebuilt mmcv wheel first; fall back to mmcv-lite if Windows CUDA 12.8 wheel isn't cached
try {
    mim install "mmcv>=2.0.0rc4,<2.2.0"
} catch {
    Write-Host "Prebuilt mmcv wheel not found for current Torch/CUDA; installing mmcv>=2.0.0..." -ForegroundColor Yellow
    pip install "mmcv>=2.0.0,<2.2.0"
}

# 4. Install scientific & COCO evaluation packages (using xtcocotools binary wheel for Windows)
Write-Host "[4/6] Installing scientific, visualization, and COCO packages..." -ForegroundColor Green
pip install numpy scipy matplotlib opencv-python pillow tqdm json_tricks munkres chumpy xtcocotools

# 5. Install bundled MMPose in editable mode
Write-Host "[5/6] Installing bundled MMPose and UA-PoseEstimation..." -ForegroundColor Green
pip install -v -e .\mmpose --no-build-isolation
pip install -v -e . --no-build-isolation

# 6. Run Smoke Test to verify GPU, SOCA-PRNet-34, UA-SimCCHead, and UA-HeatmapHead
Write-Host "[6/6] Running Hardware & Architecture Smoke Test..." -ForegroundColor Green
python tests\smoke_test.py

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! Next commands to run on your Windows PC:           " -ForegroundColor Cyan
Write-Host "  1. Download pretrained weights & COCO val2017:                     " -ForegroundColor White
Write-Host "     python scripts\download_coco_and_weights.py                     " -ForegroundColor Yellow
Write-Host "  2. Run 500-image COCO evaluation & generate Figures 1, 2, 3:       " -ForegroundColor White
Write-Host "     .\run_experiments_windows.bat eval                              " -ForegroundColor Yellow
Write-Host "  3. Fine-tune or train Uncertainty-Aware models on RTX 5060:        " -ForegroundColor White
Write-Host "     .\run_experiments_windows.bat finetune                          " -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
