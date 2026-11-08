# PowerShell setup for Windows 11 + NVIDIA RTX 5060 (Blackwell, sm_120).
# Usage: powershell -ExecutionPolicy Bypass -File .\setup_windows_rtx5060.ps1
#
# Creates .venv with Python 3.12, PyTorch 2.11 + CUDA 12.8 and the OpenMMLab stack.
# OpenMMLab's packages predate this GPU's PyTorch, so ua_pose/compat.py bridges the gaps.

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Creating virtual environment .venv ..." -ForegroundColor Green
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip setuptools wheel

Write-Host "[2/5] Installing PyTorch with CUDA 12.8 ..." -ForegroundColor Green
& $py -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

Write-Host "[3/5] Installing remaining packages ..." -ForegroundColor Green
& $py -m pip install -r requirements.txt

Write-Host "[4/5] Installing bundled MMPose (without its dependency list) ..." -ForegroundColor Green
& $py -m pip install -e .\mmpose --no-deps --no-build-isolation

Write-Host "[5/5] Running tests ..." -ForegroundColor Green
& $py -m pytest tests\test_core.py -q -W ignore
& $py -W ignore tests\smoke_test.py

Write-Host ""
Write-Host "Setup complete. Activate with .\.venv\Scripts\Activate.ps1, then:" -ForegroundColor Cyan
Write-Host "  run_experiments_windows.bat download" -ForegroundColor Yellow
Write-Host "  run_experiments_windows.bat eval" -ForegroundColor Yellow
Write-Host "  run_experiments_windows.bat finetune" -ForegroundColor Yellow
