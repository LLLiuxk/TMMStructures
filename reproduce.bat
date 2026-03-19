@echo off
setlocal

echo ==========================================================
echo  TMMStructures - Microstructure Reproduction Tool
echo ==========================================================
echo.

set /p ds_path="1. Enter dataset directory [Default: Output/dataset/batch_1]: "
if "%ds_path%"=="" set ds_path=Output/dataset/batch_1

echo.
set /p s_id="2. Enter Sample ID or Sequence Number (e.g. 42 or sample_0042): "

if "%s_id%"=="" (
    echo [Error] No ID entered. Exiting.
    pause
    exit /b
)

echo.
echo [Action] Calling verification script for ID: %s_id%
python verify_reproducibility.py %s_id% %ds_path%

echo.
echo [Done] You can find the 'repro_sample_XXXX.png' in the root directory.
pause
