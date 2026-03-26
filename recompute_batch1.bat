@echo off
setlocal enabledelayedexpansion
echo ======================================================
echo  TMMStructures - Full Batch 1 Data Restoration
echo ======================================================
echo.

:: --- USER CONFIGURATION ---
:: Number of CPU cores to use. 
:: Set to a lower number (e.g., 4 or 8) if you want to use your PC for other tasks.
set NUM_WORKERS=8

:: Whether to regenerate radar plots (SLOW)
:: Set to --radar to enable, or leave empty to only compute data
set PLOT_RADARS=--radar

:: Target directory
set TARGET=Output/dataset/batch_1
:: --------------------------

echo [Settings]
echo Target: %TARGET%
echo Workers: %NUM_WORKERS%
echo Radars: %PLOT_RADARS%
echo.

:: 1. Recompute properties
echo [1/2] Computing mechanical/thermal properties...
echo Processes are running on %NUM_WORKERS% cores. System should remain responsive.
python recompute_dataset_mp.py %TARGET% %PLOT_RADARS% --radar-dir %TARGET%/images --num-workers %NUM_WORKERS%

:: 2. Generate updated property coverage summary (Ashby plots)
echo.
echo [2/2] Generating property coverage summary (Ashby plots)...
python plot_property_coverage.py %TARGET%

echo.
echo ======================================================
echo  Processing Complete!
echo  Results saved to: %TARGET%/dataset_schema.json
echo  Summary plot: %TARGET%/property_coverage_summary.png
echo ======================================================
pause
