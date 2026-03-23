@echo off
echo =========================================================================
echo Batch Recomputation of Homogenization and Radar Charts
echo =========================================================================
echo This script will recompute the C_eff and kappa_eff for an existing dataset
echo using the updated 4-fold mirroring logic in homogenize.py and regenerate 
echo the radar charts using logarithmic scaling.
echo.

set TARGET_DIR=Output\dataset\batch_1
if not "%~1"=="" set TARGET_DIR=%~1

echo Target Directory: %TARGET_DIR%
echo.
echo Starting multi-processing python script...
python recompute_dataset_mp.py "%TARGET_DIR%"

echo.
echo Recomputation Finished.
pause
