@echo off
echo ==========================================================
echo  TMMStructures - Dataset Generation and Analysis Pipeline
echo ==========================================================

echo [1/2] Generating Microstructures and Computing Properties...
python generate_dataset.py

echo [2/2] Generating Property Coverage and HS Bound Analysis...
python plot_property_coverage.py

echo.
echo Pipeline Complete. Results are in Output\dataset\
pause
