@echo off
setlocal enabledelayedexpansion
echo ======================================================
echo  TMMStructures - Batch 2 Generation (Part 2/2)
echo ======================================================
echo.
echo Target: 25,000 samples (IDs 25000 to 49999)
echo Output: Output/dataset/batch_2/
echo Nodes:  3 to 4 per edge
echo.

python generate_dataset.py --config dataset_config.json --offset 25000 --num-samples 25000

echo.
echo ======================================================
echo  Part 2 Generation Complete!
echo ======================================================
pause
