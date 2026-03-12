@echo off
cd /d "%~dp0"
echo ============================================================
echo  TMMStructures - Running Dataset Generator
echo ============================================================
python generate_dataset.py
echo.
echo Press any key to exit...
pause >nul
