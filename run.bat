@echo off
setlocal enabledelayedexpansion
echo ======================================================
echo  TMMStructures - Script Runner
echo ======================================================
echo.

REM ========================================================
REM  修改下面这一行来运行不同的 Python 脚本
REM ========================================================
set SCRIPT=compact_json.py Output/dataset/batch_1/dataset_schema.json

echo  Running: python %SCRIPT%
echo.
python %SCRIPT%
echo.
echo ======================================================
echo  Script finished.
echo ======================================================
pause
