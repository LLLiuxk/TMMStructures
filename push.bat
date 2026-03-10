@echo off
echo.
echo ====================================
echo Pushing changes to GitHub...
echo ====================================
echo.

:: Automatically Stage all changes
git add .

:: Commit with a generic timestamp message
:: "Auto-sync from computer: YYYY-MM-DD HH:MM:SS"
git commit -m "Auto-sync: %date% %time%"

:: Push to remote
git push origin main

echo.
echo ====================================
echo Push complete!
echo ====================================
pause
