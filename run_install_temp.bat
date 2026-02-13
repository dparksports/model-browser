@echo off
cd /d "C:\Users\k2\mymeetings"
echo ==========================================================
echo   Starting Meeting Detector Installation (Admin Mode)
echo ==========================================================
echo.
echo   Python Path: "C:\Users\k2\venvs\meetings\Scripts\python.exe"
echo.

"C:\Users\k2\venvs\meetings\Scripts\python.exe" install_libraries.py --yes
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed or CUDA install was cancelled.
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Installation complete.
echo.
pause
del "%~f0"
