@echo off
REM Demo Mode - Dashboard + Models (no sensor needed for testing)

echo.
echo ======================================================================
echo Vibration TCM System - Demo Mode
echo ======================================================================
echo.
echo Starting 4 components (no sensor required)...
echo.

REM Start Dashboard
echo [1/4] Starting Dashboard...
start "Dashboard" cmd /k "streamlit run dashboard.py"
timeout /t 2 /nobreak >nul

REM Start DL Inference
echo [2/4] Starting DL Inference Engine...
start "DL Inference" cmd /k "python scripts\stream_inference.py"
timeout /t 3 /nobreak >nul

REM Start ML Inference  
echo [3/4] Starting ML Inference Engine...
start "ML Inference" cmd /k "python scripts\stream_ml_inference.py"
timeout /t 1 /nobreak >nul

REM Start Data Logger
echo [4/4] Starting Data Logger...
start "Data Logger" cmd /k "python scripts\data_logger.py"

echo.
echo ======================================================================
echo Demo mode started!
echo ======================================================================
echo.
echo Note: Without sensor data, models won't make predictions
echo Use for testing UI and system connectivity
echo.
pause
