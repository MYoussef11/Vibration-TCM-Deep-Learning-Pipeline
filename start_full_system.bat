@echo off
REM Vibration TCM System Launcher
REM Starts all components in separate windows

echo.
echo ======================================================================
echo Vibration TCM System Launcher - Full System
echo ======================================================================
echo.
echo Starting all 5 components...
echo Each will open in a new window
echo.
echo Close this window or press Ctrl+C to stop all components
echo.

REM Start Dashboard
echo [1/5] Starting Dashboard...
start "Dashboard" cmd /k "streamlit run dashboard.py"
timeout /t 2 /nobreak >nul

REM Start DL Inference
echo [2/5] Starting DL Inference Engine...
start "DL Inference" cmd /k "python scripts\stream_inference.py"
timeout /t 3 /nobreak >nul

REM Start ML Inference  
echo [3/5] Starting ML Inference Engine...
start "ML Inference" cmd /k "python scripts\stream_ml_inference.py"
timeout /t 1 /nobreak >nul

REM Start Sensor Gateway (COM8 - change if needed)
echo [4/5] Starting Sensor Gateway...
start "Sensor Gateway" cmd /k "python scripts\stream_gateway.py --port COM8"
timeout /t 1 /nobreak >nul

REM Start Data Logger
echo [5/5] Starting Data Logger...
start "Data Logger" cmd /k "python scripts\data_logger.py"
timeout /t 1 /nobreak >nul

echo.
echo ======================================================================
echo All components started!
echo ======================================================================
echo.
echo Check the separate windows for each component's output
echo If any window closes immediately, there's an error in that component
echo.
echo To stop: Close each window individually or close this window
echo.
pause
