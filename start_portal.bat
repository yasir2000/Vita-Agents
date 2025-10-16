@echo off
REM Vita Agents Healthcare Portal - Windows Startup Script
REM ========================================================

echo.
echo 🏥 Vita Agents Healthcare Portal
echo ================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if start_portal.py exists
if not exist "start_portal.py" (
    echo ❌ start_portal.py not found
    echo Please run this script from the Vita-Agents directory
    pause
    exit /b 1
)

echo ✅ Starting Vita Agents Healthcare Portal...
echo.

REM Start the portal with auto port detection
python start_portal.py --find-port

echo.
echo 🛑 Server stopped
pause