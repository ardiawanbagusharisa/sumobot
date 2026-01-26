@echo off
REM Sumobot Simulator Runner for Windows
REM
REM Usage: run_simulator.bat "C:\path\to\Sumobot.exe" 0 100 20
REM   %1 = exe path
REM   %2 = configStart
REM   %3 = configEnd
REM   %4 = batch size

if "%~4"=="" (
    echo Usage: run_simulator.bat ^<exe_path^> ^<config_start^> ^<config_end^> ^<batch_size^>
    echo.
    echo Example:
    echo   run_simulator.bat "C:\path\to\Sumobot.exe" 0 100 20
    echo.
    echo Arguments:
    echo   exe_path      : Path to Sumobot executable
    echo   config_start  : Starting configuration index
    echo   config_end    : Ending configuration index
    echo   batch_size    : Number of configs to run per instance
    exit /b 1
)

set "UNITY_PATH=%~1"
set CONFIG_START=%2
set CONFIG_END=%3
set BATCH=%4

REM Validate executable exists
if not exist "%UNITY_PATH%" (
    echo Error: Executable not found at '%UNITY_PATH%'
    exit /b 1
)

REM Validate numeric arguments
set /a test=%CONFIG_START% 2>nul
if errorlevel 1 (
    echo Error: config_start must be a number
    exit /b 1
)

set /a test=%CONFIG_END% 2>nul
if errorlevel 1 (
    echo Error: config_end must be a number
    exit /b 1
)

set /a test=%BATCH% 2>nul
if errorlevel 1 (
    echo Error: batch_size must be a number
    exit /b 1
)

if %CONFIG_START% GEQ %CONFIG_END% (
    echo Error: config_start must be less than config_end
    exit /b 1
)

if %BATCH% LEQ 0 (
    echo Error: batch_size must be positive
    exit /b 1
)

set COMMON_ARGS=-batchmode -nographics

REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo ========================================
echo Sumobot Simulator Runner
echo ========================================
echo Simulator: %UNITY_PATH%
echo Config range: %CONFIG_START% to %CONFIG_END%
echo Batch size: %BATCH%
echo Log directory: %SCRIPT_DIR%
echo.

setlocal enabledelayedexpansion
set current=%CONFIG_START%
set batch_count=0

:loop
if %current% GEQ %CONFIG_END% goto done

set /a next=%current% + %BATCH%
if !next! GTR %CONFIG_END% set next=%CONFIG_END%

set "LOGFILE=%SCRIPT_DIR%\log_!current!-!next!.txt"
set /a batch_count+=1

echo [Batch !batch_count!] Launching configs !current! to !next! (log: log_!current!-!next!.txt)
start "Sumobot Simulator" "%UNITY_PATH%" %COMMON_ARGS% --configStart=%current% --configEnd=!next! -logFile "!LOGFILE!"

set current=!next!
goto loop

:done
echo.
echo ========================================
echo All simulations launched successfully!
echo Total batches: %batch_count%
echo ========================================
pause
