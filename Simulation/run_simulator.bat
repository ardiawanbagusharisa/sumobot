@echo off
REM Sumobot Simulator Runner for Windows
REM
REM Usage:
REM   Range Mode:  run_simulator.bat "C:\path\to\Sumobot.exe" 0 100 20 [timeScale]
REM   Single Mode: run_simulator.bat "C:\path\to\Sumobot.exe" 993 single single [timeScale]

setlocal enabledelayedexpansion

set "UNITY_PATH=%~1"
set "SINGLE_MODE=false"

REM Detect mode: if %3 is "single", treat as single config mode
if /I "%~3"=="single" (
    set "SINGLE_MODE=true"
    set "CONFIG_INDEX=%~2"
    set "TIME_SCALE=%~5"
) else (
    set "CONFIG_START=%~2"
    set "CONFIG_END=%~3"
    set "BATCH=%~4"
    set "TIME_SCALE=%~5"
)

REM Build arguments
set COMMON_ARGS=-batchmode -nographics

set "TIME_SCALE_ARG="
if not "%TIME_SCALE%"=="" (
    set "TIME_SCALE_ARG=--configTimeScale=%TIME_SCALE%"
)

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo ========================================
echo Sumobot Simulator Runner
echo ========================================
echo Simulator: %UNITY_PATH%

if "!SINGLE_MODE!"=="true" (
    REM Single config mode
    echo Mode: Single Configuration
    echo Config index: !CONFIG_INDEX!
    if not "!TIME_SCALE!"=="" echo Time scale: !TIME_SCALE!x
    echo Log directory: !SCRIPT_DIR!
    echo.

    set "LOGFILE=!SCRIPT_DIR!\log_config_!CONFIG_INDEX!.txt"

    echo Launching config !CONFIG_INDEX! (log: log_config_!CONFIG_INDEX!.txt)
    start "Sumobot" "!UNITY_PATH!" !COMMON_ARGS! --configIndex=!CONFIG_INDEX! !TIME_SCALE_ARG! -logFile "!LOGFILE!"

    echo.
    echo Simulation launched successfully!
    echo Check log file: log_config_!CONFIG_INDEX!.txt
) else (
    REM Range mode
    echo Mode: Range
    echo Config range: !CONFIG_START! to !CONFIG_END!
    echo Batch size: !BATCH!
    if not "!TIME_SCALE!"=="" echo Time scale: !TIME_SCALE!x
    echo Log directory: !SCRIPT_DIR!
    echo.

    set current=!CONFIG_START!
    set batch_count=0

    :loop
    if !current! GEQ !CONFIG_END! goto done

    set /a next=!current! + !BATCH!
    if !next! GTR !CONFIG_END! set next=!CONFIG_END!

    set "LOGFILE=%SCRIPT_DIR%\log_!current!-!next!.txt"
    set /a batch_count+=1

    echo [Batch !batch_count!] Launching configs !current! to !next! (log: log_!current!-!next!.txt)
    start "Sumobot" "%UNITY_PATH%" %COMMON_ARGS% --configStart=!current! --configEnd=!next! %TIME_SCALE_ARG% -logFile "!LOGFILE!"

    set current=!next!
    goto loop

    :done
    echo.
    echo All simulations launched successfully!
    echo Total batches: %batch_count%
)

echo ========================================
pause
