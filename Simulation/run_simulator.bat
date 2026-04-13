@echo off
REM Sumobot Simulator Runner for Windows
REM
REM Usage: run_simulator.bat "C:\path\to\Sumobot.exe" 0 100 20 [timeScale]
REM   %1 = exe path
REM   %2 = configStart OR configIndex (if %3 is empty)
REM   %3 = configEnd (optional, if empty then %2 is treated as configIndex)
REM   %4 = batch size (required when using range mode)
REM   %5 = timeScale (optional)

if "%~2"=="" (
    echo Usage:
    echo   Range Mode:  run_simulator.bat ^<exe_path^> ^<config_start^> ^<config_end^> ^<batch_size^> [time_scale]
    echo   Single Mode: run_simulator.bat ^<exe_path^> ^<config_index^> "" "" [time_scale]
    echo.
    echo Examples:
    echo   REM Range mode - Process configs 0-99 with batch size 20
    echo   run_simulator.bat "C:\path\to\Sumobot.exe" 0 100 20
    echo.
    echo   REM Range mode with 5x time scale
    echo   run_simulator.bat "C:\path\to\Sumobot.exe" 0 100 20 5.0
    echo.
    echo   REM Single mode - Run only config 933
    echo   run_simulator.bat "C:\path\to\Sumobot.exe" 933 "" ""
    echo.
    echo   REM Single mode with 10x time scale
    echo   run_simulator.bat "C:\path\to\Sumobot.exe" 933 "" "" 10.0
    echo.
    echo Arguments:
    echo   exe_path      : Path to Sumobot executable
    echo   config_start  : Starting configuration index (range mode)
    echo   config_index  : Single configuration index (single mode)
    echo   config_end    : Ending configuration index (range mode, empty for single mode)
    echo   batch_size    : Number of configs to run per instance (range mode only)
    echo   time_scale    : Optional time scale multiplier (e.g., 5.0 for 5x speed)
    exit /b 1
)

set "UNITY_PATH=%~1"
set "SINGLE_MODE=false"

REM Validate executable exists
if not exist "%UNITY_PATH%" (
    echo Error: Executable not found at '%UNITY_PATH%'
    exit /b 1
)

REM Detect mode: if %3 is empty, treat as single config mode
if "%~3"=="" (
    REM Single config mode
    set "SINGLE_MODE=true"
    set "CONFIG_INDEX=%~2"
    REM Time scale is 5th parameter in single mode
    set "TIME_SCALE=%~5"
) else (
    REM Range mode
    if "%~4"=="" (
        echo Error: batch_size is required for range mode
        exit /b 1
    )
    set "CONFIG_START=%~2"
    set "CONFIG_END=%~3"
    set "BATCH=%~4"
    set "TIME_SCALE=%~5"
)

REM Validate numeric arguments based on mode
if "%SINGLE_MODE%"=="true" (
    set /a test=%CONFIG_INDEX% 2>nul
    if errorlevel 1 (
        echo Error: config_index must be a number
        exit /b 1
    )
) else (
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
)

set COMMON_ARGS=-batchmode -nographics

REM Build time scale argument if provided
set "TIME_SCALE_ARG="
if not "%TIME_SCALE%"=="" (
    set "TIME_SCALE_ARG=--configTimeScale=%TIME_SCALE%"
)

REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo ========================================
echo Sumobot Simulator Runner
echo ========================================
echo Simulator: %UNITY_PATH%

if "%SINGLE_MODE%"=="true" (
    REM Single config mode
    echo Mode: Single Configuration
    echo Config index: %CONFIG_INDEX%
    if not "%TIME_SCALE%"=="" (
        echo Time scale: %TIME_SCALE%x
    )
    echo Log directory: %SCRIPT_DIR%
    echo.

    set "LOGFILE=%SCRIPT_DIR%\log_config_%CONFIG_INDEX%.txt"

    echo Launching config %CONFIG_INDEX% (log: log_config_%CONFIG_INDEX%.txt)

    REM Launch in background
    start "Sumobot Simulator" "%UNITY_PATH%" %COMMON_ARGS% --configIndex=%CONFIG_INDEX% %TIME_SCALE_ARG% -logFile "%LOGFILE%"

    echo.
    echo ========================================
    echo Simulation launched successfully!
    echo ========================================
    echo.
    echo Process is running in the background.
    echo Check log file: log_config_%CONFIG_INDEX%.txt
    pause
) else (
    REM Range mode
    echo Mode: Range
    echo Config range: %CONFIG_START% to %CONFIG_END%
    echo Batch size: %BATCH%
    if not "%TIME_SCALE%"=="" (
        echo Time scale: %TIME_SCALE%x
    )
    echo Log directory: %SCRIPT_DIR%
    echo.

    setlocal enabledelayedexpansion
    set current=%CONFIG_START%
    set batch_count=0

    :loop
    if !current! GEQ %CONFIG_END% goto done

    set /a next=!current! + %BATCH%
    if !next! GTR %CONFIG_END% set next=%CONFIG_END%

    set "LOGFILE=%SCRIPT_DIR%\log_!current!-!next!.txt"
    set /a batch_count+=1

    echo [Batch !batch_count!] Launching configs !current! to !next! (log: log_!current!-!next!.txt)

    REM Launch in background
    start "Sumobot Simulator" "%UNITY_PATH%" %COMMON_ARGS% --configStart=!current! --configEnd=!next! %TIME_SCALE_ARG% -logFile "!LOGFILE!"

    set current=!next!
    goto loop

    :done
    echo.
    echo ========================================
    echo All simulations launched successfully!
    echo Total batches: %batch_count%
    echo ========================================
    pause
)
