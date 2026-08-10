@echo off
REM Sumobot Simulator Runner for Windows
REM
REM Usage:
REM   Range Mode:  run_simulator.bat "C:\path\to\Sumobot.exe" 0 100 20 [timeScale] [pacing flags...]
REM   Single Mode: run_simulator.bat "C:\path\to\Sumobot.exe" 993 single single [timeScale] [pacing flags...]
REM
REM Pacing Simulation flags (all optional; omitted ones keep whatever is baked into the
REM build's scene - see BattleSimulator.ApplyPacingCommandLineOverrides()):
REM   --pacingSimulation=true|false     : Enable/disable Pacing Simulation matchup generation
REM   --simTargetsFolder=<path>         : Resources-relative folder of pacing TARGET curves
REM   --simConstraintsFolder=<path>     : Resources-relative folder of pacing CONSTRAINT sets
REM   --pacingSegmentDuration=<int>     : Pacing segment duration
REM   --pacingCollisionWindow=<int>     : Pacing collision window duration
REM   --pacingMin=<float>               : Minimum pacing value
REM   --pacingMax=<float>               : Maximum pacing value
REM   --focusBotIDs=<id1,id2,...>       : Comma-separated bot IDs to mark as Focus bots
REM   --includeFocusMatchups=true|false : Also sweep Focus bots against each other
REM
REM Example:
REM   run_simulator.bat "C:\Sumobot\Sumobot.exe" 0 100 20 5.0 --pacingSimulation=true --simTargetsFolder=Pacing/Sim_Targets/60s --pacingMin=0 --pacingMax=0.474 --focusBotIDs=MCTS,NN

setlocal enabledelayedexpansion

set "UNITY_PATH=%~1"
set "SINGLE_MODE=false"

REM Detect mode: if %3 is "single", treat as single config mode
echo [DEBUG] Param 3: "%~3"
if /I "%~3"=="single" (
    echo [DEBUG] Entering SINGLE mode
    set "SINGLE_MODE=true"
    set "CONFIG_INDEX=%~2"
    set "TIME_SCALE=%~5"
) else (
    echo [DEBUG] Entering RANGE mode
    set "CONFIG_START=%~2"
    set "CONFIG_END=%~3"
    set "BATCH=%~4"
    set "TIME_SCALE=%~5"
)

REM Build arguments
set COMMON_ARGS=-batchmode -nographics

set "TIME_SCALE_ARG="
if not "!TIME_SCALE!"=="" (
    set "TIME_SCALE_ARG=--configTimeScale=!TIME_SCALE!"
)

REM Parse optional Pacing Simulation flags (args after position 5), forwarded verbatim to the
REM executable (BattleSimulator.ApplyPacingCommandLineOverrides() reads these on launch).
set "PACING_ARGS="
set "PACING_SUMMARY="
shift
shift
shift
shift
shift

:pacing_loop
if "%~1"=="" goto pacing_done
set "PARG=%~1"
set "PKEY="
for /f "tokens=1* delims==" %%A in ("!PARG!") do set "PKEY=%%A"

set "PACING_KNOWN=0"
for %%K in (--pacingSimulation --simTargetsFolder --simConstraintsFolder --pacingSegmentDuration --pacingCollisionWindow --pacingMin --pacingMax --focusBotIDs --includeFocusMatchups) do (
    if /I "!PKEY!"=="%%K" set "PACING_KNOWN=1"
)
if "!PACING_KNOWN!"=="0" (
    echo Error: Unknown argument "!PARG!"
    exit /b 1
)

set "PACING_ARGS=!PACING_ARGS! !PARG!"
set "PACING_SUMMARY=!PACING_SUMMARY! !PARG!"
shift
goto pacing_loop

:pacing_done

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo ========================================
echo Sumobot Simulator Runner v0.0.1
echo ========================================
echo Simulator: !UNITY_PATH!
echo.

echo [DEBUG] SINGLE_MODE = "!SINGLE_MODE!"

if /I "!SINGLE_MODE!"=="true" (
    echo [DEBUG] Executing SINGLE mode block
    REM Single config mode
    echo Mode: Single Configuration
    echo Config index: !CONFIG_INDEX!
    if not "!TIME_SCALE!"=="" (
        echo Time scale: !TIME_SCALE!x
    )
    echo Log directory: !SCRIPT_DIR!
    if not "!PACING_SUMMARY!"=="" (
        echo Pacing overrides:!PACING_SUMMARY!
    )
    echo.

    echo Launching config !CONFIG_INDEX! (log: log_config_!CONFIG_INDEX!.txt)
    start "SumobotSim" "!UNITY_PATH!" !COMMON_ARGS! --configIndex=!CONFIG_INDEX! !TIME_SCALE_ARG! !PACING_ARGS! --batchLogFile="log_config_!CONFIG_INDEX!.txt"

    echo.
    echo Simulation launched successfully!
    echo Check log file: log_config_!CONFIG_INDEX!.txt

    goto end
)

REM If we get here, we're in RANGE mode
echo [DEBUG] Executing RANGE mode block
echo Mode: Range
echo Config range: !CONFIG_START! to !CONFIG_END!
echo Batch size: !BATCH!
if not "!TIME_SCALE!"=="" (
    echo Time scale: !TIME_SCALE!x
)
echo Log directory: !SCRIPT_DIR!
if not "!PACING_SUMMARY!"=="" (
    echo Pacing overrides:!PACING_SUMMARY!
)
echo.

set current=!CONFIG_START!
set batch_count=0

:loop
if !current! GEQ !CONFIG_END! goto done

set /a next=!current! + !BATCH!
if !next! GTR !CONFIG_END! set next=!CONFIG_END!

set /a batch_count+=1

echo [Batch !batch_count!] Launching configs !current! to !next! (log: log_!current!-!next!.txt)
start "Sumobot" "!UNITY_PATH!" !COMMON_ARGS! --configStart=!current! --configEnd=!next! !TIME_SCALE_ARG! !PACING_ARGS! --batchLogFile="log_!current!-!next!.txt"

set current=!next!
goto loop

:done
echo.
echo All simulations launched successfully!
echo Total batches: !batch_count!

:end
echo.
echo ========================================
pause
