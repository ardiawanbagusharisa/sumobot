# Sumobot Simulator Runner for Windows (PowerShell)
#
# Usage:
#   Range Mode:  .\run_simulator.ps1 "C:\path\to\Sumobot.exe" 0 100 20 [timeScale] [-PacingSimulation ...]
#   Single Mode: .\run_simulator.ps1 "C:\path\to\Sumobot.exe" 993 single single [timeScale] [-PacingSimulation ...]
#
# Pacing Simulation overrides (all optional; omitted ones keep whatever is baked into the
# build's scene - see BattleSimulator.ApplyPacingCommandLineOverrides()):
#   -PacingSimulation <bool>        : Enable/disable Pacing Simulation matchup generation
#   -SimTargetsFolder <path>        : Resources-relative folder of pacing TARGET curves
#   -SimConstraintsFolder <path>    : Resources-relative folder of pacing CONSTRAINT sets
#   -PacingSegmentDuration <int>    : Pacing segment duration
#   -PacingCollisionWindow <int>    : Pacing collision window duration
#   -PacingMin <float>              : Minimum pacing value
#   -PacingMax <float>              : Maximum pacing value
#   -FocusBotIDs <id1,id2,...>      : Comma-separated bot IDs to mark as Focus bots
#   -IncludeFocusMatchups <bool>    : Also sweep Focus bots against each other
#
# Example:
#   .\run_simulator.ps1 "C:\Sumobot\Sumobot.exe" 0 100 20 5.0 -PacingSimulation $true -SimTargetsFolder "Pacing/Sim_Targets/60s" -PacingMin 0 -PacingMax 0.474 -FocusBotIDs "MCTS,NN"

param(
    [Parameter(Mandatory=$true)]
    [string]$UnityPath,

    [Parameter(Mandatory=$true)]
    [string]$Param2,

    [Parameter(Mandatory=$false)]
    [string]$Param3 = "",

    [Parameter(Mandatory=$false)]
    [string]$Param4 = "",

    [Parameter(Mandatory=$false)]
    [string]$TimeScale = "",

    [Parameter(Mandatory=$false)]
    [Nullable[bool]]$PacingSimulation = $null,

    [Parameter(Mandatory=$false)]
    [string]$SimTargetsFolder = "",

    [Parameter(Mandatory=$false)]
    [string]$SimConstraintsFolder = "",

    [Parameter(Mandatory=$false)]
    [Nullable[int]]$PacingSegmentDuration = $null,

    [Parameter(Mandatory=$false)]
    [Nullable[int]]$PacingCollisionWindow = $null,

    [Parameter(Mandatory=$false)]
    [Nullable[double]]$PacingMin = $null,

    [Parameter(Mandatory=$false)]
    [Nullable[double]]$PacingMax = $null,

    [Parameter(Mandatory=$false)]
    [string]$FocusBotIDs = "",

    [Parameter(Mandatory=$false)]
    [Nullable[bool]]$IncludeFocusMatchups = $null
)

# Detect mode
$SingleMode = $false
if ($Param3 -eq "single") {
    $SingleMode = $true
    $ConfigIndex = $Param2
} else {
    $ConfigStart = [int]$Param2
    $ConfigEnd = [int]$Param3
    $Batch = [int]$Param4
}

# Build common args
$CommonArgs = "-batchmode -logFile run_simulator.log"

# Add time scale if provided
$TimeScaleArg = ""
if ($TimeScale -ne "") {
    $TimeScaleArg = "--configTimeScale=$TimeScale"
}

# Build Pacing Simulation overrides, forwarded verbatim to the executable
# (BattleSimulator.ApplyPacingCommandLineOverrides() reads these on launch).
$PacingArgsList = @()
if ($PacingSimulation -ne $null) { $PacingArgsList += "--pacingSimulation=$($PacingSimulation.ToString().ToLower())" }
if ($SimTargetsFolder -ne "") { $PacingArgsList += "--simTargetsFolder=`"$SimTargetsFolder`"" }
if ($SimConstraintsFolder -ne "") { $PacingArgsList += "--simConstraintsFolder=`"$SimConstraintsFolder`"" }
if ($PacingSegmentDuration -ne $null) { $PacingArgsList += "--pacingSegmentDuration=$PacingSegmentDuration" }
if ($PacingCollisionWindow -ne $null) { $PacingArgsList += "--pacingCollisionWindow=$PacingCollisionWindow" }
if ($PacingMin -ne $null) { $PacingArgsList += "--pacingMin=$PacingMin" }
if ($PacingMax -ne $null) { $PacingArgsList += "--pacingMax=$PacingMax" }
if ($FocusBotIDs -ne "") { $PacingArgsList += "--focusBotIDs=`"$FocusBotIDs`"" }
if ($IncludeFocusMatchups -ne $null) { $PacingArgsList += "--includeFocusMatchups=$($IncludeFocusMatchups.ToString().ToLower())" }
$PacingArgs = ($PacingArgsList -join " ")

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "========================================"
Write-Host "Sumobot Simulator Runner (PowerShell)"
Write-Host "========================================"
Write-Host "Simulator: $UnityPath"
Write-Host ""

if ($SingleMode) {
    # Single config mode
    Write-Host "Mode: Single Configuration"
    Write-Host "Config index: $ConfigIndex"
    if ($TimeScale -ne "") {
        Write-Host "Time scale: ${TimeScale}x"
    }
    Write-Host "Log directory: $ScriptDir"
    if ($PacingArgs -ne "") {
        Write-Host "Pacing overrides: $PacingArgs"
    }
    Write-Host ""

    Write-Host "Launching config $ConfigIndex (log: log_config_$ConfigIndex.txt)"

    $LogFile = "log_config_$ConfigIndex.txt"
    Start-Process -FilePath $UnityPath -ArgumentList "$CommonArgs --configIndex=$ConfigIndex $TimeScaleArg $PacingArgs --batchLogFile=`"$LogFile`"" -WindowStyle Hidden

    Write-Host ""
    Write-Host "Simulation launched successfully!"
    Write-Host "Check log file: $LogFile"
} else {
    # Range mode
    Write-Host "Mode: Range"
    Write-Host "Config range: $ConfigStart to $ConfigEnd"
    Write-Host "Batch size: $Batch"
    if ($TimeScale -ne "") {
        Write-Host "Time scale: ${TimeScale}x"
    }
    Write-Host "Log directory: $ScriptDir"
    if ($PacingArgs -ne "") {
        Write-Host "Pacing overrides: $PacingArgs"
    }
    Write-Host ""

    $current = $ConfigStart
    $batchCount = 0

    while ($current -lt $ConfigEnd) {
        $next = $current + $Batch
        if ($next -gt $ConfigEnd) {
            $next = $ConfigEnd
        }

        $batchCount++
        $LogFile = "log_$current-$next.txt"

        Write-Host "[Batch $batchCount] Launching configs $current to $next (log: $LogFile)"

        Start-Process -FilePath $UnityPath -ArgumentList "$CommonArgs --configStart=$current --configEnd=$next $TimeScaleArg $PacingArgs --batchLogFile=`"$LogFile`"" -WindowStyle Hidden

        $current = $next
    }

    Write-Host ""
    Write-Host "All simulations launched successfully!"
    Write-Host "Total batches: $batchCount"
}

Write-Host ""
Write-Host "========================================"
Read-Host "Press Enter to continue"
