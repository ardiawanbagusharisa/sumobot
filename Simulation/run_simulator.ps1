# Sumobot Simulator Runner for Windows (PowerShell)
#
# Usage:
#   Range Mode:  .\run_simulator.ps1 "C:\path\to\Sumobot.exe" 0 100 20 [timeScale]
#   Single Mode: .\run_simulator.ps1 "C:\path\to\Sumobot.exe" 993 single single [timeScale]

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
    [string]$TimeScale = ""
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
$CommonArgs = "-batchmode -nographics"

# Add time scale if provided
$TimeScaleArg = ""
if ($TimeScale -ne "") {
    $TimeScaleArg = "--configTimeScale=$TimeScale"
}

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
    Write-Host ""

    Write-Host "Launching config $ConfigIndex (log: log_config_$ConfigIndex.txt)"

    $LogFile = "log_config_$ConfigIndex.txt"
    Start-Process -FilePath $UnityPath -ArgumentList "$CommonArgs --configIndex=$ConfigIndex $TimeScaleArg --batchLogFile=`"$LogFile`"" -WindowStyle Hidden

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

        Start-Process -FilePath $UnityPath -ArgumentList "$CommonArgs --configStart=$current --configEnd=$next $TimeScaleArg --batchLogFile=`"$LogFile`"" -WindowStyle Hidden

        $current = $next
    }

    Write-Host ""
    Write-Host "All simulations launched successfully!"
    Write-Host "Total batches: $batchCount"
}

Write-Host ""
Write-Host "========================================"
Read-Host "Press Enter to continue"
