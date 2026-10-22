param(
    [string]$Executable,
    [string]$EnvironmentName = "",
    [switch]$AutoOnline
)

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $projectRoot "Builds\Windows\Sumobot.exe"
}

$resolvedExecutable = [System.IO.Path]::GetFullPath($Executable)
if (-not (Test-Path -LiteralPath $resolvedExecutable)) {
    throw "Sumobot build not found at '$resolvedExecutable'. Build it from Sumobot > Multiplayer > Build Windows Test Client first."
}

$buildDirectory = Split-Path -Parent $resolvedExecutable
$matchPool = "local-" + [Guid]::NewGuid().ToString("N")
$environmentArgument = if ([string]::IsNullOrWhiteSpace($EnvironmentName)) {
    @()
} else {
    @("-ugs-environment=$EnvironmentName")
}
$playerOneAutoArgument = if ($AutoOnline) { @("-auto-create-room") } else { @() }
$playerTwoAutoArgument = if ($AutoOnline) { @("-auto-join-room") } else { @() }

$commonArguments = @(
    "-match-pool=$matchPool",
    "-screen-fullscreen", "0",
    "-screen-width", "960",
    "-screen-height", "540"
)

$playerOneArguments = @("-ugs-profile=sumobot_p1", "-logFile", (Join-Path $buildDirectory "player1.log")) + $environmentArgument + $playerOneAutoArgument + $commonArguments
$playerTwoArguments = @("-ugs-profile=sumobot_p2", "-logFile", (Join-Path $buildDirectory "player2.log")) + $environmentArgument + $playerTwoAutoArgument + $commonArguments

Start-Process -FilePath $resolvedExecutable -WorkingDirectory $buildDirectory -ArgumentList $playerOneArguments
if ($AutoOnline) {
    Write-Host "Waiting for player one to create the online session..."
    Start-Sleep -Seconds 8
}
Start-Process -FilePath $resolvedExecutable -WorkingDirectory $buildDirectory -ArgumentList $playerTwoArguments

Write-Host "Started two Sumobot clients with UGS profiles sumobot_p1 and sumobot_p2 in matchmaking pool $matchPool."
