#!/bin/bash
# Sumobot Simulator Runner for macOS/Linux
#
# Usage: ./run_simulator.sh "/path/to/Sumobot" 0 100 20 [timeScale] [pacing flags...]
#   $1 = exe path (or .app path on macOS)
#   $2 = configStart OR configIndex (if $3 is empty)
#   $3 = configEnd (optional, if empty then $2 is treated as configIndex)
#   $4 = batch size (required when using range mode)
#   $5 = timeScale (optional)
#   $6+ = optional Pacing Simulation overrides, as --key=value flags (see below)

# Function to display usage
usage() {
    cat << EOF
Usage:
  Range Mode:  $0 <sumobot_path> <config_start> <config_end> <batch_size> [time_scale] [pacing flags...]
  Single Mode: $0 <sumobot_path> <config_index> "" "" [time_scale] [pacing flags...]

Examples:
  # Range mode - Process configs 0-99 with batch size 20
  ./run_simulator.sh "/Applications/Sumobot.app" 0 100 20

  # Range mode with 5x time scale
  ./run_simulator.sh "/Applications/Sumobot.app" 0 100 20 5.0

  # Single mode - Run only config 933
  ./run_simulator.sh "/Applications/Sumobot.app" 933 "" ""

  # Single mode with 10x time scale
  ./run_simulator.sh "/Applications/Sumobot.app" 933 "" "" 10.0

  # Range mode with a Pacing Simulation sweep
  ./run_simulator.sh "/Applications/Sumobot.app" 0 100 20 5.0 \\
      --pacingSimulation=true \\
      --simTargetsFolder="Pacing/Sim_Targets/60s" \\
      --simConstraintsFolder="Pacing/Sim_Constraints" \\
      --pacingMin=0 --pacingMax=0.474 \\
      --focusBotIDs="MCTS,NN" --includeFocusMatchups=false

Arguments:
  sumobot_path  : Path to Sumobot executable or .app bundle (macOS)
  config_start  : Starting configuration index (range mode)
  config_index  : Single configuration index (single mode)
  config_end    : Ending configuration index (range mode, empty for single mode)
  batch_size    : Number of configs to run per instance (range mode only)
  time_scale    : Optional time scale multiplier (e.g., 5.0 for 5x speed)

Pacing Simulation flags (all optional; omitted ones keep whatever is baked into
the build's scene - see BattleSimulator.ApplyPacingCommandLineOverrides()):
  --pacingSimulation=true|false     : Enable/disable Pacing Simulation matchup generation
  --simTargetsFolder=<path>         : Resources-relative folder of pacing TARGET curves
  --simConstraintsFolder=<path>     : Resources-relative folder of pacing CONSTRAINT sets
  --pacingSegmentDuration=<int>     : Pacing segment duration
  --pacingCollisionWindow=<int>     : Pacing collision window duration
  --pacingMin=<float>               : Minimum pacing value
  --pacingMax=<float>               : Maximum pacing value
  --focusBotIDs=<id1,id2,...>       : Comma-separated bot IDs to mark as Focus bots
  --includeFocusMatchups=true|false : Also sweep Focus bots against each other

The script will launch multiple instances of the simulator, each processing
a batch of configurations in parallel.
EOF
    exit 1
}

# Function to find executable in .app bundle
find_executable() {
    local path="$1"

    # If it's a .app bundle on macOS
    if [[ "$path" == *.app ]]; then
        # Try to find the executable inside
        local app_name=$(basename "$path" .app)
        local executable="$path/Contents/MacOS/$app_name"

        if [[ -f "$executable" ]]; then
            echo "$executable"
            return 0
        fi

        # Try to find any executable in MacOS folder
        local macos_dir="$path/Contents/MacOS"
        if [[ -d "$macos_dir" ]]; then
            local found=$(find "$macos_dir" -type f -perm +111 -print -quit 2>/dev/null)
            if [[ -n "$found" ]]; then
                echo "$found"
                return 0
            fi
        fi

        echo "Error: Could not find executable in .app bundle: $path" >&2
        exit 1
    fi

    echo "$path"
}

# Check arguments
if [[ $# -lt 2 ]]; then
    usage
fi

UNITY_PATH="$1"
TIME_SCALE=""
SINGLE_MODE=false

# Detect mode: if $3 is empty, treat as single config mode
if [[ -z "$3" ]]; then
    # Single config mode
    SINGLE_MODE=true
    CONFIG_INDEX="$2"
    # Time scale is 5th parameter in single mode
    TIME_SCALE="$5"
else
    # Range mode
    if [[ $# -lt 4 ]]; then
        usage
    fi
    CONFIG_START="$2"
    CONFIG_END="$3"
    BATCH="$4"
    TIME_SCALE="$5"
fi

# Find actual executable (handle .app bundles)
UNITY_PATH=$(find_executable "$UNITY_PATH")

# Validate executable exists
if [[ ! -f "$UNITY_PATH" ]]; then
    echo "Error: Executable not found at '$UNITY_PATH'"
    exit 1
fi

# Check if executable is actually executable
if [[ ! -x "$UNITY_PATH" ]]; then
    echo "Warning: '$UNITY_PATH' is not executable"
    echo "Try running: chmod +x '$UNITY_PATH'"
    exit 1
fi

# Validate numeric arguments based on mode
if [[ "$SINGLE_MODE" = true ]]; then
    if ! [[ "$CONFIG_INDEX" =~ ^[0-9]+$ ]]; then
        echo "Error: config_index must be a number"
        exit 1
    fi
else
    if ! [[ "$CONFIG_START" =~ ^[0-9]+$ ]]; then
        echo "Error: config_start must be a number"
        exit 1
    fi

    if ! [[ "$CONFIG_END" =~ ^[0-9]+$ ]]; then
        echo "Error: config_end must be a number"
        exit 1
    fi

    if ! [[ "$BATCH" =~ ^[0-9]+$ ]]; then
        echo "Error: batch_size must be a number"
        exit 1
    fi

    if [[ $CONFIG_START -ge $CONFIG_END ]]; then
        echo "Error: config_start must be less than config_end"
        exit 1
    fi

    if [[ $BATCH -le 0 ]]; then
        echo "Error: batch_size must be positive"
        exit 1
    fi
fi

# Validate time scale if provided
if [[ -n "$TIME_SCALE" ]]; then
    if ! [[ "$TIME_SCALE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "Error: time_scale must be a positive number"
        exit 1
    fi
fi

# Parse optional Pacing Simulation flags ($6 onward) into PACING_ARGS, forwarded verbatim to
# the executable (BattleSimulator.ApplyPacingCommandLineOverrides() reads these on launch).
PACING_ARGS=()
PACING_SUMMARY=""
if [[ $# -ge 6 ]]; then
    for arg in "${@:6}"; do
        case "$arg" in
            --pacingSimulation=*|--includeFocusMatchups=*)
                val="${arg#*=}"
                if [[ "$val" != "true" && "$val" != "false" ]]; then
                    echo "Error: ${arg%%=*} must be true or false"
                    exit 1
                fi
                ;;
            --pacingSegmentDuration=*|--pacingCollisionWindow=*)
                val="${arg#*=}"
                if ! [[ "$val" =~ ^[0-9]+$ ]]; then
                    echo "Error: ${arg%%=*} must be an integer"
                    exit 1
                fi
                ;;
            --pacingMin=*|--pacingMax=*)
                val="${arg#*=}"
                if ! [[ "$val" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                    echo "Error: ${arg%%=*} must be a positive number"
                    exit 1
                fi
                ;;
            --simTargetsFolder=*|--simConstraintsFolder=*|--focusBotIDs=*)
                ;;
            *)
                echo "Error: Unknown argument '$arg'"
                usage
                ;;
        esac
        PACING_ARGS+=("$arg")
        PACING_SUMMARY="$PACING_SUMMARY $arg"
    done
fi

COMMON_ARGS="-batchmode"

# Build time scale argument if provided
TIME_SCALE_ARG=""
if [[ -n "$TIME_SCALE" ]]; then
    TIME_SCALE_ARG="--configTimeScale=$TIME_SCALE"
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Sumobot Simulator Runner"
echo "========================================"
echo "Simulator: $UNITY_PATH"

if [[ "$SINGLE_MODE" = true ]]; then
    # Single config mode
    echo "Mode: Single Configuration"
    echo "Config index: $CONFIG_INDEX"
    if [[ -n "$TIME_SCALE" ]]; then
        echo "Time scale: ${TIME_SCALE}x"
    fi
    echo "Platform: $(uname -s)"
    echo "Log directory: $SCRIPT_DIR"
    if [[ -n "$PACING_SUMMARY" ]]; then
        echo "Pacing overrides:$PACING_SUMMARY"
    fi
    echo ""

    echo "Launching config $CONFIG_INDEX (log: log_config_${CONFIG_INDEX}.txt)"

    # Launch in background
    "$UNITY_PATH" $COMMON_ARGS --configIndex=$CONFIG_INDEX $TIME_SCALE_ARG "${PACING_ARGS[@]}" --batchLogFile="log_config_${CONFIG_INDEX}.txt" &

    echo ""
    echo "========================================"
    echo "Simulation launched successfully!"
    echo "========================================"
    echo ""
    echo "Process is running in the background."
    echo "Check log file: log_config_${CONFIG_INDEX}.txt"
else
    # Range mode
    echo "Mode: Range"
    echo "Config range: $CONFIG_START to $CONFIG_END"
    echo "Batch size: $BATCH"
    if [[ -n "$TIME_SCALE" ]]; then
        echo "Time scale: ${TIME_SCALE}x"
    fi
    echo "Platform: $(uname -s)"
    echo "Log directory: $SCRIPT_DIR"
    if [[ -n "$PACING_SUMMARY" ]]; then
        echo "Pacing overrides:$PACING_SUMMARY"
    fi
    echo ""

    current=$CONFIG_START
    batch_count=0

    while [[ $current -lt $CONFIG_END ]]; do
        next=$((current + BATCH))
        if [[ $next -gt $CONFIG_END ]]; then
            next=$CONFIG_END
        fi

        batch_count=$((batch_count + 1))

        echo "[Batch $batch_count] Launching configs $current to $next (log: log_${current}-${next}.txt)"

        # Launch in background
        "$UNITY_PATH" $COMMON_ARGS --configStart=$current --configEnd=$next $TIME_SCALE_ARG "${PACING_ARGS[@]}" --batchLogFile="log_${current}-${next}.txt" &

        current=$next
    done

    echo ""
    echo "========================================"
    echo "All simulations launched successfully!"
    echo "Total batches: $batch_count"
    echo "========================================"
    echo ""
    echo "Processes are running in the background."
    echo "Check log files (log_*.txt) for progress."
fi
