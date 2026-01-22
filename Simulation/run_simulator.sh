#!/bin/bash
# Sumobot Simulator Runner for macOS/Linux
#
# Usage: ./run_simulator.sh "/path/to/Sumobot" 0 100 20
#   $1 = exe path (or .app path on macOS)
#   $2 = configStart
#   $3 = configEnd
#   $4 = batch size

# Function to display usage
usage() {
    cat << EOF
Usage: $0 <sumobot_path> <config_start> <config_end> <batch_size>

Example:
  # macOS
  ./run_simulator.sh "/Applications/Sumobot.app" 0 100 20

  # Linux
  ./run_simulator.sh "/path/to/Sumobot.x86_64" 0 100 20

Arguments:
  sumobot_path  : Path to Sumobot executable or .app bundle (macOS)
  config_start  : Starting configuration index
  config_end    : Ending configuration index
  batch_size    : Number of configs to run per instance

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
if [[ $# -ne 4 ]]; then
    usage
fi

UNITY_PATH="$1"
CONFIG_START="$2"
CONFIG_END="$3"
BATCH="$4"

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

# Validate numeric arguments
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

COMMON_ARGS="-batchmode -nographics"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Sumobot Simulator Runner"
echo "========================================"
echo "Simulator: $UNITY_PATH"
echo "Config range: $CONFIG_START to $CONFIG_END"
echo "Batch size: $BATCH"
echo "Platform: $(uname -s)"
echo "Log directory: $SCRIPT_DIR"
echo ""

current=$CONFIG_START
batch_count=0

while [[ $current -lt $CONFIG_END ]]; do
    next=$((current + BATCH))
    if [[ $next -gt $CONFIG_END ]]; then
        next=$CONFIG_END
    fi

    LOGFILE="$SCRIPT_DIR/log_${current}-${next}.txt"
    batch_count=$((batch_count + 1))

    echo "[Batch $batch_count] Launching configs $current to $next (log: log_${current}-${next}.txt)"

    # Launch in background
    "$UNITY_PATH" $COMMON_ARGS --configStart=$current --configEnd=$next -logFile "$LOGFILE" &

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
