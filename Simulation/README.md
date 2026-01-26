# Sumobot Player Documentation

Clone Sumobot with `git clone https://github.com/ardiawanbagusharisa/sumobot.git`

# Simulation

## Simulation (Single Mode)

1. Open Sumobot on Unity
2. Open Battle.scene
3. Navigate to `BattleManager` object under `Managers`
4. In the Inspector, Look up to the Battle Simulator component and select Simulator Mode to `Simple`
4. Adjust parameters:
     `Total Simulations` - total simulation that will be accomplished,
     `Time Scale` - how many speed of time scale every battle played,
     `Swap AI Interval` - interval of bot swap between left and right among simulation count
     `Simulation On Start` - if enabled, the time scale will be applied earlier right after the play button pressed
5. (Optional) Enable Bot Left and Right in the Bot Manager component and choose bot type

## Simulation (Batch Mode)

1. Open Sumobot on Unity
2. Open Battle.scene
3. Navigate to `BattleManager` object under `Managers`
4. On the Inspector, Look up to the Battle Simulator component and select Simulator Mode to `Advanced`
5. Expand Advanced Mode Settings (Batch) and adjust the parameters

    - Timers - `15`, `30`, `45`, `60` | Match duration in seconds. Choose any combination.
    - ActionIntervals - `0.1`, `0.2`, `0.5` | Agent decision-making interval in seconds. Choose any combination.
    - Skills - `0`, `1` | Skill settings: `0` = Boost, `1` = Stone. Choose any combination.
    - RoundSystem - `1`, `3`, `5` | Best-of-N format: `1` = single match, `3` = best of 3, `5` = best of 5. Choose any combination.
    - Iteration - Any positive integer | Number of times to run each configuration (e.g., `50`, `100`)
    - Bot Selection - All 13 agents displayed on Battle Simualtor inspector under Bot Selection menu, you can choose any combination of the bot type or simulate all of them

    **Note:** You can use any subset of the available values. For example:
    - `"Timers": [30, 60]` (only 30s and 60s matches)
    - `"ActionIntervals": [0.2]` (only 0.2s interval)
    - `"Skills": [1]` (only with skills enabled).

6. Build and take a note the location of the Sumobot application (.exe / .app)

7. Run Simulation

    Choose the script for your platform (can be found on Sumobot/Simulation/...):

    **Windows**
    ```batch
    run_simulator.bat "C:\path\to\Sumobot.exe" 0 100 20
    ```

    **macOS**
    ```bash
    ./run_simulator.sh "/path/to/Sumobot.app" 0 100 20
    ```

    **Linux**
    ```bash
    ./run_simulator.sh "/path/to/Sumobot.x86_64" 0 100 20
    ```

    Change the /path/to with location of Sumobot executable that have just been built

    #### Parameter Details

    ##### 1. **sumobot_path**
    - **Windows**: Path to `.exe` file
      - Example: `"C:\Program Files\Sumobot\Sumobot.exe"`
      - Use quotes if path contains spaces
    - **macOS**: Path to `.app` bundle or executable inside
      - Example: `"/Applications/Sumobot.app/Contents/MacOS/Sumobot"`
      - Script automatically finds executable inside .app
      - Make sure file has execute permissions: `chmod +x /path/to/executable`
    - **Linux**: Path to executable
      - Example: `"/opt/Sumobot/Sumobot.x86_64"`
      - Make sure file has execute permissions: `chmod +x /path/to/executable`

    ##### 2. **config_start** (Starting Index)
    - First configuration to process
    - Usually `0` to start from the beginning
    - Must be less than `config_end`

    ##### 3. **config_end** (Ending Index)
    - Last configuration to process (exclusive)
    - Example: If you have 100 configs (0-99), use `100` as end
    - Total configs processed: `config_end - config_start`

    ##### 4. **batch_size** (Parallel Instances)
    - Number of configs each simulator instance processes
    - Controls how many parallel instances run
    - **Total instances** = `(config_end - config_start) / batch_size`

    **Example:**
    ```bash
    # Process configs 0-99 with batch size 20
    ./run_simulator.sh "/Applications/Sumobot.app" 0 100 20

    # This launches 5 parallel instances:
    #   Instance 1: configs 0-19
    #   Instance 2: configs 20-39
    #   Instance 3: configs 40-59
    #   Instance 4: configs 60-79
    #   Instance 5: configs 80-99
    ```

8. Additional Setup on Bot_LLM (LLM Agent)

    Since running Bot_LLM requires huge resource which can't be hosted on github, we need to use docker image that already hosted on docker hub.

    1. Install docker desktop ([Windows](https://docs.docker.com/desktop/setup/install/windows-install/) / [Mac](https://docs.docker.com/desktop/setup/install/mac-install/) / [Linux](https://docs.docker.com/desktop/setup/install/linux/))
    2. Open terminal/powershell and change directory to the docker-compose.yml at `Assets/Scripts/Bot/Example/ML/LLM/docker-compose.yml`
    3. Run `docker-compose up -d`, wait until done
    4. Wait 5-10 mins, or observe the status with running `docker logs sumobot-api -f --tail 20` until it shows `"GET /health HTTP/1.1" 200 OK`
    5. Done, you can run Bot_LLM which will hit the API contains process of Milvus vector search to get the action by the given bot state
    6. (Optional) If you run long-running and many simulation instance, adjust the number of API worker to support extensive requests from Bot_LLM. Locate `Assets/Scripts/Bot/Example/ML/LLM/docker-compose.yml` file, at the api service line 83, adjust value of WORKER (default set to 4) depends on local machine capability. WORKER will spawn separated process to support API request concurrently from Bot_LLM.

# Additional Information

### How Total Configurations Are Calculated

The simulator creates all possible combinations (Cartesian product) of your selected parameters:

**Example:**
```json
"Timers": [15, 30, 45, 60],          // 4 options
"ActionIntervals": [0.1, 0.2, 0.5],  // 3 options
"RoundSystem": [1, 3, 5],             // 3 options
"Skills": [0, 1],                     // 2 options
"SelectedAgents": ["Bot_BT", "Bot_DQN"]  // 2 agents = 1 matchup
```

**Total Configurations** = 4 × 3 × 3 × 2 × 1 = **72 configurations**

Each configuration represents a unique combination that will be tested during simulation.

---

## Parameters

All scripts use the same 4 parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| **sumobot_path** | String | Path to Sumobot executable or .app bundle | `"C:\Games\Sumobot.exe"` or `"/Applications/Sumobot.app/Contents/MacOS/Sumobot"` |
| **config_start** | Integer | Starting configuration index (inclusive) | `0` |
| **config_end** | Integer | Ending configuration index (exclusive) | `100` |
| **batch_size** | Integer | Number of configurations per parallel instance | `20` |



---

## How Batching Works

```
Total Configs: 100 (0 to 99)
Batch Size: 20

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Instance 1 │  │  Instance 2 │  │  Instance 3 │  │  Instance 4 │  │  Instance 5 │
│             │  │             │  │             │  │             │  │             │
│ Configs     │  │ Configs     │  │ Configs     │  │ Configs     │  │ Configs     │
│   0-19      │  │  20-39      │  │  40-59      │  │  60-79      │  │  80-99      │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
     ↓                ↓                ↓                ↓                ↓
log_0-20.txt    log_20-40.txt   log_40-60.txt   log_60-80.txt   log_80-100.txt
```

All instances run **simultaneously** (in parallel), significantly reducing total simulation time.

---

## Computer Specifications

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Recommended batch size**: `10-20`
- **Parallel instances**: 2-4

### Recommended Specifications
- **CPU**: 8+ cores (modern i7/Ryzen 7)
- **RAM**: 16 GB
- **Recommended batch size**: `20-50`
- **Parallel instances**: 4-8

### High Performance Setup
- **CPU**: 12+ cores (i9/Ryzen 9/Threadripper)
- **RAM**: 32 GB+
- **Recommended batch size**: `50-100`
- **Parallel instances**: 8-16

### How to Choose Batch Size

**Formula:**
```
Optimal Batch Size = Total Configs / (CPU Cores × Memory Factor)

Memory Factor:
- 8 GB RAM  = 0.5  (conservative)
- 16 GB RAM = 1.0  (balanced)
- 32 GB RAM = 1.5  (aggressive)
```

**Examples:**

| Total Configs | CPU Cores | RAM | Recommended Batch Size | Instances | Est. Speedup |
|--------------|-----------|-----|------------------------|-----------|--------------|
| 100 | 4 | 8 GB | 25 | 4 | 4x |
| 100 | 8 | 16 GB | 12-15 | 7-8 | 7-8x |
| 240 | 12 | 32 GB | 20 | 12 | 12x |
| 500 | 16 | 64 GB | 30 | 17 | 16x |

### Performance Tips

1. **Monitor System Resources**
   - **Windows**: Task Manager → Performance
   - **macOS**: Activity Monitor
   - **Linux**: `htop` or `top`

2. **Signs Batch Size is Too Small**
   - CPU usage < 80%
   - Too many instances (overhead)
   - Slow startup time dominates

3. **Signs Batch Size is Too Large**
   - RAM usage > 90%
   - System becomes unresponsive
   - Slowdowns or crashes

4. **Optimal Settings**
   - CPU usage: 70-90%
   - RAM usage: 60-80%
   - Instances running: Close to CPU core count

### Batch Size Examples

```bash
# Conservative (safe for most systems)
./run_simulator.sh "/path/to/Sumobot" 0 100 50
# → 2 parallel instances, low memory usage

# Balanced (recommended for 8-core, 16GB RAM)
./run_simulator.sh "/path/to/Sumobot" 0 240 30
# → 8 parallel instances, good performance

# Aggressive (high-end system with 16+ cores, 32GB+ RAM)
./run_simulator.sh "/path/to/Sumobot" 0 1000 100
# → 10 parallel instances, maximum speed
```

---

## Output Files

Each batch creates a log file:

```
log_0-20.txt       ← Configs 0-19
log_20-40.txt      ← Configs 20-39
log_40-60.txt      ← Configs 40-59
...
```

**Log files contain:**
- Simulation progress
- Configuration details
- Error messages (if any)
- Performance metrics

---

## Troubleshooting

### Error: "Executable not found"
- **Check path**: Ensure the executable path is correct
- **Use absolute path**: Avoid relative paths
- **Quotes**: Use quotes if path contains spaces
  ```bash
  # Good
  "./run_simulator.sh '/Applications/My Games/Sumobot.app/Contents/MacOS/Sumobot' 0 100 20"

  # Bad
  "./run_simulator.sh /Applications/My Games/Sumobot.app/Contents/MacOS/Sumobot 0 100 20"
  ```

### Error: "Permission denied" (macOS/Linux)
Make the script executable:
```bash
chmod +x run_simulator.sh
chmod +x /path/to/Sumobot.x86_64
```

### Simulations Running Slowly
- **Reduce batch size**: More instances = less work per instance
- **Check CPU/RAM usage**: Might be bottlenecked
- **Close other applications**: Free up resources

### System Becomes Unresponsive
- **Increase batch size**: Fewer parallel instances
- **Reduce total configs**: Process in smaller batches
- **Upgrade RAM**: If memory is bottleneck

### Some Configurations Fail
- **Check log files**: Look for errors in `log_*.txt`
- **Run single config**: Test with batch size = 1
- **Unity Player logs**: Check Unity player logs for crashes
