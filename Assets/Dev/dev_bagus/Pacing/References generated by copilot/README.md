# Pacing Framework - Implementation Guide

## Overview

The pacing framework is now implemented as an **extensive wrapper layer** that can be applied to any bot without modifying its source code. It automatically detects the bot type and applies the corresponding pacing profile, with comprehensive logging and inspector visualization.

## Architecture

### Core Components

1. **PacingProfile.cs** (`Assets/Scripts/Bot/Example/Pacing/`)
   - Defines global and local constraints for threat/tempo factors
   - Stores factor weights and target pacing patterns
   - Supports action filtering based on pacing thresholds
   - Can be customized per bot type

2. **PacedAgent.cs** (`Assets/Scripts/Bot/Example/Pacing/`)
   - Generic wrapper that works with any bot
   - Auto-loads pacing profile based on wrapped bot type
   - Intercepts and filters actions
   - Tracks action history (original vs filtered)
   - Logs detailed summaries at game end

3. **PacingDebugDisplay.cs** (`Assets/Scripts/Bot/Example/Pacing/`)
   - Provides in-game and inspector visualization
   - Shows real-time pacing curves (actual vs target)
   - Displays pacing factors and action statistics
   - Can be attached to a GameObject for overlay debugging

## Setup Instructions

### Step 1: Create Pacing Profiles

For each bot type you want to pace:

1. Create a folder: `Assets/Resources/Pacing/`
2. Right-click → Create → SumoBot → PacingProfile
3. Name it `Pacing_BT.asset`, `Pacing_DQN.asset`, etc.
4. Configure:
   - **Segment Duration**: How long each pacing segment is (5-15s recommended)
   - **Threat/Tempo Weights**: Balance between threat and tempo factors
   - **Global Constraints**: Min/max ranges for normalization
   - **Target Pattern**: Constant, Linear, Exponential, or Custom curve
   - (Optional) **Action Filtering**: Enable and select which actions to filter

Example configuration for `Pacing_BT`:
```
Segment Duration: 10s
Threat Weight: 0.5
Tempo Weight: 0.5
Pattern: LinearIncrease (or your chosen pattern)
```

### Step 2: Wrap Your Bot

Instead of using the bot directly:

**Before:**
```csharp
// Use AIBot_BT directly
```

**After:**
1. In the battle setup, use `PacedAgent` instead
2. Assign the original bot to the `wrappedBot` field
3. Leave `pacingProfile` empty (it auto-loads from Resources)

### Step 3: Optional - Inspector Visualization

1. Create an empty GameObject in your scene: `DebugPacing`
2. Add component: `PacingDebugDisplay`
3. Assign the `PacedAgent` instance to the `pacedAgent` field
4. Toggle `showPacingCurve`, `showActionStatistics`, `showPacingFactors`
5. Run the game - the overlay will display in the top-left corner

## How It Works

### Pacing Calculation

Each update cycle:

1. **Collect Telemetry**: Sample distance, angles, velocities, collisions, skill status
2. **Compute Factors**: Normalize each threat/tempo factor against constraints
3. **Combine Scores**: 
   - Threat = weighted average of [collision, enemy skill, angle delta, distance delta]
   - Tempo = weighted average of [action intensity, density, distance to enemy, velocity delta]
   - Overall = weighted blend of Threat and Tempo
4. **Evaluate Target**: Use the pacing curve to get the target pacing value for current elapsed time
5. **Output Frame**: PacingFrame contains all metrics + history for analysis

### Action Filtering

If `enableActionFiltering` is true in the profile:

1. Each action queued by the bot is checked against the filter list
2. If action type is in `filterableActions` AND current pacing < `filterThreshold`, action is suppressed
3. Filtered actions are logged and tracked in `ActionHistory`
4. At game end, a detailed breakdown is printed to console

Example:
```
Filter Threshold: 0.5
Filterable Actions: [TurnLeft, TurnRight]

If pacing = 0.3 (low): Turn actions are blocked
If pacing = 0.7 (high): Turn actions pass through
```

## Logging & Analysis

### Console Output

At the end of each game, you'll see:

```
========== PACING SUMMARY for Bot_BT ==========
Total Original Actions: 245
Total Filtered Actions: 23
Filtering Rate: 9.4%

--- Original Action Breakdown ---
  Accelerate: 120
  TurnLeft: 68
  TurnRight: 57

--- Filtered Action Breakdown ---
  TurnLeft: 15
  TurnRight: 8

--- Pacing Statistics ---
Pacing Segments: 4
Avg Threat: 0.412
Avg Tempo: 0.523
Avg Current Pacing: 0.468
Avg Target Pacing: 0.500
=======================================
```

### Inspector Data

The `PacingDebugDisplay` shows:
- **Pacing Curve**: Real-time cyan line (actual) vs yellow line (target)
- **Action Statistics**: Breakdown of original and filtered actions
- **Pacing Factors**: Real-time values of each threat/tempo component
- **Current State**: Segment index, elapsed time, current vs target pacing

## Constraint Calibration

To calibrate constraints for your arena:

1. Run multiple games with baseline bots (no pacing)
2. Check console logs for typical ranges of each variable
3. Update `globalConstraints` in the profile:
   - `min` = typical minimum observed value
   - `max` = typical maximum observed value
4. Normalize values should then map ~0 to ~1

Example for collision ratio:
```
If observed range is 0.1 to 0.4:
  collisionRatio = MinMax(0.1, 0.4)
```

## Advanced: Local Constraints

For per-segment customization:

1. In the profile, add entries to `localConstraints`
2. Set `segmentIndex` (0 = first 10s, 1 = next 10s, etc.)
3. Set `weight` (0 = ignore local, 1 = fully override global)
4. Define segment-specific constraint ranges

Example: Make attacks harder in the final segment
```
Segment 3 (final 10s):
  weight: 0.8
  attackDistance: [1.5, 1.5]  (strict)
  agentAngle: [5, 5]  (must be perfectly aligned)
```

## Troubleshooting

### Profile Not Loading

```
[Bot_BT] Pacing profile not found at: Assets/Resources/Pacing/Pacing_BT.asset
```

**Solution**: Ensure:
- Profile asset exists at `Assets/Resources/Pacing/Pacing_BT.asset`
- Naming matches bot ID suffix exactly (case-sensitive)
- No spaces or special characters in the path

### Actions Not Being Filtered

- Verify `enableActionFiltering = true` in profile
- Check `filterThreshold` is not too low (0 = no filtering)
- Ensure desired action types are in `filterableActions` list
- Check console for "Filtered action" log messages

### Pacing Curve Looks Flat

- Verify constraints aren't inverted (min > max)
- Check that constraints match your game's data range
- If using custom curve, ensure it has variation (not horizontal line)
- Try increasing threat/tempo weights to amplify signals

## Next Steps for Analysis

Use the pacing data in Python notebooks:

1. Export pacing history + action logs
2. Compare different pacing patterns
3. Analyze win rate vs pacing curve shape
4. Correlate actions filtered vs performance drop
5. Fine-tune weights for each bot type

Example analysis:
```python
# Plot pacing vs win rate
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(pacing_history, label='Pacing')
ax1.plot(target_curve, label='Target')
ax1.legend()
ax1.set_title('Pacing Over Time')
ax2.bar(action_types, filter_counts)
ax2.set_title('Actions Filtered by Type')
plt.show()
```

## Files Created

- `Assets/Scripts/Bot/Example/Pacing/PacingProfile.cs` - Profile definition + controller
- `Assets/Scripts/Bot/Example/Pacing/PacedAgent.cs` - Generic wrapper bot
- `Assets/Scripts/Bot/Example/Pacing/PacingDebugDisplay.cs` - Inspector visualization
- `Assets/Resources/Pacing/Pacing_BT.asset` - Example profile (create for each bot)

## Implementation Notes

- The wrapper is **non-invasive**: original bot code unchanged
- **Auto-load**: No manual profile assignment needed (uses Resources folder)
- **Comprehensive logging**: Track every filtered action + final summary
- **Real-time viz**: Inspector shows curves and stats during play
- **Extensible**: Easy to add new filters or metrics
