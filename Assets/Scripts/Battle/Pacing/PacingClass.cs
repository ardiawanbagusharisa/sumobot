using System;
using System.Collections.Generic;
using System.Linq;
using SumoCore;
using SumoManager;
using Unity.VisualScripting;
using UnityEngine;

[Serializable]
public class PacingEvaluation
{
	public int SegmentIndex;
	public float ActualThreat;
	public float TargetThreat;
	public float ThreatDelta;
	public float ActualTempo;
	public float TargetTempo;
	public float TempoDelta;

	public bool IsThreatAboveTarget => ThreatDelta > 0;
	public bool IsTempoAboveTarget => TempoDelta > 0;

	/// <summary>
	/// Creates a deep copy of this PacingEvaluation instance.
	/// Useful to prevent reference sharing when passing to methods that may modify the data.
	/// </summary>
	public PacingEvaluation Copy()
	{
		return new PacingEvaluation
		{
			SegmentIndex = this.SegmentIndex,
			ActualThreat = this.ActualThreat,
			TargetThreat = this.TargetThreat,
			ThreatDelta = this.ThreatDelta,
			ActualTempo = this.ActualTempo,
			TargetTempo = this.TargetTempo,
			TempoDelta = this.TempoDelta
		};
	}

	public override string ToString()
	{
		return $"Segment {SegmentIndex}: Threat={ActualThreat:F3} (Target={TargetThreat:F3}, Delta={ThreatDelta:F3}), " +
			   $"Tempo={ActualTempo:F3} (Target={TargetTempo:F3}, Delta={TempoDelta:F3})";
	}
}

public class SegmentPacing
{
	public ThreatAspect Threat;
	public TempoAspect Tempo;

	public SegmentPacing(SegmentData data, ConstraintConfig constraints)
	{
		Threat = new ThreatAspect(data, constraints);
		Tempo = new TempoAspect(data, constraints);
	}

	public float GetOverallPacing()
	{
		float threatValue = float.IsNaN(Threat.Value) ? 0f : Threat.Value;
		float tempoValue = float.IsNaN(Tempo.Value) ? 0f : Tempo.Value;

		float numerator = threatValue * Threat.Weight + tempoValue * Tempo.Weight;
		float denominator = Threat.Weight + Tempo.Weight;

		if (denominator == 0f)
			return 0f;

		float result = numerator / denominator;

		// Ensure result is never NaN
		return float.IsNaN(result) ? 0f : result;
	}

	/// <summary>
	/// Gets percentile-based pacing score (0-100) using linear interpolation.
	/// Maps [minPacing, maxPacing] to [0, 100].
	/// </summary>
	public float GetPercentilePacing(float minPacing, float maxPacing)
	{
		float rawPacing = GetOverallPacing();
		if (Mathf.Approximately(maxPacing, minPacing))
			return 50f;

		float percentile = ((rawPacing - minPacing) / (maxPacing - minPacing)) * 100f;
		return Mathf.Clamp(percentile, 0f, 100f);
	}

	/// <summary>
	/// Gets normalized pacing (0-1) using linear scaling.
	/// Maps [minPacing, maxPacing] to [0, 1].
	/// </summary>
	public float GetNormalizedPacing(float minPacing, float maxPacing)
	{
		float rawPacing = GetOverallPacing();
		if (Mathf.Approximately(maxPacing, minPacing))
			return 0.5f;

		float normalized = (rawPacing - minPacing) / (maxPacing - minPacing);
		return Mathf.Clamp01(normalized);
	}
}

public class ThreatAspect : Aspect
{
	public ThreatAspect(SegmentData data, ConstraintConfig constraints) : base(data, constraints)
	{
		Factors.Add(new Factor(FactorType.HitCollision, 1f));
		Factors.Add(new Factor(FactorType.Ability, 1f));
		Factors.Add(new Factor(FactorType.Angle, 1f));
		Factors.Add(new Factor(FactorType.SafeDistance, 1f));
		Calculate();
	}
}

public class TempoAspect : Aspect
{
	public TempoAspect(SegmentData data, ConstraintConfig constraints) : base(data, constraints)
	{
		Factors.Add(new Factor(FactorType.ActionIntensity, 1f));
		Factors.Add(new Factor(FactorType.ActionDensity, 1f));
		Factors.Add(new Factor(FactorType.BotsDistance, 1f));
		Factors.Add(new Factor(FactorType.Velocity, 1f));
		Calculate();
	}
}

public abstract class Aspect
{
	protected SegmentData Data;
	protected ConstraintConfig Constraints;
	protected List<Factor> Factors = new();
	public float Value { get; protected set; }
	public float Weight = 1f;

	public Aspect(SegmentData data, ConstraintConfig constraints)
	{
		Data = new SegmentData(data);
		Constraints = constraints;
		Calculate();
	}

	protected virtual void Calculate()
	{
		float weightedSum = 0f;
		float totalWeight = 0f;

		foreach (var f in Factors)
		{
			float v = f.Evaluate(Data, Constraints);

			// Skip NaN values to prevent contamination
			if (float.IsNaN(v))
				continue;

			weightedSum += v * f.Weight;
			totalWeight += f.Weight;
		}

		Value = totalWeight > 0 ? weightedSum / totalWeight : 0f;

		// Ensure Value is never NaN
		if (float.IsNaN(Value))
			Value = 0f;
	}

	// Get the list of factors including its
	public List<(AspectType aspect, FactorType factor, float value, float weight)> GetFactorsInfo()
	{
		AspectType aspectType = (this is ThreatAspect) ? AspectType.Threat : AspectType.Tempo;
		return Factors.Select(f => (aspectType, f.Type, f.Evaluate(Data, Constraints), f.Weight)).ToList();
	}
}


/// <summary>
/// Stores collision data for a segment with pre-calculated window.
/// Note: Actual collision history (per-second) is now managed by PacingHandler.
/// </summary>
[Serializable]
public class CollisionWindowData
{
	// Current segment's collisions only
	public List<CollisionType> CurrentSegmentCollisions = new();

	// Pre-calculated window including history (set during segment finalization)
	public List<CollisionType> WindowCollisions = new();

	// Window size used for calculation (for debugging/transparency)
	public int WindowSize = 0;

	public CollisionWindowData() { }

	public CollisionWindowData(CollisionWindowData other)
	{
		CurrentSegmentCollisions = new(other.CurrentSegmentCollisions);
		WindowCollisions = new(other.WindowCollisions);
		WindowSize = other.WindowSize;
	}

	/// <summary>
	/// Add collision to current segment (for legacy tracking).
	/// Note: Persistent time-based storage is now handled by PacingHandler.
	/// </summary>
	public void AddCollision(CollisionType type)
	{
		CurrentSegmentCollisions.Add(type);
	}

	/// <summary>
	/// Calculate and store the collision window using true time-based lookup (per second).
	/// Looks back N seconds in the provided collision history.
	/// </summary>
	/// <param name="collisionHistory">Persistent collision history from PacingHandler</param>
	/// <param name="currentSecond">Current game second (for looking back)</param>
	/// <param name="windowDurationSeconds">Duration in seconds to look back (e.g., 3 seconds).
	/// 0 = current second only, -1 = all history</param>
	public void CalculateWindow(Dictionary<int, List<CollisionType>> collisionHistory, int currentSecond, float windowDurationSeconds)
	{
		WindowCollisions.Clear();

		Debug.Log($"[CollisionWindowData] CalculateWindow called: currentSecond={currentSecond}, windowDurationSeconds={windowDurationSeconds}s");

		if (windowDurationSeconds == 0)
		{
			// Current second only
			if (collisionHistory.ContainsKey(currentSecond))
			{
				WindowCollisions.AddRange(collisionHistory[currentSecond]);
			}
			WindowSize = 0;
			Debug.Log($"  Mode: Current second only, WindowCollisions.Count={WindowCollisions.Count}");
		}
		else if (windowDurationSeconds < 0)
		{
			// All history
			foreach (var kvp in collisionHistory)
			{
				WindowCollisions.AddRange(kvp.Value);
			}
			WindowSize = -1;
			Debug.Log($"  Mode: All history, WindowCollisions.Count={WindowCollisions.Count}");
		}
		else
		{
			// Time-based sliding window: look back N seconds
			int windowSeconds = Mathf.CeilToInt(windowDurationSeconds);
			int startSecond = Mathf.Max(0, currentSecond - windowSeconds + 1);
			WindowSize = windowSeconds;

			Debug.Log($"  Mode: Time-based window, windowSeconds={windowSeconds}, looking back from second {startSecond} to {currentSecond}");

			for (int second = startSecond; second <= currentSecond; second++)
			{
				if (collisionHistory.ContainsKey(second))
				{
					int secondCollisions = collisionHistory[second].Count;
					WindowCollisions.AddRange(collisionHistory[second]);
					Debug.Log($"    Second {second}: added {secondCollisions} collisions, total now={WindowCollisions.Count}");
				}
			}

		}
		Debug.Log($"  Final: WindowSize={WindowSize}, WindowCollisions.Count={WindowCollisions.Count}");
	}
}

[Serializable]
public class SegmentData
{
	// Collision data with window (refactored)
	public CollisionWindowData CollisionData = new();

	// Legacy property for backwards compatibility (can be removed later)
	[DoNotSerialize]
	public List<CollisionType> Collisions => CollisionData.CurrentSegmentCollisions;

	// Threat fields
	public List<float> Angles = new();
	public List<float> SafeDistances = new();

	// Tempo fields
	public List<ISumoAction> Actions = new();
	public List<float> BotsDistances = new();
	public List<float> Velocities = new();

	public SegmentData() { }

	public SegmentData(SegmentData other)
	{
		CollisionData = new(other.CollisionData);
		Angles = new(other.Angles);
		SafeDistances = new(other.SafeDistances);
		Actions = new(other.Actions);
		BotsDistances = new(other.BotsDistances);
		Velocities = new(other.Velocities);
	}

	public void Reset()
	{
		// Note: CollisionsBySecond is now managed by PacingHandler, not cleared here
		CollisionData.CurrentSegmentCollisions.Clear();
		CollisionData.WindowCollisions.Clear();
		Angles.Clear();
		SafeDistances.Clear();
		Actions.Clear();
		BotsDistances.Clear();
		Velocities.Clear();
	}

	public void RegisterCollision(CollisionType type) => CollisionData.AddCollision(type);
	public void RegisterAngle(float angle) => Angles.Add(angle);
	public void RegisterSafeDistance(float d) => SafeDistances.Add(d);
	public void RegisterAction(ISumoAction action) => Actions.Add(action);
	public void RegisterBotsDistance(float d) => BotsDistances.Add(d);
	public void RegisterVelocity(float v) => Velocities.Add(v);

	// Helper functions
	public Dictionary<ISumoAction, int> GetActionCounts()
	{
		return Actions.GroupBy(a => a).ToDictionary(g => g.Key, g => g.Count());
	}

	public Dictionary<CollisionType, int> GetCollisionCounts()
	{
		return CollisionData.CurrentSegmentCollisions.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
	}
}

[Serializable]
public class PacingTargetConfig
{
	public List<float> ThreatTargets;
	public List<float> TempoTargets;
	public ConstraintConfig GlobalConstraints;

	// Cache for calibrated targets (computed once when min/max is set)
	[NonSerialized]
	private List<float> _calibratedThreatTargets = null;
	[NonSerialized]
	private List<float> _calibratedTempoTargets = null;
	[NonSerialized]
	private float _cachedMinPacing = -1f;
	[NonSerialized]
	private float _cachedMaxPacing = -1f;

	public PacingTargetConfig()
	{
		ThreatTargets = new List<float>();
		TempoTargets = new List<float>();
		GlobalConstraints = new ConstraintConfig();
	}

	/// <summary>
	/// Gets threat targets converted from percentile (0-1) to raw pacing values.
	/// Uses linear interpolation: Lerp(minPacing, maxPacing, percentile).
	/// </summary>
	public List<float> GetCalibratedThreatTargets(float minPacing, float maxPacing)
	{
		// Recalculate if min/max changed
		if (!Mathf.Approximately(_cachedMinPacing, minPacing) ||
		    !Mathf.Approximately(_cachedMaxPacing, maxPacing) ||
		    _calibratedThreatTargets == null)
		{
			_calibratedThreatTargets = new List<float>();
			foreach (float normalizedPercentile in ThreatTargets)
				_calibratedThreatTargets.Add(PercentileToRaw(normalizedPercentile, minPacing, maxPacing));
			_cachedMinPacing = minPacing;
			_cachedMaxPacing = maxPacing;
		}

		return _calibratedThreatTargets;
	}

	/// <summary>
	/// Gets tempo targets converted from percentile (0-1) to raw pacing values.
	/// Uses linear interpolation: Lerp(minPacing, maxPacing, percentile).
	/// </summary>
	public List<float> GetCalibratedTempoTargets(float minPacing, float maxPacing)
	{
		// Recalculate if min/max changed
		if (!Mathf.Approximately(_cachedMinPacing, minPacing) ||
		    !Mathf.Approximately(_cachedMaxPacing, maxPacing) ||
		    _calibratedTempoTargets == null)
		{
			_calibratedTempoTargets = new List<float>();
			foreach (float normalizedPercentile in TempoTargets)
				_calibratedTempoTargets.Add(PercentileToRaw(normalizedPercentile, minPacing, maxPacing));
			_cachedMinPacing = minPacing;
			_cachedMaxPacing = maxPacing;
		}

		return _calibratedTempoTargets;
	}

	/// <summary>
	/// Inverse percentile lookup: normalized percentile (0-1) → raw pacing (0-1).
	/// Uses linear interpolation: Lerp(minPacing, maxPacing, normalizedPercentile).
	/// Example: 0.90 with range [0.2, 0.8] → Lerp(0.2, 0.8, 0.90) = 0.74
	/// </summary>
	private float PercentileToRaw(float normalizedPercentile, float minPacing, float maxPacing)
	{
		// Simple linear interpolation between min and max
		float t = Mathf.Clamp01(normalizedPercentile);
		return Mathf.Lerp(minPacing, maxPacing, t);
	}

	/// <summary>
	/// Invalidates cached calibrated targets (call when min/max changes).
	/// </summary>
	public void InvalidateCache()
	{
		_calibratedThreatTargets = null;
		_calibratedTempoTargets = null;
		_cachedMinPacing = -1f;
		_cachedMaxPacing = -1f;
	}
}

[Serializable]
public class ConstraintConfig
{
	// Defaults 
	// [Todo] Consider to make scriptable object for each bot's constraints.
	public ConstraintMinMax CollisionRatio = new(0, 1);
	public ConstraintMinMax AbilityRatio = new(0, 0.2f);
	public ConstraintMinMax Angle = new(0, 180);
	public ConstraintMinMax SafeDistance = new(1, 5);

	public ConstraintMinMax ActionIntensity = new(0, 50);
	public ConstraintMinMax ActionDensity = new(0, 1);
	public ConstraintMinMax BotsDistance = new(1, 5);
	public ConstraintMinMax Velocity = new(0, 10);
}

[Serializable]
public class ConstraintMinMax
{
	public float Min;
	public float Max;
	public float MinLimit;
	public float MaxLimit;
	public float Weight = 1f;

	public ConstraintMinMax(float minLimit, float maxLimit)
	{
		MinLimit = minLimit;
		MaxLimit = maxLimit;
		Min = minLimit;
		Max = maxLimit;
	}

	public float Normalize(float value, bool absolute = false)
	{
		var min = Mathf.Max(Min, MinLimit);
		var max = Mathf.Min(Max, MaxLimit);
		if (Mathf.Approximately(max, min))
			return 0f;
		var val = (value - min) / (max - min);
		var result = absolute ? Mathf.Clamp01(Mathf.Abs(val)) : Mathf.Clamp01(val);
		return result;

	}
}

public class Factor
{
	public FactorType Type { get; private set; }
	public float Weight { get; private set; }
	public Factor(FactorType type, float weight)
	{
		Type = type;
		Weight = weight;
	}

	public float Evaluate(SegmentData data, ConstraintConfig constraints)
	{
		float score;
		// HitCollision requires history and window size; other factors use only current segment data
		if (Type == FactorType.HitCollision)
			score = EvaluateHitCollisionWithWindow(data, constraints);
		else if (Type == FactorType.Ability)
			score = EvaluateAbility(data, constraints);
		else if (Type == FactorType.Angle)
			score = EvaluateAngle(data, constraints);
		else if (Type == FactorType.SafeDistance)
			score = EvaluateSafeDistance(data, constraints);
		else if (Type == FactorType.ActionIntensity)
			score = EvaluateActionIntensity(data, constraints);
		else if (Type == FactorType.ActionDensity)
		{
			score = EvaluateActionDensity(data, constraints);
		}
		else if (Type == FactorType.BotsDistance)
			score = EvaluateBotsDistance(data, constraints);
		else if (Type == FactorType.Velocity)
			score = EvaluateVelocity(data, constraints);
		else
			throw new ArgumentException("Invalid factor type");

		return score;
	}

	/// <summary>
	/// Evaluate the ratio of hit collisions among all collisions using a sliding window.
	/// windowSize: -1 = all history, 0 = current only, N = last N segments from history
	/// </summary>
	private float EvaluateHitCollisionWithWindow(SegmentData data, ConstraintConfig constraints)
	{
		// Use pre-calculated collision window from SegmentData
		// The window was already calculated when the segment was finalized
		var windowCollisions = data.CollisionData.WindowCollisions;

		// Count hits and strucks from the pre-calculated window
		float totalHitCollisions = windowCollisions.Count(c => c == CollisionType.Hit);
		float totalStruckCollisions = windowCollisions.Count(c => c == CollisionType.Struck);

		Debug.Log($"[HitCollision] WindowSize={data.CollisionData.WindowSize}, Window Total={windowCollisions.Count}, Hits={totalHitCollisions}, Strucks={totalStruckCollisions}");

		if (totalHitCollisions + totalStruckCollisions == 0f)
		{
			// No collisions in window
			Debug.Log($"  No collisions in window, returning normalized 0");
			return constraints.CollisionRatio.Normalize(0f);
		}

		float hitCollisionRatio = totalHitCollisions / (totalHitCollisions + totalStruckCollisions);
		float normalized = constraints.CollisionRatio.Normalize(hitCollisionRatio);
		Debug.Log($"  Collision Ratio={hitCollisionRatio:F3}, Normalized={normalized:F3}");
		return normalized;
	}

	// Evaluate the ratio of ability usage among all actions.
	private float EvaluateAbility(SegmentData data, ConstraintConfig constraints)
	{
		if (data.Actions.Count == 0) return 0f;

		float abilityCount = data.Actions.Count(a => a.Type == ActionType.Dash || a.Type == ActionType.SkillBoost || a.Type == ActionType.SkillStone);
		float abilityRatio = abilityCount / data.Actions.Count;

		return constraints.AbilityRatio.Normalize(abilityRatio);
	}

	// Evaluate the average angle between the bot and its opponents when they collide or are close.
	// Inverted: small angle (head-on) = high threat, large angle (side/back) = low threat
	private float EvaluateAngle(SegmentData data, ConstraintConfig constraints)
	{
		return data.Angles.Count > 0 ? 1f - constraints.Angle.Normalize(data.Angles.Average()) : 0f;
	}

	// Evaluate the average distance between the bot and its opponents when they collide or are close.
	// Inverted: small distance (close combat) = high threat, large distance (far) = low threat
	private float EvaluateSafeDistance(SegmentData data, ConstraintConfig constraints)
	{
		return data.SafeDistances.Count > 0 ? 1f - constraints.SafeDistance.Normalize(data.SafeDistances.Average()) : 0f;
	}

	// Evaluate the number of actions performed by the bot.
	private float EvaluateActionIntensity(SegmentData data, ConstraintConfig constraints)
	{
		return constraints.ActionIntensity.Normalize(data.Actions.Count);
	}

	// Evaluate the entropy of the action distribution.
	private float EvaluateActionDensity(SegmentData data, ConstraintConfig constraints)
	{
		if (data.Actions.Count == 0) return 0f;
		var counts = data.GetActionCounts();
		float total = data.Actions.Count;
		float entropy = 0f;
		foreach (var c in counts.Values)
		{
			float p = c / total;
			entropy -= p * Mathf.Log(p, 2); // Log with base 2
		}
		return constraints.ActionDensity.Normalize(entropy);
	}

	// Evaluate the average distance between the bot and its opponents.
	private float EvaluateBotsDistance(SegmentData data, ConstraintConfig constraints)
	{
		return data.BotsDistances.Count > 0 ? 1f - constraints.BotsDistance.Normalize(data.BotsDistances.Average()) : 0f;
	}

	// Evaluate the average velocity of the bot.
	private float EvaluateVelocity(SegmentData data, ConstraintConfig constraints)
	{
		return data.Velocities.Count > 0 ? constraints.Velocity.Normalize(data.Velocities.Average()) : 0f;
	}
}

public class GamePacingItem
{
	public List<SegmentData> SegmentGameplayDatas = new();
	public List<SegmentPacing> SegmentPacings = new();
}

public class GamePacing
{
	public Dictionary<int, Dictionary<int, GamePacingItem>> PacingHistories = new() { };

	public void InitBattle()
	{
		var currGameIdx = LogManager.CurrentGameIndex;
		var roundIdx = LogManager.GetCurrentRound().Index;

		Logger.Info($"[PacingClass][GamePacing] InitBattle called for Game {currGameIdx}, Round {roundIdx}");

		// Always create a fresh GamePacingItem for the round to prevent cross-round contamination
		if (PacingHistories.TryGetValue(currGameIdx, out var rounds))
		{
			// Overwrite existing round data with fresh instance (important for rematches/new rounds)
			rounds[roundIdx] = new GamePacingItem();
			Logger.Info($"  Replaced existing round data for Game {currGameIdx}, Round {roundIdx}");
		}
		else
		{
			// Create new game entry with first round
			PacingHistories.Add(currGameIdx, new() { [roundIdx] = new GamePacingItem() });
			Logger.Info($"  Created new game entry for Game {currGameIdx}, Round {roundIdx}");
		}
	}

	public GamePacingItem CurrentRound()
	{
		var currGameIdx = LogManager.CurrentGameIndex;
		var currRound = LogManager.GetCurrentRound();

		if (currRound == null)
		{
			Logger.Error("[GamePacing] CurrentRound() called but LogManager.GetCurrentRound() is null!");
			return new GamePacingItem(); // Return empty instance to prevent crashes
		}

		var roundIdx = currRound.Index;

		// Ensure game exists in history
		if (!PacingHistories.ContainsKey(currGameIdx))
		{
			Logger.Warning($"[GamePacing] Game {currGameIdx} not found in history. Creating new entry.");
			PacingHistories[currGameIdx] = new Dictionary<int, GamePacingItem>();
		}

		// Ensure round exists in game
		if (!PacingHistories[currGameIdx].ContainsKey(roundIdx))
		{
			Logger.Warning($"[GamePacing] Round {roundIdx} not found for Game {currGameIdx}. Creating new GamePacingItem.");
			Logger.Warning($"  This might indicate InitBattle() was not called for this round!");
			PacingHistories[currGameIdx][roundIdx] = new GamePacingItem();
		}

		return PacingHistories[currGameIdx][roundIdx];
	}
}

public enum AspectType { Threat, Tempo }

public enum CollisionType { Hit, Struck, Tie }

public enum FactorType
{
	HitCollision, Ability, Angle, SafeDistance,             // Threat factors
	ActionIntensity, ActionDensity, BotsDistance, Velocity  // Tempo factors
}

