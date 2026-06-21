using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoHelper;
using SumoInput;
using UnityEngine;

namespace PacingFramework
{

	/// <summary>
	/// Heuristic-based pacing brain using Utility AI approach.
	/// NO TRAINING REQUIRED - uses rule-based scoring to select actions that achieve pacing targets.
	/// Supports intelligent circling strategy for low pacing targets.
	/// </summary>
	public class PacingBrainHeuristic
	{
		#region Configuration
		// Pacing consideration weights
		public float ThreatDeltaWeight = 1.0f;           // How much threat delta affects scoring
		public float TempoDeltaWeight = 1.0f;            // How much tempo delta affects scoring
		public float BoundarySafetyWeight = 2.0f;        // Boundary avoidance importance (high priority!)
		public float CirclingBonusWeight = 1.5f;         // Bonus for circling when targets are low

		// Circling strategy thresholds
		public float CirclingSafeDistance = 0.3f;        // Min distance from enemy to circle safely
		public float CirclingTurnBonus = 1.5f;           // Score multiplier for turns when circling
		public float CirclingAccelBonus = 1.3f;          // Score multiplier for accel when circling
		public float CirclingAggressionPenalty = 0.5f;   // Score multiplier for dash/skills when circling

		// No-action threshold for passive behavior
		public float NoActionThreshold = 0.3f;          // Above this avg delta = return null (no actions)

		// Action base scores (before considerations)
		public float BaseScoreAccelerate = 0.7f;
		public float BaseScoreTurn = 0.5f;
		public float BaseScoreDash = 0.4f;
		public float BaseScoreSkill = 0.4f;

		// Action type preferences based on pacing needs
		public MinMax AccelerateMultiplier = new(0.5f, 0.9f);
		public MinMax DashMultiplier = new(0.5f, 1f);
		public MinMax SkillMultiplier = new(0.5f, 1f);
		public MinMax TurnMultiplier = new(0.3f, 0.7f);
		#endregion

		private SumoController controller;

		#region Constructor & Initialization
		public PacingBrainHeuristic(SumoController controller)
		{
			this.controller = controller;
		}

		/// <summary>
		/// Updates the controller reference. Used when PacingBrain is reused across rounds.
		/// </summary>
		public void UpdateController(SumoController newController)
		{
			this.controller = newController;
		}
		#endregion

		#region Action Evaluation & Selection
		/// <summary>
		/// Evaluates a candidate action using utility-based heuristics.
		/// Higher score = better for achieving pacing balance.
		/// </summary>
		public float EvaluateAction(ISumoAction action, PacingEvaluation currentPacing, SumoAPI api, List<ISumoAction> previousActions)
		{
			// Build context for evaluation
			var context = BuildContext(currentPacing, api, previousActions);

			// Calculate base score for action type
			float baseScore = GetBaseScore(action);

			// Apply pacing considerations
			float pacingScore = EvaluatePacingFit(action, context);

			// Apply boundary safety
			float safetyScore = EvaluateBoundarySafety(action, context);

			// Apply circling strategy bonus if applicable
			float circlingBonus = EvaluateCirclingStrategy(action, context);

			// Combine scores (multiplicative for strong preferences, additive for bonuses)
			float finalScore = baseScore + pacingScore + safetyScore + circlingBonus;

			return finalScore;
		}

		/// <summary>
		/// Selects the best action from candidates based on heuristic evaluation.
		/// Returns null if targets are very low (passive behavior).
		/// </summary>
		public ISumoAction SelectBestAction(List<ISumoAction> candidates, PacingEvaluation currentPacing, SumoAPI api, List<ISumoAction> previousActions)
		{
			if (candidates.Count == 0)
				return null;

			// Check if we're exceeding targets (positive delta) - if so, return null for passive behavior
			float avgDelta = (currentPacing.ThreatDelta + currentPacing.TempoDelta) / 2f;
			// Dynamic threshold scales with targets: high targets = higher threshold (more permissive)
			float avgTarget = (currentPacing.TargetThreat + currentPacing.TargetTempo) / 2f;
			float dynamicThreshold = Mathf.Lerp(0.3f, 0.5f, avgTarget);

			if (avgDelta > dynamicThreshold)
			{
				return null; // Already meeting/exceeding targets = do nothing (passive)
			}

			float bestScore = float.MinValue;
			ISumoAction bestAction = candidates[0];

			foreach (var action in candidates)
			{
				float score = EvaluateAction(action, currentPacing, api, previousActions);

				if (score > bestScore)
				{
					bestScore = score;
					bestAction = action;
				}
			}

			return bestAction;
		}
		#endregion

		#region Evaluation Context
		private struct EvaluationContext
		{
			public float ThreatDelta;           // Negative = need more threat, Positive = too much threat
			public float TempoDelta;            // Negative = need more tempo, Positive = too much tempo
			public float TargetThreat;          // Target threat value (0-1)
			public float TargetTempo;           // Target tempo value (0-1)
			public float AvgTarget;             // Average of target threat/tempo
			public float DistanceFromCenter;    // 0 = center, 1 = edge
			public float DistanceToEnemy;       // Normalized distance
			public float AngleToEnemy;          // Angle to enemy in degrees
			public float FacingToOutside;       // +1 = facing out (dangerous), -1 = facing in (safe)
			public bool IsInDangerZone;         // Close to arena edge AND facing outward
			public bool IsFacingEnemy;          // Angle to enemy < 20 degrees
			public bool ShouldCircle;           // Low targets + safe from enemy = circle
		}

		private EvaluationContext BuildContext(PacingEvaluation pacing, SumoAPI api, List<ISumoAction> previousActions)
		{
			var (currentPos, currentRot) = previousActions.Count > 0
				? api.Simulate(previousActions)
				: (api.MyRobot.Position, api.MyRobot.Rotation);

			float distFromCenter = currentPos.magnitude / api.BattleInfo.ArenaRadius;
			float distToEnemy = api.DistanceNormalized(currentPos, api.EnemyRobot.Position);
			float avgTarget = (pacing.TargetThreat + pacing.TargetTempo) / 2f;
			float avgDelta = (pacing.ThreatDelta + pacing.TempoDelta) / 2f;

			// Calculate angle to enemy
			float angleToEnemy = api.Angle();
			bool isFacingEnemy = Mathf.Abs(angleToEnemy) < 30;

			// Calculate facing direction relative to arena center
			Vector2 centerToMe = (currentPos - api.BattleInfo.ArenaPosition).normalized;
			float zRot = currentRot % 360f;
			if (zRot < 0) zRot += 360f;
			Vector2 facingDir = Quaternion.Euler(0, 0, zRot) * Vector2.up;
			float facingToOutside = Vector2.Dot(facingDir, centerToMe);

			// Danger zone: close to edge (>0.40) AND facing outward (>-0.3)
			bool isInDangerZone = distFromCenter > 0.55f && facingToOutside > 0.3f;

			// Circle when we're meeting/exceeding pacing targets (positive delta) and enemy is far
			bool shouldCircle = avgTarget < 0.3f && distToEnemy > CirclingSafeDistance;

			return new EvaluationContext
			{
				ThreatDelta = pacing.ThreatDelta,
				TempoDelta = pacing.TempoDelta,
				TargetThreat = pacing.TargetThreat,
				TargetTempo = pacing.TargetTempo,
				AvgTarget = avgTarget,
				DistanceFromCenter = distFromCenter,
				DistanceToEnemy = distToEnemy,
				AngleToEnemy = angleToEnemy,
				FacingToOutside = facingToOutside,
				IsInDangerZone = isInDangerZone,
				IsFacingEnemy = isFacingEnemy,
				ShouldCircle = shouldCircle,
			};
		}
		#endregion

		#region Scoring Functions
		private float GetBaseScore(ISumoAction action)
		{
			switch (action.Type)
			{
				case ActionType.Accelerate:
					return BaseScoreAccelerate;
				case ActionType.TurnLeft:
				case ActionType.TurnRight:
					return BaseScoreTurn;
				case ActionType.Dash:
					return BaseScoreDash;
				case ActionType.SkillBoost:
				case ActionType.SkillStone:
					return BaseScoreSkill;
				default:
					return 0.5f;
			}
		}

		/// <summary>
		/// Uses target value (0-1) to interpolate multiplier.
		/// High target (1.0) = favor max multiplier, Low target (0.0) = favor min multiplier
		/// </summary>
		private float GetMultiplierFromTarget(float target, MinMax multiplierRange, bool inverse = false)
		{
			// Target is already normalized 0-1
			float t = Mathf.Clamp01(target);

			// t = 0 (low target) → use min multiplier
			// t = 1 (high target) → use max multiplier
			// Lerp from min to max as t goes from 0 to 1
			return Mathf.Lerp(multiplierRange.min, multiplierRange.max, t);
		}

		/// <summary>
		/// Uses delta (actual - target) to interpolate multiplier based on urgency.
		/// Delta < 0 (need more) = favor max multiplier, Delta > 0 (too much) = favor min multiplier
		/// </summary>
		private float GetMultiplierFromDelta(float delta, MinMax multiplierRange, bool inverse = false)
		{
			// Normalize delta to 0-1: clamp to range [-0.5, 0.5] then map to [0, 1]
			float normalizedDelta = Mathf.Clamp(delta, -0.5f, 0.5f);
			float t = (normalizedDelta + 0.5f) / 1.0f; // Map [-0.5, 0.5] to [0, 1]

			// If inverse, flip the t value (for actions that reduce instead of increase)
			if (inverse)
				t = 1f - t;

			// t = 0 (delta = -0.5, desperately need more) → use max multiplier
			// t = 1 (delta = +0.5, way too much) → use min multiplier
			// Lerp from max to min as t goes from 0 to 1
			return Mathf.Lerp(multiplierRange.min, multiplierRange.max, t);
		}

		/// <summary>
		/// Evaluates how well the action fits current pacing needs.
		/// Returns a multiplier (0-2 range, 1 = neutral).
		/// </summary>
		private float EvaluatePacingFit(ISumoAction action, EvaluationContext ctx)
		{
			float threatMultiplier = 1.0f;
			float tempoMultiplier = 1.0f;

			switch (action.Type)
			{
				case ActionType.Accelerate:
					// Acceleration increases tempo and helps close distance for threat
					// Use delta for responsive adjustment based on current need
					threatMultiplier = GetMultiplierFromDelta(ctx.ThreatDelta, AccelerateMultiplier);
					tempoMultiplier = GetMultiplierFromDelta(ctx.TempoDelta, AccelerateMultiplier);

					// If low target AND facing enemy, favor acceleration (maintain presence without aggression)
					// if (ctx.ThreatDelta < 0.25f && ctx.IsFacingEnemy)
					// {
					// 	threatMultiplier *= 1.5f; // Boost to prefer movement over turning
					// 	tempoMultiplier *= 1.5f;
					// }
					break;

				case ActionType.TurnLeft:
				case ActionType.TurnRight:
					// Turning reduces threat (defensive repositioning), inverse = true
					// TurnMultiplier is (1.5, 0.5) so high target → 0.5 (low turning) naturally
					threatMultiplier = GetMultiplierFromDelta(ctx.ThreatDelta, TurnMultiplier);
					tempoMultiplier = GetMultiplierFromDelta(ctx.TempoDelta, TurnMultiplier);

					// If low target AND already facing enemy, heavily penalize turning (prevent jiggling)
					// if (ctx.ThreatDelta > 0.3f && ctx.IsFacingEnemy)
					// {
					// 	threatMultiplier *= 0.5f; // Strong penalty to avoid unnecessary turning
					// 	tempoMultiplier *= 0.5f;
					// }
					break;

				case ActionType.Dash:
					// Dash increases both threat and tempo significantly
					threatMultiplier = GetMultiplierFromDelta(ctx.ThreatDelta, DashMultiplier);
					tempoMultiplier = GetMultiplierFromDelta(ctx.TempoDelta, DashMultiplier);

					// If high target but NOT facing enemy, penalize dash (ineffective if not aimed)
					// if (ctx.AvgTarget > 0.5f && !ctx.IsFacingEnemy)
					// {
					// 	threatMultiplier *= 0.5f; // Reduce effectiveness when not aimed
					// 	tempoMultiplier *= 0.5f;
					// }
					break;

				case ActionType.SkillBoost:
				case ActionType.SkillStone:
					// Skills increase threat significantly
					threatMultiplier = GetMultiplierFromDelta(ctx.ThreatDelta, SkillMultiplier);
					tempoMultiplier = GetMultiplierFromDelta(ctx.TempoDelta, SkillMultiplier);

					// If high target but NOT facing enemy, penalize skills (ineffective if not aimed)
					// if (ctx.AvgTarget > 0.5f && !ctx.IsFacingEnemy)
					// {
					// 	threatMultiplier *= 0.3f; // Strong penalty since skills are precision actions
					// 	tempoMultiplier *= 0.3f;
					// }
					break;
			}

			// Average threat and tempo multipliers for combined score
			float combinedScore = (threatMultiplier + tempoMultiplier) / 2f;
			return Mathf.Max(combinedScore, 5);
		}

		/// <summary>
		/// Evaluates boundary safety. Returns multiplier that heavily penalizes dangerous moves.
		/// </summary>
		private float EvaluateBoundarySafety(ISumoAction action, EvaluationContext ctx)
		{
			float score = 1.0f;

			// Check if near edge but facing inward (safer condition)
			bool nearEdgeButSafe = ctx.DistanceFromCenter > 0.50f && ctx.FacingToOutside < 0.5f;

			// If in danger zone (near edge AND facing outward), penalize aggressive actions
			if (ctx.IsInDangerZone)
			{
				switch (action.Type)
				{
					case ActionType.Dash:
						score *= (0.1f + (ctx.FacingToOutside < 0.3f ? 2f : -0.1f)); // Risky when near edge
						break;
					case ActionType.SkillBoost:
					case ActionType.SkillStone:
						score *= 0.5f; // Skills can push us out
						break;
					case ActionType.Accelerate:
						score *= (0.5f + (ctx.FacingToOutside < 0.3f ? 1f : -0.1f)); // Stronger penalty to prevent aggressive forward movement near edge
						break;
					case ActionType.TurnLeft:
					case ActionType.TurnRight:
						// Extra bonus if facing outward (need to turn around!)
						float turnBonus = ctx.FacingToOutside < 0.3f ? 2.0f : 1.5f;
						score *= turnBonus; // Strongly favor turning to reposition safely
						break;
				}

				// Extra penalty if VERY close to edge
				if (ctx.DistanceFromCenter > 0.7f)
				{
					score *= 0.3f; // Severe penalty to all actions in critical zone
				}
			}
			else if (nearEdgeButSafe)
			{
				// Near edge but facing inward - allow aggressive plays with small penalty
				switch (action.Type)
				{
					case ActionType.Dash:
					case ActionType.SkillBoost:
					case ActionType.SkillStone:
						score *= 0.8f; // Slight penalty but not as harsh as danger zone
						break;
					case ActionType.Accelerate:
						score *= 0.9f; // Allow forward movement when facing center
						break;
				}
			}
			else
			{
				// Safe from boundary - slight bonus for aggressive plays
				if (action.Type == ActionType.Dash || action.Type == ActionType.SkillBoost || action.Type == ActionType.SkillStone)
					score *= 1.1f;
			}

			return score;
		}

		/// <summary>
		/// Evaluates circling strategy bonus.
		/// When targets are low and enemy is far, favor circling (high tempo, low threat).
		/// </summary>
		private float EvaluateCirclingStrategy(ISumoAction action, EvaluationContext ctx)
		{
			if (!ctx.ShouldCircle)
				return 0f; // No bonus

			float bonus = 0f;

			switch (action.Type)
			{
				case ActionType.TurnLeft:
				case ActionType.TurnRight:
					// Turning is key for circling
					bonus = CirclingBonusWeight * CirclingTurnBonus;
					break;

				case ActionType.Accelerate:
					// Acceleration maintains tempo while circling
					bonus = CirclingBonusWeight * CirclingAccelBonus * 0.5f;
					break;

				case ActionType.Dash:
					// Dash can help circle faster, but risky
					bonus = CirclingBonusWeight * 0.3f;
					break;

				case ActionType.SkillBoost:
				case ActionType.SkillStone:
					// Skills break circling pattern - penalty instead of bonus
					bonus = -CirclingBonusWeight * CirclingAggressionPenalty;
					break;
			}

			return bonus;
		}
		#endregion

		#region Lifecycle (No-op for heuristic approach)
		// These methods exist for interface compatibility with NN version
		public void TrainFromExperience(PacingEvaluation currentPacing)
		{
			// No training needed for heuristic approach
		}

		public void OnEpisodeEnd()
		{
			// No episode tracking needed
		}

		public void OnRoundStart()
		{
			// No state to reset
		}

		public void SaveModelToDisk()
		{
			// No model to save
		}

		public int GetEpisodeCount()
		{
			return 0; // Not applicable
		}
		#endregion
	}
}