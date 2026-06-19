using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
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
		public float LowPacingThreshold = 0.3f;          // Below this avg target = prefer circling
		public float CirclingSafeDistance = 0.4f;        // Min distance from enemy to circle safely
		public float CirclingTurnBonus = 1.5f;           // Score multiplier for turns when circling
		public float CirclingAccelBonus = 1.3f;          // Score multiplier for accel when circling
		public float CirclingAggressionPenalty = 0.5f;   // Score multiplier for dash/skills when circling

		// Action base scores (before considerations)
		public float BaseScoreAccelerate = 0.6f;
		public float BaseScoreTurn = 0.5f;
		public float BaseScoreDash = 0.4f;
		public float BaseScoreSkill = 0.4f;

		// Action type preferences based on pacing needs
		public float ThreatReductionTurnPreference = 1.2f;     // Prefer turns when reducing threat
		public float TempoBoostDashPreference = 1.5f;          // Prefer dash when boosting tempo
		public float AggressionSkillPreference = 1.4f;         // Prefer skills when need aggression
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
			float finalScore = baseScore * pacingScore * safetyScore + circlingBonus;

			return Mathf.Clamp01(finalScore);
		}

		/// <summary>
		/// Selects the best action from candidates based on heuristic evaluation.
		/// </summary>
		public ISumoAction SelectBestAction(List<ISumoAction> candidates, PacingEvaluation currentPacing, SumoAPI api, List<ISumoAction> previousActions)
		{
			if (candidates.Count == 0)
				return null;

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
			public float AvgTarget;             // Average of target threat/tempo
			public float DistanceFromCenter;    // 0 = center, 1 = edge
			public float DistanceToEnemy;       // Normalized distance
			public bool IsInDangerZone;         // Close to arena edge
			public bool ShouldCircle;           // Low targets + safe from enemy = circle
			public bool NeedHigherThreat;       // ThreatDelta < -0.1
			public bool NeedHigherTempo;        // TempoDelta < -0.1
			public bool NeedLowerThreat;        // ThreatDelta > 0.1
			public bool NeedLowerTempo;         // TempoDelta > 0.1
		}

		private EvaluationContext BuildContext(PacingEvaluation pacing, SumoAPI api, List<ISumoAction> previousActions)
		{
			var (currentPos, _) = previousActions.Count > 0
				? api.Simulate(previousActions)
				: (api.MyRobot.Position, api.MyRobot.Rotation);

			float distFromCenter = currentPos.magnitude / api.BattleInfo.ArenaRadius;
			float distToEnemy = api.DistanceNormalized(currentPos, api.EnemyRobot.Position);
			float avgTarget = (pacing.TargetThreat + pacing.TargetTempo) / 2f;

			bool shouldCircle = avgTarget < LowPacingThreshold && distToEnemy > CirclingSafeDistance;

			return new EvaluationContext
			{
				ThreatDelta = pacing.ThreatDelta,
				TempoDelta = pacing.TempoDelta,
				AvgTarget = avgTarget,
				DistanceFromCenter = distFromCenter,
				DistanceToEnemy = distToEnemy,
				IsInDangerZone = distFromCenter > 0.6f,
				ShouldCircle = shouldCircle,
				NeedHigherThreat = pacing.ThreatDelta < -0.1f,
				NeedHigherTempo = pacing.TempoDelta < -0.1f,
				NeedLowerThreat = pacing.ThreatDelta > 0.1f,
				NeedLowerTempo = pacing.TempoDelta > 0.1f
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
		/// Evaluates how well the action fits current pacing needs.
		/// Returns a multiplier (0-2 range, 1 = neutral).
		/// </summary>
		private float EvaluatePacingFit(ISumoAction action, EvaluationContext ctx)
		{
			float score = 1.0f; // Neutral baseline

			switch (action.Type)
			{
				case ActionType.Accelerate:
					// Acceleration increases tempo moderately
					if (ctx.NeedHigherTempo)
						score *= 1.3f;
					else if (ctx.NeedLowerTempo)
						score *= 0.7f;
					break;

				case ActionType.TurnLeft:
				case ActionType.TurnRight:
					// Turning reduces threat (defensive repositioning)
					if (ctx.NeedLowerThreat)
						score *= ThreatReductionTurnPreference;
					else if (ctx.NeedHigherThreat)
						score *= 0.8f;
					break;

				case ActionType.Dash:
					// Dash increases both threat and tempo significantly
					if (ctx.NeedHigherTempo || ctx.NeedHigherThreat)
						score *= TempoBoostDashPreference;
					else if (ctx.NeedLowerTempo || ctx.NeedLowerThreat)
						score *= 0.4f; // Strong penalty for dash when need to calm down
					break;

				case ActionType.SkillBoost:
				case ActionType.SkillStone:
					// Skills are high threat actions
					if (ctx.NeedHigherThreat)
						score *= AggressionSkillPreference;
					else if (ctx.NeedLowerThreat)
						score *= 0.5f;
					break;
			}

			return Mathf.Clamp(score, 0.1f, 2.0f);
		}

		/// <summary>
		/// Evaluates boundary safety. Returns multiplier that heavily penalizes dangerous moves.
		/// </summary>
		private float EvaluateBoundarySafety(ISumoAction action, EvaluationContext ctx)
		{
			float score = 1.0f;

			// If in danger zone, penalize aggressive actions
			if (ctx.IsInDangerZone)
			{
				switch (action.Type)
				{
					case ActionType.Dash:
						score *= 0.3f; // Risky when near edge
						break;
					case ActionType.SkillBoost:
					case ActionType.SkillStone:
						score *= 0.5f; // Skills can push us out
						break;
					case ActionType.TurnLeft:
					case ActionType.TurnRight:
						score *= 1.3f; // Turns help reposition safely
						break;
				}

				// Extra penalty if VERY close to edge
				if (ctx.DistanceFromCenter > 0.8f)
				{
					score *= 0.5f; // Half all scores when critical
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