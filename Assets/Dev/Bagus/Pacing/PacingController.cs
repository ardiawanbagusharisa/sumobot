using SumoBot;
using SumoCore;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using static PacingFramework.SegmentPacing.FactorType;

namespace PacingFramework
{
	/// <summary>
	/// Collection of pacing information for each segment, including: runtime gameplay data and pacing details. 
	/// </summary>
	public struct GamePacingHistory { 
		public List<SegmentGameplayData> GameplayData;
		public List<SegmentPacing> Pacing;
		
		// For visualizations.  
		public AnimationCurve ThreatCurve;
		public AnimationCurve TempoCurve;
	}

	/// <summary>
	/// The pacing and constraints targets used for evaluation designed by designer. 
	/// </summary>
	public struct GamePacingTarget {
		// Constraints for all segments. Local constraints are inside Pacing and are optional. 
		public Constraints GlobalConstraints;
		public List<SegmentPacing> Pacing;

		// For visualizations.  
		public AnimationCurve ThreatCurve;
		public AnimationCurve TempoCurve;
	}

	/// <summary>
	/// The main script that manages the pacing integration, including: 
	/// 1. Collects gameplay data and convert to pacing variables. 
	/// 2. Calculates pacing aspects (threat and tempo) using factor formulas.
	/// [Todo] 3. Evaluates the pacing against the target. 
	/// [Todo] 4. Provides the pacing information for further purposes, including pacing curve visualization.
	/// </summary>
	public class PacingController: MonoBehaviour {
		// Target pacing and history pacing. Gameplay data are stored in PacingHistory.GameplayData. 
		#region Pacing Target and History
		public GamePacingHistory PacingHistory;
		public GamePacingTarget PacingTarget;
		#endregion

		// Runtime configurations. 
		#region Runtime Configs
		public float SegmentDuration = 1f;
		private float battleDuration = 60f;
		private SumoAPI api;
		private List<ISumoAction> originalActions = new();
		private List<ISumoAction> pacedActions = new();
		private int currentSegmentIndex = -1;
		private SegmentGameplayData currentGameplayData; 
		private SegmentPacing currentSegmentPacing;
		#endregion

		public void Init(SumoAPI api) {
			// [Todo 1] Initialize all the variables. Also duplicate the constraints from target to currentSegmentPacing and PacingTarget for later use.

			// [Delete]
			//PacingsTarget.Clear();
			//PacingsHistory.Clear();
			//TargetThreats = new();
			//TargetTempos = new();
			//HistoryThreats = new();
			//HistoryTempos = new();
			//SegmentDuration = 1f;
			//currentSegmentIndex = -1;
			//battleDuration = api.BattleInfo.Duration;
			//this.api = api;
			//originalActions.Clear();
			//pacedActions.Clear();
			//pacingSegmentsInfo.Clear();
		}

		// [Todo 2] Implement all Register functions from currentGameplayData. 

		// [Todo 3] Check this function. We need to finalize the segment when the segment changes.
		private void FinalizeSegment(float elapsed, int segmentIndex) {
			if (PacingHistory.Pacing.Count == 0 && currentSegmentIndex < 0)
				return;

			currentSegmentPacing.Reset();
			currentSegmentIndex = segmentIndex;

			// [Todo 5] After currentSegmentData is populated on the Tick(), sample pacing variables from gameplay data into currentSegmentPacing.

			// [Todo 6] In that currentSegmentPacing, compute pacing factors and aspects from the variables; use the constraints from target for the normalization. 
			// then add the data into PacingHistory.
			// For further experiment, we can also resolve the specific constraints for specific bot. 

			// [Todo 7] Update the pacing curves for visualization. 

			// [Todo 8] Evaluate the actual pacing with the target. 
			// EvaluateTarget(segment);
		}

		// Call this function every game tick to update gameplay data.
		public void Tick() {
			float elapsed = Mathf.Clamp(api.BattleInfo.Duration - api.BattleInfo.TimeLeft, 0f, battleDuration);
			int segmentIndex = Mathf.FloorToInt(elapsed / SegmentDuration);

			if (segmentIndex != currentSegmentIndex) 
				FinalizeSegment(elapsed, segmentIndex);

			// [Todo 4] Populate currentSegmentData from api using register functions.
		}

		// [Todo 9]
		// Resolve the constraints according the segment index, then solve the blending between global and local constraints. 
		//private SegmentPacing.Constraints ResolveConstraints(int segmentIndex) {
		//	// [Todo] Resolve different constraints for different bots here. 
		//	return null;
		//}
		//private static PacingConstraints BlendConstraints(PacingConstraints global, PacingConstraints local, float weight) {
		//	// Handle local and global constraints blending. 
		//	return null;
		//}

		private static MinMax Lerp(MinMax a, MinMax b, float t) {
			return new MinMax {
				min = Mathf.Lerp(a.min, b.min, t),
				max = Mathf.Lerp(a.max, b.max, t)
			};
		}

		// ================================================

		private static float Normalize(float value, MinMax range) {
			float denom = range.max - range.min;
			if (Mathf.Approximately(denom, 0f)) return 0.5f;
			float normalized = (value - range.min) / denom;
			return Mathf.Clamp01(normalized);
		}

		private static float DistanceToEdge(Vector2 position, BattleInfoAPI battleInfo) {
			float distToCenter = Vector2.Distance(position, battleInfo.ArenaPosition);
			return Mathf.Max(0f, battleInfo.ArenaRadius - distToCenter);
		}

		private static float FacingDelta(float myAverageAngle, float enemyAverageAngle) {
			float enemyFacingMe = Mathf.InverseLerp(180f, 0f, enemyAverageAngle);
			float iAmBehindEnemy = 1f - Mathf.InverseLerp(180f, 0f, myAverageAngle);
			return Mathf.Clamp01(enemyFacingMe * iAmBehindEnemy);
		}

		public float EvaluateTarget(float elapsed, float battleDuration) {
			//	float duration = referenceDuration > 0f ? referenceDuration : Mathf.Max(1f, battleDuration);
			//	float t = Mathf.Clamp01(elapsed / duration);

			//	return pattern switch {
			//		PacingPattern.ConstantLow => constantLow,
			//		PacingPattern.ConstantBalanced => constantBalanced,
			//		PacingPattern.ConstantHigh => constantHigh,
			//		PacingPattern.LinearIncrease => t,
			//		PacingPattern.LinearDecrease => 1f - t,
			//		PacingPattern.ExponentialIncrease => Mathf.Pow(t, exponentialK),
			//		PacingPattern.ExponentialDecrease => 1f - Mathf.Pow(t, exponentialK),
			//		_ => Mathf.Clamp01(customCurve.Evaluate(t)),
			//	};
			return 0;
		}
	}

}
