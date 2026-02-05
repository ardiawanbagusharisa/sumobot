using SumoBot;
using SumoCore;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace PacingFramework
{
	// [Todo] Edit later. Handle the gameplay data using Pacing.pacingSegmentInfo.
	public class PacingController: MonoBehaviour {
		// [Todo] Change this with scriptableobject later. 
		public List<Pacing> pacingsTarget = new List<Pacing>();	
		public List<Pacing> pacingsHistory = new List<Pacing>();
		public AnimationCurve pacingsTargetCurve;
		public AnimationCurve pacingsHistoryCurve;
		public AnimationCurve threatHistoryCurve;
		public AnimationCurve tempoHistoryCurve;

		// Used to get gameplay info and original actions.
		private SumoAPI api;     
		private List<ISumoAction> originalActions = new();
		private List<ISumoAction> pacedActions = new();

		// Store the runtime gameplay data. 
		private List<PacingSegmentInfo> pacingSegmentsInfo = new();
		//private PacingSegmentInfo currentSegmentInfo;
		private Pacing currentPacing = new();
		private int currentSegment = -1;
		private float battleDuration;

		public float segmentDuration = 0.1f;
		
		public void Init(SumoAPI api) { 
			pacingsTarget.Clear();
			pacingsHistory.Clear();
			pacingsTargetCurve = new AnimationCurve();
			pacingsHistoryCurve = new AnimationCurve();
			this.api = api;
			battleDuration = api.BattleInfo.Duration;
			currentSegment = -1;
			originalActions.Clear();
			pacedActions.Clear();
			pacingSegmentsInfo.Clear();
			//currentSegmentInfo = new PacingSegmentInfo();
			currentPacing = new Pacing();
			threatHistoryCurve = new AnimationCurve();
			tempoHistoryCurve = new AnimationCurve();
		}

		public void RegisterAction(ISumoAction action) { 
			//currentSegmentInfo.RegisterAction(action);
			currentPacing.pacingSegmentInfo.RegisterAction(action);
		}

		public void RegisterCollision(BounceEvent bounce) {
			//currentSegmentInfo.RegisterCollision(bounce, api);
			currentPacing.pacingSegmentInfo.RegisterCollision(bounce, api);
		}

		public Pacing Tick() {
			// Check if pacings target is null
			if (pacingsTarget == null || api == null) {
				Debug.LogError("Pacing target or api is not found");
			}

			float elapsed = Mathf.Clamp(api.BattleInfo.Duration - api.BattleInfo.TimeLeft, 0f, battleDuration);
			int segmentIndex = Mathf.FloorToInt(elapsed / segmentDuration);

			if (segmentIndex != currentSegment) {
				FinalizeSegment(elapsed);
				currentPacing.pacingSegmentInfo.Reset();
				currentSegment = segmentIndex;
			}

			currentPacing.pacingSegmentInfo.Sample(api);

			// [Todo] Get constraints and factors from target. 
			// ResolveConstraints(segmentIndex); // Handle constraints for different bots. 
			PacingFactors factors = ComputeFactors(currentPacing.pacingSegmentInfo, api, pacingsTarget.First().pacingConstraints);

			float threat = WeightedAverage(
				new[] { factors.collision, factors.enemySkill, factors.deltaAngle, factors.deltaDistance },
				new[] { factors.weightCollision, factors.weightEnemySkill, factors.weightEnemyAngle, factors.deltaDistance });

			float tempo = WeightedAverage(
				new[] { factors.actionIntensity, factors.actionDensity, factors.avgDistanceToEnemy, factors.deltaVelocity },
				new[] { factors.weightActionIntensity, factors.actionDensity, factors.avgDistanceToEnemy, factors.deltaVelocity });

			float overallPacing = ComputeOverallPacing(threat, tempo);

			// [Todo] Evaluate the actual pacing with the target. 
			//float target = EvaluateTarget(elapsed, battleDuration);

			// prepare currentpacing values
			/*
			public int segmentIndex = -1; v
			public PacingAspects pacingAspects = new PacingAspects(); v
			public PacingFactors pacingFactors = new PacingFactors(); v
			public PacingConstraints pacingConstraints = new PacingConstraints(); 
			public PacingSegmentInfo pacingSegmentInfo = new PacingSegmentInfo();
			*/
			PacingAspects currentPacingAspects = new PacingAspects(
				threat, tempo, pacingsTarget.First().pacingAspects.weightThreat, pacingsTarget.First().pacingAspects.weightTempo, overallPacing);

			// Add pacingConstraints and pacingSegmentInfo
			currentPacing.pacingConstraints = pacingsTarget.First().pacingConstraints.Clone();
			// pacingSegmentInfo alread added. 
			//currentPacing.pacingSegmentInfo

			currentPacing = new Pacing(segmentIndex, currentPacingAspects, factors, currentPacing.pacingConstraints, currentPacing.pacingSegmentInfo);
			
			if (history.Count == 0 || history[^1].segmentIndex != segmentIndex) {
				history.Add(currentPacing);
				pacingsHistoryCurve.AddKey(elapsed, overallPacing);
				threatHistoryCurve.AddKey(elapsed, threat);
				tempoHistoryCurve.AddKey(elapsed, tempo);

			} else {
				history[^1] = currentFrame;
				pacingsHistoryCurve.MoveKey(pacingsHistoryCurve.length - 1, new Keyframe(elapsed, overallPacing));
				threatHistoryCurve.MoveKey(threatHistoryCurve.length - 1, new Keyframe(elapsed, threat));
				tempoHistoryCurve.MoveKey(tempoHistoryCurve.length - 1, new Keyframe(elapsed, tempo));
			}

			return Pacing;
		}

		// [Todo] Edit later. 
		private void FinalizeSegment(float elapsed) {
			if (pacingsHistory.Count == 0 && currentSegment < 0) return;
		}

		// [Todo]
		// Resolve Constraints, Calculate factors and aspects
		private PacingConstraints ResolveConstraints(int segmentIndex) {
			// Resolve different constraints for different bots here. 
			return null;
		}

		private static PacingConstraints BlendConstraints(PacingConstraints global, PacingConstraints local, float weight) {
			// Handle local and global constraints blending. 
			return null;
		}

		private static MinMax Lerp(MinMax a, MinMax b, float t) {
			return new MinMax {
				min = Mathf.Lerp(a.min, b.min, t),
				max = Mathf.Lerp(a.max, b.max, t)
			};
		}

		private static float ComputeOverallPacing(float threat, float tempo) {
			float wThreat = Mathf.Max(0.0001f, Mathf.Abs(threat));
			float wTempo = Mathf.Max(0.0001f, Mathf.Abs(tempo));
			return (threat * wThreat + tempo * wTempo) / (wThreat + wTempo);
		}

		private static float WeightedAverage(IReadOnlyList<float> values, IReadOnlyList<float> weights) {
			float total = 0f;
			float weightSum = 0f;
			for (int i = 0; i < values.Count; i++) {
				float w = Mathf.Max(0f, weights[i]);
				weightSum += w;
				total += values[i] * w;
			}
			return weightSum > 0f ? total / weightSum : 0f;
		}

		// [Todo] These factor calculations are generated by AI. Some are still wrong and need edit. 
		private static PacingFactors ComputeFactors(PacingSegmentInfo acc, SumoAPI api, PacingConstraints constraints) {
			float collisionRatio = acc.TotalCollisions > 0 ? (float)acc.StruckCollisions / acc.TotalCollisions : 0f;
			float collision = Normalize(collisionRatio, constraints.struckCollision);

			// [Todo] Handle this manually later. 
			float skillState = acc.Samples > 0 ? acc.EnemySkillSum / acc.Samples : (api.EnemyRobot.Skill.IsActive ? 1f : api.EnemyRobot.Skill.IsSkillOnCooldown ? 0f : 0.5f);
			float enemySkill = Normalize(skillState, constraints.enemySkill);

			float avgAgentAngle = acc.Samples > 0 ? acc.AgentAngleSum / acc.Samples : Mathf.Abs(api.Angle());
			float avgEnemyAngle = acc.Samples > 0 ? acc.EnemyAngleSum / acc.Samples : Mathf.Abs(api.Angle(oriPos: api.EnemyRobot.Position, oriRot: api.EnemyRobot.Rotation, targetPos: api.MyRobot.Position));
			float deltaAngle = Normalize(FacingDelta(avgAgentAngle, avgEnemyAngle), new MinMax(0f, 1f));

			float avgAgentEdge = acc.Samples > 0 ? acc.AgentEdgeSum / acc.Samples : DistanceToEdge(api.MyRobot.Position, api.BattleInfo);
			float avgEnemyEdge = acc.Samples > 0 ? acc.EnemyEdgeSum / acc.Samples : DistanceToEdge(api.EnemyRobot.Position, api.BattleInfo);
			float edgeDelta = avgEnemyEdge - avgAgentEdge; // positive when we are closer to edge
			float edgeRange = Mathf.Max(constraints.agentDistanceEdge.min, constraints.enemyDistanceEdge.min);
			edgeRange = Mathf.Max(edgeRange, 0.1f);
			float deltaDistance = Normalize(edgeDelta, new MinMax(-edgeRange, edgeRange));

			float actionIntensity = Normalize(acc.ActionCount, constraints.totalAction);

			int possibleActions = Enum.GetValues(typeof(ActionType)).Length;
			float actionDensityRaw = possibleActions > 0 ? (float)acc.UniqueActionTypes.Count / possibleActions : 0f;
			float actionDensity = Normalize(actionDensityRaw * constraints.actionVariation.max, constraints.actionVariation);

			float avgDist = acc.Samples > 0 ? acc.DistanceToEnemySum / acc.Samples : Vector2.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
			float avgDistanceToEnemy = Normalize(avgDist, constraints.avgDistanceToEnemy);

			float avgAgentVel = acc.Samples > 0 ? acc.AgentVelocitySum / acc.Samples : api.MyRobot.LinearVelocity.magnitude;
			float avgEnemyVel = acc.Samples > 0 ? acc.EnemyVelocitySum / acc.Samples : api.EnemyRobot.LinearVelocity.magnitude;
			float velocityDelta = avgAgentVel - avgEnemyVel;
			float deltaVelocity = Normalize(velocityDelta, new MinMax(-constraints.enemyVelocity.max, constraints.agentVelocity.max));

			return new PacingFactors(
				collision,
				enemySkill,
				deltaAngle,
				deltaDistance,
				actionIntensity,
				actionDensity,
				avgDistanceToEnemy,
				deltaVelocity);
		}

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

		//public float EvaluateTarget(float elapsed, float battleDuration) {
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
		//}

	}


	[Serializable]
	public struct MinMax
	{
		public float min;
		public float max;

		public MinMax(float min, float max) {
			this.min = min;
			this.max = max;
		}
	}

	public enum PacingAspectType {
		Threat, 
		Tempo
	};

	// [Todo] This stucture seems unecessary. We could also implement the fields directly into the PacingController. 
	public class Pacing {
		public int segmentIndex = -1;
		public PacingAspects pacingAspects = new PacingAspects();
		public PacingFactors pacingFactors = new PacingFactors();
		public PacingConstraints pacingConstraints = new PacingConstraints();
		public PacingSegmentInfo pacingSegmentInfo = new PacingSegmentInfo();

		public Pacing() { }
        public Pacing(
			int segmentIndex, 
			PacingAspects pacingAspects, 
			PacingFactors pacingFactors, 
			PacingConstraints pacingConstraints, 
			PacingSegmentInfo pacingSegmentInfo) {
            this.segmentIndex = segmentIndex;
            this.pacingAspects = pacingAspects;
            this.pacingFactors = pacingFactors;
            this.pacingConstraints = pacingConstraints;
            this.pacingSegmentInfo = pacingSegmentInfo;
        }
    }

	public class PacingAspects
	{
		public float threat = 0f;
		public float tempo = 0f;
		public float weightThreat = 1f;
		public float weightTempo = 1f;
		public float overallPacing = 0f;

		public PacingAspects() { }

		public PacingAspects(float threat, float tempo, float weightThreat, float weightTempo, float overallPacing) { 
			this.threat = threat;
			this.tempo = tempo;
			this.weightThreat = weightThreat;
			this.weightTempo = weightTempo;
			this.overallPacing = overallPacing;
		}

		public PacingAspects Clone() {
			return new PacingAspects();
		}

		public static float Normalize(float value, float min, float max) {
			return max - min != 0 ? (value - min) / (max - min) : 0f;
		}

	}

	public class PacingFactors {
		#region Threat 
		public float collision = 0f;
		public float enemySkill = 0f;
		//public float enemyAngle = 0f;
		public float deltaAngle = 0f;
		//public float distanceToEdge = 0f;
		public float deltaDistance = 0f;
		public float weightCollision = 1f;
		public float weightEnemySkill = 1f;
		public float weightEnemyAngle = 1f;
		public float weightDistanceToEdge = 1f;
		#endregion
		#region Tempo 
		public float actionIntensity = 0f;
		public float actionDensity = 0f;
		//public float distanceToEnemy = 0f;
		public float avgDistanceToEnemy = 0f;
		public float deltaVelocity = 0f;
		public float weightActionIntensity = 1f;
		public float weightActionDensity = 1f;
		public float weightDistanceToEnemy = 1f;
		public float weightDeltaVelocity = 1f;
        

        public PacingFactors(float collision, float enemySkill, float deltaAngle, float deltaDistance, float actionIntensity, float actionDensity, float avgDistanceToEnemy, float deltaVelocity) {
            this.collision = collision;
            this.enemySkill = enemySkill;
            this.deltaAngle = deltaAngle;
            this.deltaDistance = deltaDistance;
            this.actionIntensity = actionIntensity;
            this.actionDensity = actionDensity;
            this.avgDistanceToEnemy = avgDistanceToEnemy;
            this.deltaVelocity = deltaVelocity;
        }

        public PacingFactors() {
        }
        #endregion

        public static PacingFactors Default() { 
			return new PacingFactors();
		}

		public float GetWeights(PacingAspectType type) {
			float sum = -1f;
			if (type == PacingAspectType.Threat) {
				sum = weightCollision + weightEnemySkill + weightEnemyAngle + weightDistanceToEdge;
			} else {
				sum = weightActionIntensity + weightActionDensity + weightDistanceToEnemy + weightDeltaVelocity;
			}
			
			return sum;
		}

		public float GetTotalWeights() {
			return GetWeights(PacingAspectType.Threat) + GetWeights(PacingAspectType.Tempo);
		}
	}

	[Serializable]
	public class PacingConstraints {
		#region Threat 
		[Header("Threat")]
		public MinMax struckCollision = new(0f, 1f);
		public MinMax totalCollision = new(0f, 1f);
		public MinMax enemyAngle = new(0f, 180f);
		public MinMax agentAngle = new(0f, 180f);
		public MinMax enemyDistanceEdge = new(0f, 6f);
		public MinMax agentDistanceEdge = new(0f, 6f);
		public MinMax enemySkill = new (0f, 1f);
		#endregion
		#region Tempo 
		[Header("Tempo")]
		public MinMax totalAction = new(0f, 250f);
		public MinMax actionVariation = new(0f, 5f);
		public MinMax avgDistanceToEnemy = new(0f, 8f);
		public MinMax agentVelocity = new(0f, 8f);
		public MinMax enemyVelocity = new(0f, 8f);
		#endregion

		public static PacingConstraints Default() {
			return new PacingConstraints();
		}
		public PacingConstraints Clone() {
			return new PacingConstraints {
				struckCollision = struckCollision,
				totalCollision = totalCollision,
				enemyAngle = enemyAngle,
				agentAngle = agentAngle,
				enemyDistanceEdge = enemyDistanceEdge,
				agentDistanceEdge = agentDistanceEdge,
				enemySkill = enemySkill,

				totalAction = totalAction,
				actionVariation = actionVariation,
				avgDistanceToEnemy = avgDistanceToEnemy,
				agentVelocity = agentVelocity,
				enemyVelocity = enemyVelocity
			};

		}
	}

	public class PacingSegmentInfo {
		#region Threat 
		public int TotalCollisions { get; private set; }
		public int StruckCollisions { get; private set; }
		public float AgentAngleSum { get; private set; }
		public float EnemyAngleSum { get; private set; }
		public float AgentEdgeSum { get; private set; }
		public float EnemyEdgeSum { get; private set; }
		#endregion

		#region Tempo 
		public int ActionCount { get; private set; }
		public HashSet<ActionType> UniqueActionTypes { get; } = new();
		public float DistanceToEnemySum { get; private set; }
		public float AgentVelocitySum { get; private set; }
		public float EnemyVelocitySum { get; private set; }
		public float EnemySkillSum { get; private set; }
		#endregion

		public int Samples { get; private set; }

		public void Reset() {
			TotalCollisions = 0;
			StruckCollisions = 0;
			AgentAngleSum = 0f;
			EnemyAngleSum = 0f;
			AgentEdgeSum = 0f;
			EnemyEdgeSum = 0f;
			
			ActionCount = 0;
			UniqueActionTypes.Clear();
			DistanceToEnemySum = 0f;
			AgentVelocitySum = 0f;
			EnemyVelocitySum = 0f;
			EnemySkillSum = 0f;
			
			Samples = 0;
		}

		public void RegisterAction(ISumoAction action) {
			if (action == null)
				return;

			ActionCount++;
			UniqueActionTypes.Add(action.Type);
		}

		public void RegisterCollision(BounceEvent bounce, SumoAPI api) {
			if (bounce == null)
				return;

			TotalCollisions++;

			if (api != null && bounce.Actor == api.MyRobot.Side) { 
				StruckCollisions++;
			}
		}

		// [Todo] Change sum with average. 
		public void Sample(SumoAPI api) {
			if (api == null)
				return;

			Samples++;
			SumoBotAPI myRobot = api.MyRobot;
			SumoBotAPI enemyRobot = api.EnemyRobot;
			
			DistanceToEnemySum += Vector2.Distance(myRobot.Position, enemyRobot.Position);
			AgentEdgeSum += DistanceToEdge(myRobot.Position, api.BattleInfo);
			EnemyEdgeSum += DistanceToEdge(enemyRobot.Position, api.BattleInfo);
			AgentAngleSum += Mathf.Abs(api.Angle());
			EnemyAngleSum += Mathf.Abs(api.Angle(
				oriPos: enemyRobot.Position,
				oriRot: enemyRobot.Rotation,
				targetPos: myRobot.Position));
			AgentVelocitySum += myRobot.LinearVelocity.magnitude;
			EnemyVelocitySum += enemyRobot.LinearVelocity.magnitude;
			EnemySkillSum += enemyRobot.Skill.IsActive ? 1f : enemyRobot.Skill.IsSkillOnCooldown ? 0f : 0.5f;
		}

		private static float DistanceToEdge(Vector2 position, BattleInfoAPI battleInfo) {
			float distToCenter = Vector2.Distance(position, battleInfo.ArenaPosition);
			return Mathf.Max(0f, battleInfo.ArenaRadius - distToCenter);
		}
	}
}
