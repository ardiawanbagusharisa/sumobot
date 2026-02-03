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

		// Used to get gameplay info and original actions.
		private SumoAPI api;     
		private List<ISumoAction> originalActions = new();
		private List<ISumoAction> pacedActions = new();

		// Store the runtime gameplay data. 
		private List<PacingSegmentInfo> pacingSegmentsInfo = new();
		private PacingSegmentInfo currentSegmentInfo;
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
			currentSegmentInfo = new PacingSegmentInfo();
		}

		public void RegisterAction(ISumoAction action) { 
			currentSegmentInfo.RegisterAction(action);
		}

		public void RegisterCollision(BounceEvent bounce) {
			currentSegmentInfo.RegisterCollision(bounce, api);
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
				currentSegmentInfo.Reset();
				currentSegment = segmentIndex;
			}

			currentSegmentInfo.Sample(api);

			// [Todo] Get constraints and factors from target. 
			//PacingConstraints constraints = ResolveConstraints(segmentIndex);
			//PacingFactors factors = ComputeFactors(accumulator, api, constraints);

			//float threat = WeightedAverage(
			//	new[] { factors.collision, factors.enemySkill, factors.deltaAngle, factors.deltaDistance },
			//	new[] { profile.factorWeights.collision, profile.factorWeights.enemySkill, profile.factorWeights.deltaAngle, profile.factorWeights.deltaDistance });

			//float tempo = WeightedAverage(
			//	new[] { factors.actionIntensity, factors.actionDensity, factors.avgDistanceToEnemy, factors.deltaVelocity },
			//	new[] { profile.factorWeights.actionIntensity, profile.factorWeights.actionDensity, profile.factorWeights.avgDistanceToEnemy, profile.factorWeights.deltaVelocity });

			//float overall = CombineThreatTempo(threat, tempo);
			//float target = profile.EvaluateTarget(elapsed, battleDuration);

			//currentFrame = new PacingFrame(segmentIndex, elapsed, threat, tempo, overall, target, factors);

			//if (history.Count == 0 || history[^1].segmentIndex != segmentIndex) {
			//	history.Add(currentFrame);
			//	runtimeCurve.AddKey(elapsed, overall);
			//} else {
			//	history[^1] = currentFrame;
			//	runtimeCurve.MoveKey(runtimeCurve.length - 1, new Keyframe(elapsed, overall));
			//}

			//return currentFrame;
			return null;
		}

		// [Todo] Edit later. 
		private void FinalizeSegment(float elapsed) {
			if (pacingsHistory.Count == 0 && currentSegment < 0) return;
		}

		// [Todo]
		// Resolve Constraints, Calculate factors and aspects
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

	public class Pacing {
		public int segmentIndex = -1;
		public PacingAspects pacingAspects = new PacingAspects();
		public PacingFactors pacingFactors = new PacingFactors();
		public PacingConstraints pacingConstraints = new PacingConstraints();
		public PacingSegmentInfo pacingSegmentInfo = new PacingSegmentInfo();
	}

	public class PacingAspects
	{
		public float threat = 0f;
		public float tempo = 0f;
		public float weightThreat = 1f;
		public float weightTempo = 1f;
		public float overallPacing = 0f;

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
		public float enemyAngle = 0f;
		public float distanceToEdge = 0f;
		public float weightCollision = 1f;
		public float weightEnemySkill = 1f;
		public float weightEnemyAngle = 1f;
		public float weightDistanceToEdge = 1f;
		#endregion
		#region Tempo 
		public float actionIntensity = 0f;
		public float actionDensity = 0f;
		public float distanceToEnemy = 0f;
		public float deltaVelocity = 0f;
		public float weightActionIntensity = 1f;
		public float weightActionDensity = 1f;
		public float weightDistanceToEnemy = 1f;
		public float weightDeltaVelocity = 1f;
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
		#endregion
		#region Tempo 
		[Header("Tempo")]
		public MinMax totalAction = new(0f, 250f);
		public MinMax actionVariation = new(0f, 5f);
		public MinMax avgDistToEnemy = new(0f, 8f);
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

				totalAction = totalAction,
				actionVariation = actionVariation,
				avgDistToEnemy = avgDistToEnemy,
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
