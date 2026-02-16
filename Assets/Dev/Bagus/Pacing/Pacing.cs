using SumoBot;
using SumoCore;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>
/// This script implements a concept of pacing, a measure of the intensity and flow of the gameplay, which can be used to adapt the bot's strategy and behavior in real-time.
/// 
/// Structure: 
/// - Pacing: the flow of the gameplay in a specific time segment, which includes the pacing aspects, factors, constraints and segment info. 
/// -- PacingAspects: the high level aspects of pacing, which are threat and tempo.
/// --- PacingFactors: the specific factors that contribute to the pacing aspects. For example, the effectiveness of our bot collision behaviour, skill availability contribute to threat, while action intensity and distance to enemy contribute to tempo.
/// ---- PacingVariables: the parameters that define how the pacing factors are calculated and weighted, which can be set globally or locally for each segment, and can also be blended together. 
/// ---- PacingConstraints: the expected range of the pacing factors, which can be used to normalize the factors and evaluate the pacing. These constraints can be set globally or locally for each segment, and can also be blended together. PacingConstraints are similar to PacingParameters. 
/// 
/// Other terms: 
/// - PacingSegmentInfo: the raw gameplay data collected in a specific time segment, which is used to calculate the pacing factors.
/// - PacingController: the main class that manages the pacing system, which collects gameplay data, calculates pacing factors and aspects, evaluates the pacing against the target, and provides the pacing information to other parts of the bot.
/// - PacingPattern: the predefined patterns of pacing target over time, which can be used to create different pacing curves for different strategies and playstyles. We can also use custom curves for more specific pacing targets.
/// - PacingHistory: the historical record of pacing information over time, which can be used for analysis and debugging, as well as for more advanced adaptive strategies that consider the pacing history.
/// - PacingTarget: the desired pacing information at a specific time, which can be defined by the pacing pattern and constraints, and can be used to evaluate the actual pacing and guide the bot's behavior.
/// - PacingEvaluation: the process of comparing the actual pacing with the target pacing, which can be used to determine how well the bot is performing and how it should adapt its strategy and behavior. This can involve calculating the difference between the actual and target pacing, as well as considering the pacing history and trends.
/// </summary>

namespace PacingFramework
{
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

	public enum PacingAspectType
	{
		Threat,
		Tempo
	};

	public enum CollisionType
	{
		Hit,
		Struck,
		Tie
	};

	public enum SkillState
	{
		Active,
		Available,
		Cooldown
	};

	/// <summary>
	/// PacingVariables is a structure that encapsulates various unnormalized parameters related to the pacing of agent and enemy interactions within a simulation environment. 
	/// This structure is designed to hold data obtained from SegmentData, which later will be normalized before pacing calculation. 
	/// </summary>
	public struct PacingVariables
	{
		#region Threat 
		public int CollisionsCount;
		public int HitCollisionsCount;
		public float AverageAgentAngle;
		public float AverageEnemyAngle;
		public float AverageAgentEdgeDistance;
		public float AverageEnemyEdgeDistance;
		public float AverageAgentSkillState;
		#endregion
		#region Tempo 
		public int ActionCount;
		public float AverageActionVariationRatio;
		public float AverageBotDistance;
		public float AverageAgentVelocity;
		public float AverageEnemyVelocity;
		#endregion
	}

	/// <summary>
	/// Constraint is a set of lower and upper limits for variable's normalizations. Constraint fields are implemented in PacingConstraints, 
	/// where each bot will have specific PacingConstraints. 
	/// </summary>
	public class Constraint
	{
		public float min;
		public float max;

		public Constraint(float min, float max) {
			this.min = min;
			this.max = max;
		}

		public float Range => max - min;

		public bool IsInRange(float value) {
			return value >= min && value <= max;
		}

		// The normalization of PacingVariables depend on each bot's constraints. 
		public float Normalize(float value) {
			return Mathf.Clamp01((value - min) / Range);
		}

		// These two functions are used to calculate the penalty for potential further purposes. 
		// For example: Score -= CollisionConstraint.NormalizedViolation(stats.CollisionCount);
		public float DistanceFromRange(float value) {
			if (value < min)
				return min - value;
			if (value > max)
				return value - max;
			return 0f;
		}

		public float NormalizedViolation(float value) {
			if (IsInRange(value))
				return 0f;
			if (Range == 0f)
				return 0f;
			return DistanceFromRange(value) / Range;
		}
	}

	/// <summary>
	/// Gameplay or runtime data collected in a segment (few seconds), used to calculate pacing factors. 
	/// Please note that the time segment range is determined by the PacingController. 
	/// </summary>
	public class SegmentData
	{
		#region Threat related data 
		public List<CollisionType> Collisions = new();      // Any type of collisions. 
		public Dictionary<CollisionType, int> CollisionsCount { get { return GetCollisionsCount(); } private set { } }
		public List<float> AgentAngles = new();             // Angles of our bot towards enemy.
		public List<float> EnemyAngles = new();             // Angles of enemy towards our bot. 
		public List<float> AgentEdgeDistances = new();      // Distances between our bot and edge of arena. 
		public List<float> EnemyEdgeDistances = new();      // Distances between enemy and edge of arena.
		public List<SkillState> AgentSkillStates = new();   // Skill state of Agent.
		#endregion

		#region Tempo related data 
		// [Edit] Should this be iSumoAction instead?  
		public List<ActionType> Actions = new();        // Actions performed by our bot. 
		public Dictionary<ActionType, int> ActionsCount { get { return GetActionsCount(); } private set { } }
		public List<float> BotDistances = new();        // Distances between our bot and enemy. 
		public List<float> AgentVelocities = new();     // Velocities of our bot. 
		public List<float> EnemyVelocities = new();     // Velocities of enemy. 
		#endregion

		public Dictionary<CollisionType, int> GetCollisionsCount() {
			Dictionary<CollisionType, int> collisionsCount = new();

			if (Collisions == null || Collisions.Count == 0)
				return collisionsCount;

			foreach (CollisionType type in Enum.GetValues(typeof(CollisionType))) {
				int count = Collisions.Count(c => c == type);
				collisionsCount[type] = count;
			}

			return collisionsCount;
		}

		public Dictionary<ActionType, int> GetActionsCount() {
			Dictionary<ActionType, int> actionsCount = new();

			if (Actions == null || Actions.Count == 0)
				return actionsCount;

			foreach (ActionType type in Enum.GetValues(typeof(ActionType))) {
				int count = Actions.Count(a => a == type);
			}

			return actionsCount;
		}

		public void Reset() {
			Collisions.Clear();
			CollisionsCount.Clear();
			AgentAngles.Clear();
			EnemyAngles.Clear();
			AgentEdgeDistances.Clear();
			EnemyEdgeDistances.Clear();
			Actions.Clear();
			ActionsCount.Clear();
			BotDistances.Clear();
			AgentVelocities.Clear();
			EnemyVelocities.Clear();
			AgentSkillStates.Clear();
		}

		private static float DistanceToEdge(Vector2 position, BattleInfoAPI battleInfo) {
			float distToCenter = Vector2.Distance(position, battleInfo.ArenaPosition);
			return Mathf.Max(0f, battleInfo.ArenaRadius - distToCenter);
		}

		// Populate the segment gameplay info by registering data from api. 
		#region Register data methods
		public void RegisterAction(ISumoAction action) {
			if (action == null)
				return;
			Actions.Add(action.Type);
		}

		public void RegisterActions(SumoAPI api) {
			if (api == null)
				return;

			foreach (ActionType action in api.MyRobot.ActiveActions.Keys.ToList()) {
				Actions.Add(action);
			}
		}

		public void RegisterCollision(BounceEvent bounce, SumoAPI api) {
			if (bounce == null || api == null)
				return;

			CollisionType collisionType;
			if (bounce.Actor == api.MyRobot.Side) {
				collisionType = CollisionType.Hit;
			} else if (bounce.Actor == api.EnemyRobot.Side) {
				collisionType = CollisionType.Struck;
			} else {
				collisionType = CollisionType.Tie;
			}
			Collisions.Add(collisionType);
		}

		public void RegisterAngles(SumoAPI api) {
			if (api == null)
				return;

			// [Edit] Remove Mathf.Abs later, and handle the direction in factors?
			AgentAngles.Add(Mathf.Abs(api.Angle()));
			EnemyAngles.Add(Mathf.Abs(api.Angle(oriPos: api.EnemyRobot.Position, oriRot: api.EnemyRobot.Rotation, targetPos: api.MyRobot.Position)));
		}

		public void RegisterDistances(SumoAPI api) {
			if (api == null)
				return;

			AgentEdgeDistances.Add(Mathf.Abs(DistanceToEdge(api.MyRobot.Position, api.BattleInfo)));
			EnemyEdgeDistances.Add(Mathf.Abs(DistanceToEdge(api.EnemyRobot.Position, api.BattleInfo)));

			BotDistances.Add(Vector2.Distance(api.MyRobot.Position, api.EnemyRobot.Position));
		}

		public void RegisterVelocities(SumoAPI api) {
			if (api == null)
				return;
			AgentVelocities.Add(api.MyRobot.LinearVelocity.magnitude);
			EnemyVelocities.Add(api.EnemyRobot.LinearVelocity.magnitude);
		}

		public void RegisterSkillStates(SumoAPI api) {
			if (api == null)
				return;

			SkillState state = SkillState.Available;
			if (api.MyRobot.Skill.IsActive == true) {
				state = SkillState.Active;
			} else if (api.MyRobot.Skill.IsSkillOnCooldown == true) {
				state = SkillState.Cooldown;
			}

			AgentSkillStates.Add(state);
		}
		#endregion
	}

	/// <summary>
	/// PacingConstrains is a collection of constraints that define acceptable ranges for various pacing metrics. 
	/// For each bot normalization, we can easily create instances of it. 
	/// </summary> 
	public class PacingConstraints
	{
		#region Threat
		[Header("Threat")]
		public Constraint CollisionsCount;
		public Constraint HitCollisionCount;
		public Constraint AverageAgentAngle;
		public Constraint AverageEnemyAngle;
		public Constraint AverageAgentEdgeDistance;
		public Constraint AverageEnemyEdgeDistance;
		public Constraint AverageAgentSkillState;
		#endregion

		#region Tempo
		[Header("Tempo")]
		public Constraint ActionCount;
		public Constraint AverageActionVariationRatio;
		public Constraint AverageBotDistance;
		public Constraint AverageAgentVelocity;
		public Constraint AverageEnemyVelocity;
		#endregion

		public PacingConstraints(
			Constraint collisionsCount, Constraint hitCollisionCount,
			Constraint averageAgentAngle, Constraint averageEnemyAngle,
			Constraint averageAgentEdgeDistance, Constraint averageEnemyEdgeDistance,
			Constraint averageAgentSkillState, Constraint actionCount,
			Constraint averageActionVariationRatio, Constraint averageBotDistance,
			Constraint averageAgentVelocity, Constraint averageEnemyVelocity) {
			// Threat
			CollisionsCount = collisionsCount;
			HitCollisionCount = hitCollisionCount;
			AverageAgentAngle = averageAgentAngle;
			AverageEnemyAngle = averageEnemyAngle;
			AverageAgentEdgeDistance = averageAgentEdgeDistance;
			AverageEnemyEdgeDistance = averageEnemyEdgeDistance;
			AverageAgentSkillState = averageAgentSkillState;
			// Tempo
			ActionCount = actionCount;
			AverageActionVariationRatio = averageActionVariationRatio;
			AverageBotDistance = averageBotDistance;
			AverageAgentVelocity = averageAgentVelocity;
			AverageEnemyVelocity = averageEnemyVelocity;
		}

		public PacingConstraints() : this(
			// Threat
			new Constraint(0f, 1f), new Constraint(0f, 1f),
			new Constraint(0f, 180f), new Constraint(0f, 180f),
			new Constraint(0f, 6f), new Constraint(0f, 6f),
			new Constraint(0f, 1f),
			//Tempo
			new Constraint(0f, 250f), new Constraint(0f, 5f),
			new Constraint(0f, 8f), new Constraint(0f, 8f),
			new Constraint(0f, 8f)
		) { }

		public PacingConstraints(PacingConstraints other) {
			// Threat 
			CollisionsCount = new Constraint(other.CollisionsCount.min, other.CollisionsCount.max);
			HitCollisionCount = new Constraint(other.HitCollisionCount.min, other.HitCollisionCount.max);
			AverageAgentAngle = new Constraint(other.AverageAgentAngle.min, other.AverageAgentAngle.max);
			AverageEnemyAngle = new Constraint(other.AverageEnemyAngle.min, other.AverageEnemyAngle.max);
			AverageAgentEdgeDistance = new Constraint(other.AverageAgentEdgeDistance.min, other.AverageAgentEdgeDistance.max);
			AverageEnemyEdgeDistance = new Constraint(other.AverageEnemyEdgeDistance.min, other.AverageEnemyEdgeDistance.max);
			AverageAgentSkillState = new Constraint(other.AverageAgentSkillState.min, other.AverageAgentSkillState.max);
			//Tempo 
			ActionCount = new Constraint(other.ActionCount.min, other.ActionCount.max);
			AverageActionVariationRatio = new Constraint(other.AverageActionVariationRatio.min, other.AverageActionVariationRatio.max);
			AverageBotDistance = new Constraint(other.AverageBotDistance.min, other.AverageBotDistance.max);
			AverageAgentVelocity = new Constraint(other.AverageAgentVelocity.min, other.AverageAgentVelocity.max);
			AverageEnemyVelocity = new Constraint(other.AverageEnemyVelocity.min, other.AverageEnemyVelocity.max);
		}

		public PacingConstraints Clone() {
			return new PacingConstraints(this);
		}

		// Implement the default for all bots. 
		//public PacingConstraints Default() { 
		//	return new PacingConstraints(new Constraint(0f, 1f)...)
		//}
	}

	/// <summary>
	/// The main class that manages the pacing system, which collects gameplay data, calculates pacing factors and aspects, evaluates the pacing against the target, and provides the pacing information to other parts of the bot.
	/// </summary>
	public class PacingController: MonoBehaviour {
		// Target pacing and history pacing. 
		#region Pacing Target and History
		public List<Pacing> PacingsTarget = new();
		public List<Pacing> PacingsHistory = new();
		public AnimationCurve TargetThreats = new();
		public AnimationCurve TargetTempos = new();
		public AnimationCurve HistoryThreats = new();
		public AnimationCurve HistoryTempos = new();
		#endregion
		// Gamplay or runtime information. 
		#region Runtime Data
		public float SegmentDuration = 1f;
		private float battleDuration = 60f;
		private int currentSegmentIndex = -1; 
		private SumoAPI api; 
		private List<ISumoAction> originalActions = new();
		private List<ISumoAction> pacedActions = new();
		//private List<PacingSegmentInfo> pacingSegmentsInfo = new(); // Delete this, use currentSegmentInfo instead.
		private PacingSegmentInfo currentSegmentInfo = new();
		#endregion

		public void Init(SumoAPI api) { 
			PacingsTarget.Clear();
			PacingsHistory.Clear();
			TargetThreats = new();
			TargetTempos = new();
			HistoryThreats = new();
			HistoryTempos = new();
			SegmentDuration = 1f;
			currentSegmentIndex = -1;
			battleDuration = api.BattleInfo.Duration;
			this.api = api;
			originalActions.Clear();
			pacedActions.Clear();
			pacingSegmentsInfo.Clear();
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
			if (PacingsTarget == null || api == null) {
				Debug.LogError("Pacing target or api is not found");
			}

			float elapsed = Mathf.Clamp(api.BattleInfo.Duration - api.BattleInfo.TimeLeft, 0f, battleDuration);
			int segmentIndex = Mathf.FloorToInt(elapsed / SegmentDuration);

			if (segmentIndex != currentSegmentIndex) {
				FinalizeSegment(elapsed);
				currentPacing.pacingSegmentInfo.Reset();
				currentSegmentIndex = segmentIndex;
			}

			currentPacing.pacingSegmentInfo.Sample(api);

			// [Todo] Get constraints and factors from target. 
			// ResolveConstraints(segmentIndex); // Handle constraints for different bots. 
			PacingFactors factors = ComputeFactors(currentPacing.pacingSegmentInfo, api, PacingsTarget.First().pacingConstraints);

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
				threat, tempo, PacingsTarget.First().pacingAspects.weightThreat, PacingsTarget.First().pacingAspects.weightTempo, overallPacing);

			// Add pacingConstraints and pacingSegmentInfo
			currentPacing.pacingConstraints = PacingsTarget.First().pacingConstraints.Clone();
			// pacingSegmentInfo alread added. 
			//currentPacing.pacingSegmentInfo

			currentPacing = new Pacing(segmentIndex, currentPacingAspects, factors, currentPacing.pacingConstraints, currentPacing.pacingSegmentInfo);
			
			if (PacingsHistory.Count == 0 || PacingsHistory[^1].segmentIndex != segmentIndex) {
				PacingsHistory.Add(currentPacing);
				pacingsHistoryCurve.AddKey(elapsed, overallPacing);
				threatHistoryCurve.AddKey(elapsed, threat);
				tempoHistoryCurve.AddKey(elapsed, tempo);

			} else {
				PacingsHistory[^1] = currentPacing;
				pacingsHistoryCurve.MoveKey(pacingsHistoryCurve.length - 1, new Keyframe(elapsed, overallPacing));
				threatHistoryCurve.MoveKey(threatHistoryCurve.length - 1, new Keyframe(elapsed, threat));
				tempoHistoryCurve.MoveKey(tempoHistoryCurve.length - 1, new Keyframe(elapsed, tempo));
			}

			return currentPacing;
		}

		// [Todo] Edit later. 
		private void FinalizeSegment(float elapsed) {
			if (PacingsHistory.Count == 0 && currentSegmentIndex < 0) return;
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



	// ===================================================================================

	/// <summary>
	/// PacingConstraints is a structure to define expected ranges for the PacingFactors normalization. 
	/// The fields in PacingConstraints follow PacingVariables. 
	/// </summary>
	[Serializable]
	
	// [Todo] This stucture seems unecessary. We could also implement the fields directly into the PacingController. 
	public class Pacing {

		public struct PacingAspectThreat {
			public struct ThreatFactors {
				// [Edit] May need to implement the getter for each factor. 
				public float Collision;
				public float SkillAvailability;
				public float Angle;
				public float EdgeDistance;
				public float WeightCollision;
				public float WeightSkillAvailability;
				public float WeightAngle;
				public float WeightEdgeDistance;
			}

			// [Edit] Create getter function instead. 
			public float Value;		
			public ThreatFactors Factors;
		}

		public struct PacingAspectTempo {
			public struct TempoFactors {
				// [Edit] May need to implement the getter for each factor. 
				public float ActionIntensity;
				public float ActionDensity;
				public float BotDistance;
				public float Velocity;
				public float WeightActionIntensity;
				public float WeightActionDensity;
				public float WeightBotDistance;
				public float WeightVelocity;
			}

			// [Edit] Create getter function instead. 
			public float Value;
			public TempoFactors Factors;
		}

		public PacingAspectThreat Threat;
		public PacingAspectTempo Tempo;
		public float WeightThreat;
		public float WeightTempo; 

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
		public PacingFactors pacingFactors = new PacingFactors();

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

}
