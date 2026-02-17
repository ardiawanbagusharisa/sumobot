using SumoBot;
using SumoCore;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace PacingFramework 
{
	public struct MinMax
	{
		public float min;
		public float max;

		public MinMax(float min, float max) {
			this.min = min;
			this.max = max;
		}
	}

	/// <summary>
	/// A set of lower and upper limits for variable's normalizations. 
	/// Constraint fields are implemented in PacingConstraints, where each bot will have specific PacingConstraints. 
	/// </summary>
	[Serializable]
	public struct Constraint
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
	/// A collection of constraints that define acceptable ranges for various pacing metrics. 
	/// For specific bot normalization, create the specific constraints of it. 
	/// </summary>
	public struct Constraints
	{
		// Threat related constraints.
		public Constraint CollisionsCount;
		public Constraint HitCollisionCount;
		public Constraint AverageAgentAngle;
		public Constraint AverageEnemyAngle;
		public Constraint AverageAgentEdgeDistance;
		public Constraint AverageEnemyEdgeDistance;
		public Constraint AverageAgentSkillState;
		// Tempo related constraints.
		public Constraint ActionCount;
		public Constraint AverageActionVariationRatio;
		public Constraint AverageBotDistance;
		public Constraint AverageAgentVelocity;
		public Constraint AverageEnemyVelocity;

		public Constraints(
			Constraint collisionsCount, Constraint hitCollisionCount,
			Constraint averageAgentAngle, Constraint averageEnemyAngle,
			Constraint averageAgentEdgeDistance, Constraint averageEnemyEdgeDistance,
			Constraint averageAgentSkillState,
			Constraint actionCount, Constraint averageActionVariationRatio,
			Constraint averageBotDistance, Constraint averageAgentVelocity,
			Constraint averageEnemyVelocity) {
			CollisionsCount = collisionsCount;
			HitCollisionCount = hitCollisionCount;
			AverageAgentAngle = averageAgentAngle;
			AverageEnemyAngle = averageEnemyAngle;
			AverageAgentEdgeDistance = averageAgentEdgeDistance;
			AverageEnemyEdgeDistance = averageEnemyEdgeDistance;
			AverageAgentSkillState = averageAgentSkillState;
			ActionCount = actionCount;
			AverageActionVariationRatio = averageActionVariationRatio;
			AverageBotDistance = averageBotDistance;
			AverageAgentVelocity = averageAgentVelocity;
			AverageEnemyVelocity = averageEnemyVelocity;
		}

		public Constraints(Constraints other) {
			CollisionsCount = other.CollisionsCount;
			HitCollisionCount = other.HitCollisionCount;
			AverageAgentAngle = other.AverageAgentAngle;
			AverageEnemyAngle = other.AverageEnemyAngle;
			AverageAgentEdgeDistance = other.AverageAgentEdgeDistance;
			AverageEnemyEdgeDistance = other.AverageEnemyEdgeDistance;
			AverageAgentSkillState = other.AverageAgentSkillState;
			ActionCount = other.ActionCount;
			AverageActionVariationRatio = other.AverageActionVariationRatio;
			AverageBotDistance = other.AverageBotDistance;
			AverageAgentVelocity = other.AverageAgentVelocity;
			AverageEnemyVelocity = other.AverageEnemyVelocity;
		}

		public static Constraints Default() {
			return new Constraints(
				// Threat constraints.
				new Constraint(0f, 9999f),  // CollisionCount
				new Constraint(0f, 9999f),  // HitCollisionCount
				new Constraint(0f, 180f),   // AverageAgentAngle
				new Constraint(0f, 180f),   // AverageEnemyAngle
				new Constraint(0f, 6f),     // AverageAgentEdgeDistance
				new Constraint(0f, 6f),     // AverageEnemyEdgeDistance
				new Constraint(0f, 1f),     // AverageAgentSkillState
											// Tempo constraints.
				new Constraint(0f, 250f),   // ActionCount
				new Constraint(0f, 1f),     // VariationRatio
				new Constraint(0f, 3f),     // BotDistance
				new Constraint(0f, 8f),     // AgentVelocity
				new Constraint(0f, 8f)      // EnemyVelocity
			);
		}

		public Dictionary<string, Constraint> GetConstraintsByType(AspectType type) {
			Dictionary<string, Constraint> constraintsDict = new();
			if (type == AspectType.Threat) {
				constraintsDict["CollisionsCount"] = CollisionsCount;
				constraintsDict["HitCollisionCount"] = HitCollisionCount;
				constraintsDict["AverageAgentAngle"] = AverageAgentAngle;
				constraintsDict["AverageEnemyAngle"] = AverageEnemyAngle;
				constraintsDict["AverageAgentEdgeDistance"] = AverageAgentEdgeDistance;
				constraintsDict["AverageEnemyEdgeDistance"] = AverageEnemyEdgeDistance;
				constraintsDict["AverageAgentSkillState"] = AverageAgentSkillState;
			} else if (type == AspectType.Tempo) {
				constraintsDict["ActionCount"] = ActionCount;
				constraintsDict["AverageActionVariationRatio"] = AverageActionVariationRatio;
				constraintsDict["AverageBotDistance"] = AverageBotDistance;
				constraintsDict["AverageAgentVelocity"] = AverageAgentVelocity;
				constraintsDict["AverageEnemyVelocity"] = AverageEnemyVelocity;
			}
			return constraintsDict;
		}
	}

	/// <summary>
	/// The raw gameplay or runtime data collected in a segment (few seconds), used to calculate pacing factors. 
	/// The time segment range is determined by the PacingController. 
	/// </summary>
	public class SegmentGameplayData
	{
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
		public List<ActionType> Actions = new();			// Actions performed by our bot. 
		public Dictionary<ActionType, int> ActionsCount { get { return GetActionsCount(); } private set { } }
		public List<float> BotDistances = new();			// Distances between our bot and enemy. 
		public List<float> AgentVelocities = new();			// Velocities of our bot. 
		public List<float> EnemyVelocities = new();			// Velocities of enemy. 
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

		// Populate the segment gameplay data by registering the data from api. 
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
	/// A structure that encapsulates the pacing aspects, their factors, and variables for a specific segment. 
	/// </summary>
	public class SegmentPacing {
		#region Structures for aspects, factors, and variables.
		public enum AspectType
		{
			Threat,
			Tempo
		}

		public struct FactorType
		{
			public enum Threat
			{
				Collision,
				SkillAvailability,
				Angle,
				EdgeDistance
			}

			public enum Tempo
			{
				ActionIntensity,
				ActionDensity,
				BotDistance,
				Velocity
			}
		}

		public struct ThreatAspect
		{
			/// <summary>
			/// A structure that encapsulates various unnormalized parameters related to the threat aspect. 
			/// This structure is designed to hold data obtained from SegmentGameplayData, which later will be normalized by the associated constraints. 
			/// We pass the data from SegmentGameplayData when calculating the factors. 
			/// </summary> 
			public struct ThreatVariables
			{
				// [Todo] Consider to change this fields with getters where it automatically calculates the values from the raw data in SegmentGameplayData.
				public int CollisionsCount;
				public int HitCollisionsCount;
				public float AverageAgentAngle;
				public float AverageEnemyAngle;
				public float AverageAgentEdgeDistance;
				public float AverageEnemyEdgeDistance;
				public float AverageAgentSkillState;

				public ThreatVariables(int collisionsCount, int hitCollisionsCount, float averageAgentAngle, float averageEnemyAngle,
					float averageAgentEdgeDistance, float averageEnemyEdgeDistance, float averageAgentSkillState) {
					CollisionsCount = collisionsCount;
					HitCollisionsCount = hitCollisionsCount;
					AverageAgentAngle = averageAgentAngle;
					AverageEnemyAngle = averageEnemyAngle;
					AverageAgentEdgeDistance = averageAgentEdgeDistance;
					AverageEnemyEdgeDistance = averageEnemyEdgeDistance;
					AverageAgentSkillState = averageAgentSkillState;
				}

				public ThreatVariables(ThreatVariables other) {
					CollisionsCount = other.CollisionsCount;
					HitCollisionsCount = other.HitCollisionsCount;
					AverageAgentAngle = other.AverageAgentAngle;
					AverageEnemyAngle = other.AverageEnemyAngle;
					AverageAgentEdgeDistance = other.AverageAgentEdgeDistance;
					AverageEnemyEdgeDistance = other.AverageEnemyEdgeDistance;
					AverageAgentSkillState = other.AverageAgentSkillState;
				}

				public static ThreatVariables Default() {
					// Collision, HitCollision, AgentAngle, EnemyAngle, AgentEdgeDistance, EnemyEdgeDistance, AgentSkillAvailability.
					return new ThreatVariables(0, 0, 0f, 0f, 0f, 0f, 0f);
				}

				// [Todo] Create functions to get populate the fields from SegmentGameplayData. 
				// Note that the data in SegmentGameplayData is raw and unnormalized, and stored in list.
				// We need to calculate the average or other forms of aggregation for the variables used in factors before being used in factor evaluations. 
				// Then, we assume that the data in variables are ready to be passed to factors. Therefore, getters are also needed. 
				public void PopulateFromGameplayData(SegmentGameplayData gameplayData) {
					// return something... 
				}
			}

			public struct ThreatFactors
			{
				public ThreatVariables Variables;

				// [Edit] May need to implement the getter for each factor. 
				//public float Collision;
				public float SkillAvailability;
				public float Angle;
				public float EdgeDistance;
				public float WeightCollision;
				public float WeightSkillAvailability;
				public float WeightAngle;
				public float WeightEdgeDistance;
				public float TotalWeights { get { return WeightCollision + WeightSkillAvailability + WeightAngle + WeightEdgeDistance; } }

				// [Edit] Add factor calculations here. Getters() or in aspect. 
				// Each Factor evaluation needs: variables and constraints. 
				public float EvaluateCollision(Constraints constraints) {
					// This evaluation method are applied to all factor calculations. 

					// Option 1: Normalized value
					// Maps value linearly to 0 (min) to 1 (max). Outside the range will be clamped. 
					float score = constraints.CollisionsCount.Normalize(Variables.CollisionsCount);

					// Option 2: Violation-based score 
					// If value is inside range -> NormalizedViolation = 0; score = 1. 
					// If value is outside range -> penalty increases, score decreases as value moves further away.
					// Use this if we want to apply penalty instead of giving score.
					//float score = 1f - constraints.CollisionsCount.NormalizedViolation(Variables.CollisionsCount);

					return score;
				}

				public float EvaluateSkillAvailability(Constraints constraints) {
					return constraints.AverageAgentSkillState.Normalize(Variables.AverageAgentSkillState); 
					async 
				}

				
			}

			// [Todo] These factor calculations are generated by AI. Some are still wrong and need edit. 
			//private static PacingFactors ComputeFactors(PacingSegmentInfo acc, SumoAPI api, PacingConstraints constraints) {
			//	float collisionRatio = acc.TotalCollisions > 0 ? (float)acc.StruckCollisions / acc.TotalCollisions : 0f;
			//	float collision = Normalize(collisionRatio, constraints.struckCollision);

			//	// [Todo] Handle this manually later. 
			//	float skillState = acc.Samples > 0 ? acc.EnemySkillSum / acc.Samples : (api.EnemyRobot.Skill.IsActive ? 1f : api.EnemyRobot.Skill.IsSkillOnCooldown ? 0f : 0.5f);
			//	float enemySkill = Normalize(skillState, constraints.enemySkill);

			//	float avgAgentAngle = acc.Samples > 0 ? acc.AgentAngleSum / acc.Samples : Mathf.Abs(api.Angle());
			//	float avgEnemyAngle = acc.Samples > 0 ? acc.EnemyAngleSum / acc.Samples : Mathf.Abs(api.Angle(oriPos: api.EnemyRobot.Position, oriRot: api.EnemyRobot.Rotation, targetPos: api.MyRobot.Position));
			//	float deltaAngle = Normalize(FacingDelta(avgAgentAngle, avgEnemyAngle), new MinMax(0f, 1f));

			//	float avgAgentEdge = acc.Samples > 0 ? acc.AgentEdgeSum / acc.Samples : DistanceToEdge(api.MyRobot.Position, api.BattleInfo);
			//	float avgEnemyEdge = acc.Samples > 0 ? acc.EnemyEdgeSum / acc.Samples : DistanceToEdge(api.EnemyRobot.Position, api.BattleInfo);
			//	float edgeDelta = avgEnemyEdge - avgAgentEdge; // positive when we are closer to edge
			//	float edgeRange = Mathf.Max(constraints.agentDistanceEdge.min, constraints.enemyDistanceEdge.min);
			//	edgeRange = Mathf.Max(edgeRange, 0.1f);
			//	float deltaDistance = Normalize(edgeDelta, new MinMax(-edgeRange, edgeRange));

			//	float actionIntensity = Normalize(acc.ActionCount, constraints.totalAction);

			//	int possibleActions = Enum.GetValues(typeof(ActionType)).Length;
			//	float actionDensityRaw = possibleActions > 0 ? (float)acc.UniqueActionTypes.Count / possibleActions : 0f;
			//	float actionDensity = Normalize(actionDensityRaw * constraints.actionVariation.max, constraints.actionVariation);

			//	float avgDist = acc.Samples > 0 ? acc.DistanceToEnemySum / acc.Samples : Vector2.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
			//	float avgDistanceToEnemy = Normalize(avgDist, constraints.avgDistanceToEnemy);

			//	float avgAgentVel = acc.Samples > 0 ? acc.AgentVelocitySum / acc.Samples : api.MyRobot.LinearVelocity.magnitude;
			//	float avgEnemyVel = acc.Samples > 0 ? acc.EnemyVelocitySum / acc.Samples : api.EnemyRobot.LinearVelocity.magnitude;
			//	float velocityDelta = avgAgentVel - avgEnemyVel;
			//	float deltaVelocity = Normalize(velocityDelta, new MinMax(-constraints.enemyVelocity.max, constraints.agentVelocity.max));

			//	return new PacingFactors(
			//		collision,
			//		enemySkill,
			//		deltaAngle,
			//		deltaDistance,
			//		actionIntensity,
			//		actionDensity,
			//		avgDistanceToEnemy,
			//		deltaVelocity);
			//}

			public float Value;             // [Edit] Create getter function instead. 		
			public float Weight;            // This is different from factors' weights. Aspect weight is used to blend different aspects, while factor weight is used to blend different factors within the aspect.
			public ThreatFactors Factors;
			public Constraints Constraints; 

			public float GetWeight(FactorType.Threat type) {
				return type switch {
					FactorType.Threat.Collision => Factors.WeightCollision,
					FactorType.Threat.SkillAvailability => Factors.WeightSkillAvailability,
					FactorType.Threat.Angle => Factors.WeightAngle,
					FactorType.Threat.EdgeDistance => Factors.WeightEdgeDistance,
					_ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
				};
			}

			// [Edit] Add factors calculations here. Getters() or in factor. 
		}

		public struct TempoAspect
		{
			/// <summary>
			/// PacingVariables is a structure that encapsulates various unnormalized parameters related to the pacing of agent and enemy interactions within a simulation environment. 
			/// This structure is designed to hold data obtained from SegmentData, which later will be normalized before pacing calculation. 
			/// </summary>
			public struct TempoVariables
			{
				public int ActionCount;
				public float AverageActionVariationRatio;
				public float AverageBotDistance;
				public float AverageAgentVelocity;
				public float AverageEnemyVelocity;

				public TempoVariables(int actionCount, float averageActionVariationRatio, float averageBotDistance,
					float averageAgentVelocity, float averageEnemyVelocity) {
					ActionCount = actionCount;
					AverageActionVariationRatio = averageActionVariationRatio;
					AverageBotDistance = averageBotDistance;
					AverageAgentVelocity = averageAgentVelocity;
					AverageEnemyVelocity = averageEnemyVelocity;
				}

				public TempoVariables(TempoVariables other) {
					ActionCount = other.ActionCount;
					AverageActionVariationRatio = other.AverageActionVariationRatio;
					AverageBotDistance = other.AverageBotDistance;
					AverageAgentVelocity = other.AverageAgentVelocity;
					AverageEnemyVelocity = other.AverageEnemyVelocity;
				}

				public static TempoVariables Default() {
					return new TempoVariables(0, 0f, 0f, 0f, 0f);
				}
			}

			public struct TempoFactors
			{
				public TempoVariables Variables;

				// [Edit] May need to implement the getter for each factor. 
				public float ActionIntensity;
				public float ActionDensity;
				public float BotDistance;
				public float Velocity;
				public float WeightActionIntensity;
				public float WeightActionDensity;
				public float WeightBotDistance;
				public float WeightVelocity;
				public float TotalWeights { get { return WeightActionIntensity + WeightActionDensity + WeightBotDistance + WeightVelocity; } }

				// [Edit] Add factor calculations here. Getters() or in aspect.
			}

			public float Value;             // [Edit] Create getter function instead. 
			public float Weight;            // This is different from factors' weights. Aspect weight is used to blend different aspects, while factor weight is used to blend different factors within the aspect.
			public TempoFactors Factors;

			public float GetWeight(FactorType.Tempo type) {
				return type switch {
					FactorType.Tempo.ActionIntensity => Factors.WeightActionIntensity,
					FactorType.Tempo.ActionDensity => Factors.WeightActionDensity,
					FactorType.Tempo.BotDistance => Factors.WeightBotDistance,
					FactorType.Tempo.Velocity => Factors.WeightVelocity,
					_ => throw new ArgumentOutOfRangeException(nameof(type), type, null)
				};
			}

			// [Edit] Add factors calculations here. Getters() or in factor. 
		}
		#endregion

		public ThreatAspect Threat;
		public TempoAspect Tempo;

		public float TotalFactorWeights(AspectType type) {
			if (type == AspectType.Threat) {
				return Threat.Factors.TotalWeights;
			} else if (type == AspectType.Tempo) {
				return Tempo.Factors.TotalWeights;
			}
			return -1;
		}
		public float TotalAspectWeights { get { return Threat.Weight + Tempo.Weight; } }

		// [Edit] 1. Implement constructors
		// [Edit] 2. Implement aspect calculations 
		// [Edit] 3. Implement getters for aspects 

		// [Edit] 4. Implement Reset() to reset all fields including inner structures.
		public void Reset() {
			// [Todo] Reset all variables and factors to default value.
			// 
		}

		public float Overall() {
			return Threat.Value * Threat.Weight + Tempo.Value * Tempo.Weight;
		}

	}


}
