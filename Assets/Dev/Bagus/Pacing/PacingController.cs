using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoManager;
using Unity.VisualScripting;
using UnityEngine;

/// Structure: 
/// - Pacing: the flow of the gameplay in a specific time segment, which includes the pacing aspects, factors, constraints and segment data. 
/// - Pacing Aspects: the high level aspects of pacing, which are threat (danger) and tempo (action intensity).
/// - Pacing Factors: the specific factors that contribute to the pacing aspects. For example, the effectiveness of our bot hit collision behaviour and skill usage contribute to threat, while action intensity and distance to enemy effectiveness contribute to tempo.
/// - Constraints: the expected range of the pacing factors, which can be used to normalize the factors and evaluate the pacing. These constraints can be set globally or locally for each segment, and can also be blended together. 
/// - Segment Data: the raw gameplay data collected in a specific time segment, which is used to calculate the pacing factors.
/// - Pacing Controller: the main class that manages the pacing system in a bot, which collects gameplay data, calculates pacing factors and aspects, evaluates the pacing against the target, and provides the pacing information to other parts of the bot.
namespace PacingFramework
{
	public class PacingController : MonoBehaviour
	{
		// ================================
		// Runtime Config
		// ================================
		public SegmentData currentGameplayData;
		public float segmentDuration = 2f;
		public string PacingFileName = "";

		private PacingTargetConfig pacingTarget = new PacingTargetConfig();

		private int tickCount;

		private SegmentPacing currentSegmentPacing;

		private GamePacing pacingHistory = new GamePacing();

		private SumoController controller;

		// [Todo]
		// Runtime Configs ==========
		//API: SumoAPI
		//Original Actions: <SumoAction> 
		//Paced Actions: <SumoAction> 

		// Test Fields
		private int segmentIndex = 0;

		// ================================
		// Unity
		// ================================

		void Start()
		{
			if (PacingFileName.Count() == 0)
			{
				Logger.Warning($"[{controller.Side}] PacingFileName not set. Default constraints is used");
				pacingTarget = new PacingTargetConfig();
				return;
			}

			string pacingConfigPath = $"Pacing/Constraints/{PacingFileName}";
			TextAsset pacingConfigAsset = Resources.Load<TextAsset>(pacingConfigPath);
			if (pacingConfigAsset == null)
			{
				Logger.Error($"pacingConfigPath {PacingFileName} JSON not found in Resources!");
				return;
			}

			pacingTarget = JsonUtility.FromJson<PacingTargetConfig>(pacingConfigAsset.text);
			Logger.Info($"[{controller.Side}] PacingConfig loaded");
		}

		void OnEnable()
		{
			controller = GetComponent<SumoController>();
			controller.Events[SumoController.OnBounce].Subscribe(OnBounce);
			controller.Events[SumoController.OnAction].Subscribe(OnAction);
		}

		void OnDisable()
		{
			controller.Events[SumoController.OnBounce].Unsubscribe(OnBounce);
			controller.Events[SumoController.OnAction].Unsubscribe(OnAction);
		}

		// ================================
		// Core Methods
		// ================================
		public void Init()
		{
			currentGameplayData = new SegmentData();
			tickCount = 0;
			segmentIndex = 0;
		}

		// [Todo] Remove deltaTime parameter. Then call every game tick. 
		public void Tick()
		{
			tickCount += 1;
			if ((tickCount / 10) < segmentDuration)
				return;

			FinalizeSegment();
			tickCount = 0;
			currentGameplayData.Reset();
		}

		private void FinalizeSegment()
		{
			// [Todo] Handle segment's local constraints if needed. 
			currentSegmentPacing = new SegmentPacing(currentGameplayData, pacingTarget.GlobalConstraints);
			pacingHistory.SegmentGameplayDatas.Add(new SegmentData(currentGameplayData));
			pacingHistory.SegmentPacings.Add(currentSegmentPacing);

			LogManager.LogPacing(currentGameplayData, currentSegmentPacing, segmentIndex, controller.Side);

			// Test 
			DebugPacing(currentSegmentPacing);
			DebugSegmentData(currentGameplayData);
			segmentIndex++;
		}

		public GamePacing GetHistory()
		{
			return pacingHistory;
		}

		public ConstraintConfig GetConstraints()
		{
			return pacingTarget.GlobalConstraints;
		}

		// ================================
		// Test Functions
		// ================================

		private void DebugPacing(SegmentPacing pacing)
		{
			Debug.Log($"===== SEGMENT {segmentIndex} FINALIZED =====");
			Debug.Log("PACING --> Threat: " + pacing.Threat.Value + ", Tempo: " + pacing.Tempo.Value + ", Overall: " + pacing.GetOverallPacing());
		}

		private void DebugSegmentData(SegmentData data)
		{
			Debug.Log("SEGMENT DATA [Counts] --> " + "Collisions: " + data.Collisions.Count + "; Angles: " + data.Angles.Count +
				"; SafeDistances: " + data.SafeDistances.Count + "; Actions: " + data.Actions.Count +
				"; BotsDistances: " + data.BotsDistances.Count + "; Velocities: " + data.Velocities.Count);
		}

		public void TestSimulation()
		{
			Debug.Log("Running Pacing Test Simulation...");

			for (int i = 0; i < 20; i++)
			{
				currentGameplayData.RegisterCollision((CollisionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(CollisionType)).Length));
				currentGameplayData.RegisterAngle(UnityEngine.Random.Range(0f, 180f));
				currentGameplayData.RegisterSafeDistance(UnityEngine.Random.Range(1f, 5f));
				currentGameplayData.RegisterVelocity(UnityEngine.Random.Range(0f, 10f));
				currentGameplayData.RegisterBotsDistance(UnityEngine.Random.Range(1f, 5f));

				int actions = UnityEngine.Random.Range(0, 50);
				for (int j = 0; j < actions; j++)
				{
					currentGameplayData.RegisterAction((ActionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(ActionType)).Length));
				}
			}

			FinalizeSegment();
		}

		private void TestSimulationContinuous()
		{
			currentGameplayData.RegisterCollision((CollisionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(CollisionType)).Length));
			currentGameplayData.RegisterAngle(UnityEngine.Random.Range(0f, 180f));
			currentGameplayData.RegisterSafeDistance(UnityEngine.Random.Range(1f, 5f));
			currentGameplayData.RegisterVelocity(UnityEngine.Random.Range(0f, 10f));
			currentGameplayData.RegisterBotsDistance(UnityEngine.Random.Range(1f, 5f));

			int actions = UnityEngine.Random.Range(0, 50);
			for (int i = 0; i < actions; i++)
			{
				currentGameplayData.RegisterAction((ActionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(ActionType)).Length));
			}
		}

		private void OnBounce(EventParameter parameter)
		{

			CollisionType type;
			float angle = Mathf.Abs(controller.InputProvider.API.Angle());
			angle = Mathf.Min(angle, 360 - angle);

			if (parameter.BounceEvent.MyInfo.IsTieBreaker)
				type = CollisionType.Tie;
			else if (parameter.BounceEvent.MyInfo.IsActor)
				type = CollisionType.Hit;
			else
				type = CollisionType.Struck;

			currentGameplayData.RegisterAngle(angle);
			currentGameplayData.RegisterCollision(type);
			Logger.Info($"PacingController.OnBounce Type: {type}, Rotation: {angle}");
		}

		private void OnAction(EventParameter parameter)
		{
			if (!parameter.Bool) // !isExecuted 
			{
				SumoAPI api = controller.InputProvider.API;
				float arenaRadius = api.BattleInfo.ArenaRadius;
				var safeDist = Mathf.Abs((arenaRadius - api.DistanceNormalized(targetPos: api.BattleInfo.ArenaPosition)) / arenaRadius);

				currentGameplayData.RegisterAction(parameter.Action.Type);
				currentGameplayData.RegisterBotsDistance(api.DistanceNormalized());
				currentGameplayData.RegisterSafeDistance(safeDist);
				currentGameplayData.RegisterVelocity(controller.RigidBody.linearVelocity.magnitude);
			}
		}

		// [Todo] 
		// Evaluation methods =========
		//EvaluatePacing(): void // Compare the actual latest pacing in pacinghistory with the pacingtarget according to the index. 
		//EvaluateAction(): void // Filtered out the original actions into paced actions. This requires rules on how to filter the action based on the pacing values. 
	}

	// ==========================================================
	// ENUMS
	// ==========================================================

	public enum AspectType { Threat, Tempo }
	public enum CollisionType { Hit, Struck, Tie }
	// public enum ActionType { Accelerate, TurnLeft, TurnRight, Dash, SkillBoost, SkillStone, Idle }

	public enum FactorType
	{
		HitCollision, Ability, Angle, SafeDistance,             // Threat factors
		ActionIntensity, ActionDensity, BotsDistance, Velocity  // Tempo factors
	}

	// ==========================================================
	// DATA CONTAINERS CLASSES
	// ==========================================================

	public class GamePacing
	{
		public List<SegmentData> SegmentGameplayDatas = new();
		public List<SegmentPacing> SegmentPacings = new();
	}

	[Serializable]
	public class PacingTargetConfig
	{
		public List<float> ThreatTargets = new();
		public List<float> TempoTargets = new();
		public ConstraintConfig GlobalConstraints = new();
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

		public float Normalize(float value)
		{
			var min = Mathf.Min(Min, MinLimit);
			var max = Mathf.Max(Max, MaxLimit);
			if (Mathf.Approximately(max, min)) return 0f;
			return Mathf.Clamp01((value - min) / (max - min));
		}
	}

	[Serializable]
	public class SegmentData
	{
		// Threat fields
		public List<CollisionType> Collisions = new();
		public List<float> Angles = new();
		public List<float> SafeDistances = new();

		// Tempo fields
		public List<ActionType> Actions = new();
		public List<float> BotsDistances = new();
		public List<float> Velocities = new();

		public SegmentData() { }

		public SegmentData(SegmentData other)
		{
			Collisions = new(other.Collisions);
			Angles = new(other.Angles);
			SafeDistances = new(other.SafeDistances);
			Actions = new(other.Actions);
			BotsDistances = new(other.BotsDistances);
			Velocities = new(other.Velocities);
		}

		public void Reset()
		{
			Collisions.Clear();
			Angles.Clear();
			SafeDistances.Clear();
			Actions.Clear();
			BotsDistances.Clear();
			Velocities.Clear();
		}

		// [Todo] Add SumoAPI as parameter or as field. 
		// Register functions to add data to the fields. Call these functions when the corresponding events happen in the game.
		public void RegisterCollision(CollisionType type) => Collisions.Add(type);
		public void RegisterAngle(float angle) => Angles.Add(angle);
		public void RegisterSafeDistance(float d) => SafeDistances.Add(d);
		public void RegisterAction(ActionType action) => Actions.Add(action);
		public void RegisterBotsDistance(float d) => BotsDistances.Add(d);
		public void RegisterVelocity(float v) => Velocities.Add(v);

		// Helper functions
		public Dictionary<ActionType, int> GetActionCounts()
		{
			return Actions.GroupBy(a => a).ToDictionary(g => g.Key, g => g.Count());
		}

		public Dictionary<CollisionType, int> GetCollisionCounts()
		{
			return Collisions.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
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
			return (Threat.Value * Threat.Weight + Tempo.Value * Tempo.Weight) / (Threat.Weight + Tempo.Weight);
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
			if (Type == FactorType.HitCollision)
				score = EvaluateHitCollision(data, constraints);
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

		// Evaluate the ratio of hit collision among all collisions.
		private float EvaluateHitCollision(SegmentData data, ConstraintConfig constraints)
		{
			float hitCollisionCount = data.Collisions.Count(c => c == CollisionType.Hit);
			float collisionCount = data.Collisions.Count;
			float hitCollisionRatio = (collisionCount < constraints.CollisionRatio.Min) ? 0f : hitCollisionCount / collisionCount;
			return constraints.CollisionRatio.Normalize(hitCollisionRatio);
		}

		// Evaluate the ratio of ability usage among all actions.
		private float EvaluateAbility(SegmentData data, ConstraintConfig constraints)
		{
			float abilityCount = data.Actions.Count(a => a == ActionType.Dash || a == ActionType.SkillBoost || a == ActionType.SkillStone);
			float abilityRatio = (data.Actions.Count < constraints.AbilityRatio.Min) ? 0f : abilityCount / data.Actions.Count;
			return constraints.AbilityRatio.Normalize(abilityRatio);
		}

		// Evaluate the average angle between the bot and its opponents when they collide or are close.
		private float EvaluateAngle(SegmentData data, ConstraintConfig constraints)
		{
			return data.Angles.Count > 0 ? constraints.Angle.Normalize(data.Angles.Average()) : 0f;
		}

		// Evaluate the average distance between the bot and its opponents when they collide or are close.
		private float EvaluateSafeDistance(SegmentData data, ConstraintConfig constraints)
		{
			return data.SafeDistances.Count > 0 ? constraints.SafeDistance.Normalize(data.SafeDistances.Average()) : 0f;
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

	// ==========================================================
	// BASE CLASS
	// ==========================================================

	public abstract class Aspect
	{
		protected SegmentData Data;
		protected ConstraintConfig Constraints;
		protected List<Factor> Factors = new();

		public float Value { get; protected set; }
		public float Weight = 1f;

		public Aspect(SegmentData data, ConstraintConfig constraints)
		{
			Data = data;
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
				weightedSum += v * f.Weight;
				totalWeight += f.Weight;
			}

			Value = totalWeight > 0 ? weightedSum / totalWeight : 0f;
		}

		// Get the list of factors including its 
		public List<(AspectType aspect, FactorType factor, float value, float weight)> GetFactorsInfo()
		{
			AspectType aspectType = (this is ThreatAspect) ? AspectType.Threat : AspectType.Tempo;
			return Factors.Select(f => (aspectType, f.Type, f.Evaluate(Data, Constraints), f.Weight)).ToList();
		}
	}
}