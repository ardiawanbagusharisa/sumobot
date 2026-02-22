using System;
using System.Collections.Generic;
using System.Linq;
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
		public float segmentDuration = 2f;
		private float timer;

		private SegmentData currentGameplayData;
		private SegmentPacing currentSegmentPacing;

		private GamePacing pacingHistory = new GamePacing();
		private GamePacingTarget pacingTarget = new GamePacingTarget();

		// [Todo]
		// Runtime Configs ==========
		//API: SumoAPI
		//Original Actions: <SumoAction> 
		//Paced Actions: <SumoAction> 

		// Test Fields
		private int segmentIndex = 1;

		// ================================
		// Unity
		// ================================
		private void Start() {
			Init();
			//TestSimulation(); // Auto test at start
		}

		private void Update() {
			TestSimulationContinuous(); // Continuous test 
			Tick(Time.deltaTime);
		}

		// ================================
		// Core Methods
		// ================================
		public void Init() {
			currentGameplayData = new SegmentData();
			timer = 0f;
		}

		// [Todo] Remove deltaTime parameter. Then call every game tick. 
		public void Tick(float deltaTime) {
			timer += deltaTime;

			if (timer >= segmentDuration) {
				FinalizeSegment();
				timer = 0f;
				currentGameplayData.Reset();
			}
		}

		private void FinalizeSegment() {
			// [Todo] Handle segment's local constraints if needed. 
			currentSegmentPacing = new SegmentPacing(currentGameplayData, pacingTarget.GlobalConstraints);
			pacingHistory.SegmentGameplayDatas.Add(new SegmentData(currentGameplayData));
			pacingHistory.SegmentPacings.Add(currentSegmentPacing);

			// Test 
			DebugPacing(currentSegmentPacing);
			DebugSegmentData(currentGameplayData);
			segmentIndex++;
		}

		// ================================
		// Test Functions
		// ================================

		private void DebugPacing(SegmentPacing pacing) {
			Debug.Log($"===== SEGMENT {segmentIndex} FINALIZED =====");
			Debug.Log("PACING --> Threat: " + pacing.Threat.Value + ", Tempo: " + pacing.Tempo.Value + ", Overall: " + pacing.GetOverallPacing());
		}

		private void DebugSegmentData(SegmentData data) {
			Debug.Log("SEGMENT DATA [Counts] --> " + "Collisions: " + data.Collisions.Count + "; Angles: " + data.Angles.Count +
				"; SafeDistances: " + data.SafeDistances.Count + "; Actions: " + data.Actions.Count +
				"; BotsDistances: " + data.BotsDistances.Count + "; Velocities: " + data.Velocities.Count);
		}

		public void TestSimulation() {
			Debug.Log("Running Pacing Test Simulation...");

			for (int i = 0; i < 20; i++) {
				currentGameplayData.RegisterCollision((CollisionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(CollisionType)).Length));
				currentGameplayData.RegisterAngle(UnityEngine.Random.Range(0f, 180f));
				currentGameplayData.RegisterSafeDistance(UnityEngine.Random.Range(1f, 5f));
				currentGameplayData.RegisterVelocity(UnityEngine.Random.Range(0f, 10f));
				currentGameplayData.RegisterBotsDistance(UnityEngine.Random.Range(1f, 5f));

				int actions = UnityEngine.Random.Range(0, 50);
				for (int j = 0; j < actions; j++) {
					currentGameplayData.RegisterAction((ActionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(ActionType)).Length));
				}
			}

			FinalizeSegment();
		}

		private void TestSimulationContinuous() {
			currentGameplayData.RegisterCollision((CollisionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(CollisionType)).Length));
			currentGameplayData.RegisterAngle(UnityEngine.Random.Range(0f, 180f));
			currentGameplayData.RegisterSafeDistance(UnityEngine.Random.Range(1f, 5f));
			currentGameplayData.RegisterVelocity(UnityEngine.Random.Range(0f, 10f));
			currentGameplayData.RegisterBotsDistance(UnityEngine.Random.Range(1f, 5f));

			int actions = UnityEngine.Random.Range(0, 50);
			for (int i = 0; i < actions; i++) {
				currentGameplayData.RegisterAction((ActionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(ActionType)).Length));
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
	public enum ActionType { Accelerate, TurnLeft, TurnRight, Dash, SkillBoost, SkillStone, Idle }

	public enum FactorType
	{ 
		HitCollision, Ability, Angle, SafeDistance,				// Threat factors
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

	public class GamePacingTarget
	{
		public ConstraintSet GlobalConstraints = new ConstraintSet();
		// [Todo]
		// Fields ==========
		//List of Local Constraints: <Constraints>  
		//List of Target Pacing: <Segment Pacing>
	}

	public class Constraint
	{
		public float Min;
		public float Max;

		public Constraint(float min, float max) {
			Min = min;
			Max = max;
		}

		public float Normalize(float value) {
			if (Mathf.Approximately(Max, Min)) return 0f;
			return Mathf.Clamp01((value - Min) / (Max - Min));
		}
	}

	public class ConstraintSet {
		// Threat constraints
		public Constraint CollisionRatio;       // Ratio of hit collision among all collisions.
		public Constraint AbilityRatio;         // Ratio of ability usage among all actions.
		public Constraint Angle;                // Average angle between the bot and its opponents when they collide or are close.
		public Constraint SafeDistance;         // Average distance between the bot and its opponents when they collide or are close.
		// Tempo constraints
		public Constraint ActionIntensity;      // Number of actions performed by the bot.
		public Constraint ActionDensity;        // Entropy of the action distribution.
		public Constraint BotsDistance;         // Average distance between the bot and its opponents.
		public Constraint Velocity;             // Average velocity of the bot.

		public ConstraintSet() {
			// Defaults 
			// [Todo] Need to carefully check at the simulated data from the Bot. Create a structure such as ScriptableObject to easily configure constraints for all bots. 
			CollisionRatio = new(0, 1);
			AbilityRatio = new(0, 0.2f);
			Angle = new(0, 180);
			SafeDistance = new(1, 5);
			ActionIntensity = new(0, 50);
			ActionDensity = new(0, 1);
			BotsDistance = new(1, 5);
			Velocity = new(0, 10);
		}
	}

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

		public SegmentData(SegmentData other) {
			Collisions = new(other.Collisions);
			Angles = new(other.Angles);
			SafeDistances = new(other.SafeDistances);
			Actions = new(other.Actions);
			BotsDistances = new(other.BotsDistances);
			Velocities = new(other.Velocities);
		}

		public void Reset() {
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
		public Dictionary<ActionType, int> GetActionCounts() {
			return Actions.GroupBy(a => a).ToDictionary(g => g.Key, g => g.Count());
		}

		public Dictionary<CollisionType, int> GetCollisionCounts() {
			return Collisions.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count());
		}
	}

	public class SegmentPacing
	{
		public ThreatAspect Threat;
		public TempoAspect Tempo;

		public SegmentPacing(SegmentData data, ConstraintSet constraints) {
			Threat = new ThreatAspect(data, constraints);
			Tempo = new TempoAspect(data, constraints);
		}

		public float GetOverallPacing() {
			return (Threat.Value * Threat.Weight + Tempo.Value * Tempo.Weight) / (Threat.Weight + Tempo.Weight);
		}
	}

	public class ThreatAspect : Aspect
	{
		public ThreatAspect(SegmentData data, ConstraintSet constraints): base(data, constraints) 
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
		public TempoAspect(SegmentData data, ConstraintSet constraints): base(data, constraints) 
		{ 
			Factors.Add(new Factor(FactorType.ActionIntensity, 1f));
			Factors.Add(new Factor(FactorType.ActionDensity, 1f));
			Factors.Add(new Factor(FactorType.BotsDistance, 1f));
			Factors.Add(new Factor(FactorType.Velocity, 1f));
			Calculate();
		}
	}

	public class Factor {
		public FactorType Type { get; private set; }
		public float Weight { get; private set; }
		public Factor(FactorType type, float weight) {
			Type = type;
			Weight = weight;
		}

		public float Evaluate(SegmentData data, ConstraintSet constraints) {
			float score = 0f;

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
			else if (Type == FactorType.ActionDensity) {
				score = EvaluateActionDensity(data, constraints);
			} else if (Type == FactorType.BotsDistance)
				score = EvaluateBotsDistance(data, constraints);
			else if (Type == FactorType.Velocity)
				score = EvaluateVelocity(data, constraints);
			else
				throw new ArgumentException("Invalid factor type");

			return score;
		}

		// Evaluate the ratio of hit collision among all collisions.
		private float EvaluateHitCollision(SegmentData data, ConstraintSet constraints) {
			float hitCollisionCount = data.Collisions.Count(c => c == CollisionType.Hit);
			float collisionCount = data.Collisions.Count;
			float hitCollisionRatio = (collisionCount < constraints.CollisionRatio.Min) ? 0f : hitCollisionCount / collisionCount;
			return constraints.CollisionRatio.Normalize(hitCollisionRatio);
		}

		// Evaluate the ratio of ability usage among all actions.
		private float EvaluateAbility(SegmentData data, ConstraintSet constraints) {
			float abilityCount = data.Actions.Count(a => a == ActionType.Dash || a == ActionType.SkillBoost || a == ActionType.SkillStone);
			float abilityRatio = (data.Actions.Count < constraints.AbilityRatio.Min) ? 0f : abilityCount / data.Actions.Count;
			return constraints.AbilityRatio.Normalize(abilityRatio);
		}

		// Evaluate the average angle between the bot and its opponents when they collide or are close.
		private float EvaluateAngle(SegmentData data, ConstraintSet constraints) {
			return data.Angles.Count > 0 ? constraints.Angle.Normalize(data.Angles.Average()) : 0f;
		}

		// Evaluate the average distance between the bot and its opponents when they collide or are close.
		private float EvaluateSafeDistance(SegmentData data, ConstraintSet constraints) {
			return data.SafeDistances.Count > 0 ? constraints.SafeDistance.Normalize(data.SafeDistances.Average()) : 0f;
		}

		// Evaluate the number of actions performed by the bot.
		private float EvaluateActionIntensity(SegmentData data, ConstraintSet constraints) {
			return constraints.ActionIntensity.Normalize(data.Actions.Count);
		}

		// Evaluate the entropy of the action distribution.
		private float EvaluateActionDensity(SegmentData data, ConstraintSet constraints) {
			if (data.Actions.Count == 0) return 0f;
			var counts = data.GetActionCounts();
			float total = data.Actions.Count;
			float entropy = 0f;
			foreach (var c in counts.Values) {
				float p = c / total;
				entropy -= p * Mathf.Log(p);
			}
			return constraints.ActionDensity.Normalize(entropy);
		}

		// Evaluate the average distance between the bot and its opponents.
		private float EvaluateBotsDistance(SegmentData data, ConstraintSet constraints) {
			return data.BotsDistances.Count > 0 ? 1f - constraints.BotsDistance.Normalize(data.BotsDistances.Average()) : 0f;
		}

		// Evaluate the average velocity of the bot.
		private float EvaluateVelocity(SegmentData data, ConstraintSet constraints) {
			return data.Velocities.Count > 0 ? constraints.Velocity.Normalize(data.Velocities.Average()) : 0f;
		}
	}

	// ==========================================================
	// BASE CLASS
	// ==========================================================

	public abstract class Aspect
	{
		protected SegmentData Data;
		protected ConstraintSet Constraints;
		protected List<Factor> Factors = new();

		public float Value { get; protected set; }
		public float Weight = 1f;

		public Aspect(SegmentData data, ConstraintSet constraints) {
			Data = data;
			Constraints = constraints;
			Calculate();
		}

		protected virtual void Calculate() {
			float weightedSum = 0f;
			float totalWeight = 0f;

			foreach (var f in Factors) {
				float v = f.Evaluate(Data, Constraints);
				weightedSum += v * f.Weight;
				totalWeight += f.Weight;
			}

			Value = totalWeight > 0 ? weightedSum / totalWeight : 0f;
		}
	}
}