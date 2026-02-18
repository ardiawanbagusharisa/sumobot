using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace PacingFramework
{
	public class PacingController : MonoBehaviour
	{
		// ================================
		// Runtime Config
		// ================================
		public float segmentDuration = 2f;
		private float timer;

		private SegmentGameplayData currentGameplayData;
		private SegmentPacing currentSegmentPacing;

		private GamePacing pacingHistory = new GamePacing();
		private GamePacingTarget pacingTarget = new GamePacingTarget();

		// [Todo]
		// Runtime Configs ==========
		//API: SumoAPI
		//Original Actions: <SumoAction> 
		//Paced Actions: <SumoAction> 

		// ================================
		// Unity
		// ================================
		private void Start() {
			Init();
			TestSimulation(); // Auto test at start
		}

		private void Update() {
			TestSimulationContinuous(); // Continuous test 
			Tick(Time.deltaTime);
		}

		// ================================
		// Core Methods
		// ================================
		public void Init() {
			currentGameplayData = new SegmentGameplayData();
			timer = 0f;
		}

		// [Todo] Remove deltaTime parameter.
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
			pacingHistory.SegmentGameplayDatas.Add(new SegmentGameplayData(currentGameplayData));
			pacingHistory.SegmentPacings.Add(currentSegmentPacing);

			DebugPacing(currentSegmentPacing);
		}

		private void DebugPacing(SegmentPacing pacing) {
			Debug.Log("===== SEGMENT FINALIZED =====");
			Debug.Log("Threat: " + pacing.ThreatAspect.Value);
			Debug.Log("Tempo: " + pacing.TempoAspect.Value);
			Debug.Log("Overall: " + pacing.GetOverallPacing());
		}

		// ================================
		// Test Functions
		// ================================
		public void TestSimulation() {
			Debug.Log("Running Pacing Test Simulation...");

			for (int i = 0; i < 20; i++) {
				currentGameplayData.RegisterCollision(CollisionType.Hit);
				currentGameplayData.RegisterAngle(UnityEngine.Random.Range(0f, 180f));
				currentGameplayData.RegisterSafeDistance(UnityEngine.Random.Range(1f, 10f));
				currentGameplayData.RegisterVelocity(UnityEngine.Random.Range(0f, 5f));
				currentGameplayData.RegisterBotsDistance(UnityEngine.Random.Range(1f, 15f));

				currentGameplayData.RegisterAction(
					(ActionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(ActionType)).Length)
				);
			}

			FinalizeSegment();
		}

		private void TestSimulationContinuous() {
			currentGameplayData.RegisterCollision(CollisionType.Hit);
			currentGameplayData.RegisterAngle(UnityEngine.Random.Range(0f, 180f));
			currentGameplayData.RegisterSafeDistance(UnityEngine.Random.Range(1f, 10f));
			currentGameplayData.RegisterVelocity(UnityEngine.Random.Range(0f, 5f));
			currentGameplayData.RegisterBotsDistance(UnityEngine.Random.Range(1f, 15f));

			currentGameplayData.RegisterAction(
				(ActionType)UnityEngine.Random.Range(0, Enum.GetValues(typeof(ActionType)).Length)
			);
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
		public List<SegmentGameplayData> SegmentGameplayDatas = new();
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

	public class ConstraintSet
	{
		// Threat constraints
		public Constraint Collision = new(0, 10);
		public Constraint Angle = new(0, 180);
		public Constraint SafeDistance = new(0, 20);
		public Constraint DashSkill = new(0, 10);
		// Tempo constraints
		public Constraint ActionIntensity = new(0, 20);
		public Constraint ActionDensity = new(0, 3);
		public Constraint BotsDistance = new(0, 20);
		public Constraint Velocity = new(0, 10);
	}

	public class SegmentGameplayData
	{
		// Threat fields
		public List<CollisionType> Collisions = new();
		public List<float> Angles = new();
		public List<float> SafeDistances = new();

		// Tempo fields
		public List<ActionType> Actions = new();
		public List<float> BotsDistances = new();
		public List<float> Velocities = new();

		public SegmentGameplayData() { }

		public SegmentGameplayData(SegmentGameplayData other) {
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

		// [Todo] add SumoAPI as parameter or as field. 
		// Register functions to add data to the fields. 
		public void RegisterCollision(CollisionType type) => Collisions.Add(type);
		public void RegisterAngle(float angle) => Angles.Add(angle);
		public void RegisterSafeDistance(float d) => SafeDistances.Add(d);
		public void RegisterAction(ActionType action) => Actions.Add(action);
		public void RegisterBotsDistance(float d) => BotsDistances.Add(d);
		public void RegisterVelocity(float v) => Velocities.Add(v);

		// Helper functions
		public Dictionary<ActionType, int> GetActionCounts() {
			return Actions.GroupBy(a => a)
						  .ToDictionary(g => g.Key, g => g.Count());
		}

		public Dictionary<CollisionType, int> GetCollisionCounts() {
			return Collisions.GroupBy(c => c)
							 .ToDictionary(g => g.Key, g => g.Count());
		}
	}

	public class SegmentPacing
	{
		public Threat ThreatAspect;
		public Tempo TempoAspect;

		public SegmentPacing(SegmentGameplayData data, ConstraintSet constraints) {
			ThreatAspect = new Threat(data, constraints);
			TempoAspect = new Tempo(data, constraints);
		}

		public float GetOverallPacing() {
			return (ThreatAspect.Value + TempoAspect.Value) / 2f;
		}
	}

	public class Threat : Aspect
	{
		public Threat(SegmentGameplayData data, ConstraintSet constraints)
			: base(data, constraints) { }

		protected override void Calculate() {
			// [Todo] Extract the logic of each factor evaluation into separate methods for better readability and maintainability. Each factor has its own weight. 
			float collision = constraints.Collision.Normalize(data.Collisions.Count);
			float dash = constraints.DashSkill.Normalize(
				data.Actions.Count(a => a == ActionType.Dash || a == ActionType.SkillBoost)
			);

			float angle = data.Angles.Count > 0
				? constraints.Angle.Normalize(data.Angles.Average())
				: 0f;

			float safeDistance = data.SafeDistances.Count > 0
				? 1f - constraints.SafeDistance.Normalize(data.SafeDistances.Average())
				: 0f;

			Value = (collision + dash + angle + safeDistance) / 4f;
		}
	}

	public class Tempo : Aspect
	{
		public Tempo(SegmentGameplayData data, ConstraintSet constraints)
			: base(data, constraints) { }

		// [Todo] Extract the logic of each factor evaluation into separate methods for better readability and maintainability. Each factor has its own weight. 
		protected override void Calculate() {
			float actionIntensity = constraints.ActionIntensity.Normalize(data.Actions.Count);

			float actionDensity = 0f;
			if (data.Actions.Count > 0) {
				var counts = data.GetActionCounts();
				float total = data.Actions.Count;
				float entropy = 0f;

				foreach (var c in counts.Values) {
					float p = c / total;
					entropy -= p * Mathf.Log(p);
				}

				actionDensity = constraints.ActionDensity.Normalize(entropy);
			}

			float botDistance = data.BotsDistances.Count > 0
				? 1f - constraints.BotsDistance.Normalize(data.BotsDistances.Average())
				: 0f;

			float velocity = data.Velocities.Count > 0
				? constraints.Velocity.Normalize(data.Velocities.Average())
				: 0f;

			Value = (actionIntensity + actionDensity + botDistance + velocity) / 4f;
		}
	}

	public abstract class Aspect
	{
		protected SegmentGameplayData data;
		protected ConstraintSet constraints;

		public float Value { get; protected set; }
		// [Todo]
		//Weight: float 

		public Aspect(SegmentGameplayData data, ConstraintSet constraints) {
			this.data = data;
			this.constraints = constraints;
			Calculate();
		}

		protected abstract void Calculate();
		// [Todo] Implement factor weights and GetWeightByType(FactorType) to get the weight of each factor. 
	}
	
	// ====================================================================================

	// Possible implementation of factors
	public struct ThreatFactors
	{
		// Fields ==========
		//Collision: float getter // Call EvaluateCollision()
		//DashSkill: float getter // Call EvaluateDashSkill()
		//Angle: float getter // Call EvaluateAngle()
		//SafeDistance: float getter // Call EvaluateSafeDistance()

		//Weight Collision: float 
		//Weight DashSkill: float 
		//Weight Angle: float 
		//Weight SafeDistance: float
		//TotalWeights: float getter // return sum of all weights. 

		// Methods ==========
		//Constructor(weights)
		//EvaluateCollision(segmentdata, constraint): float  // Not necessary to make segmentdata and constraint as fields.
		//EvaluateDashSkill(segmentdata, constraint): float 
		//EvaluateAngle(segmentdata, constraint): float 
		//EvaluateSafeDistance(segmentdata, constraint): float 
	}

	public struct TempoFactors
	{
		// Fields ==========
		//ActionIntensity: float getter
		//ActionDensity: float getter 
		//BotsDistance: float getter 
		//Velocity: float getter

		//Weight ActionIntensity: float 
		//Weight ActionDensity: float 
		//Weight BotsDistance: float 
		//Weight Velocity: float

		// Methods ==========
		//Constructor(weights)
		//EvaluateActionIntensity(segmentdata, constraint): float // Not necessary to make segmentdata and constraint as fields. 
		//EvaluateActionDensity(segmentdata, constraint): float
		//EvaluateBotDistance(segmentdata, constraint): float
		//EvaluateVelocity(segmentdata, constraint): float
	}
}

