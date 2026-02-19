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
		public Threat(SegmentGameplayData data, ConstraintSet constraints): base(data, constraints) 
		{
			Factors.Add(new Factor(FactorType.HitCollision, 1f));
			Factors.Add(new Factor(FactorType.Ability, 1f));
			Factors.Add(new Factor(FactorType.Angle, 1f));
			Factors.Add(new Factor(FactorType.SafeDistance, 1f));
			Calculate();

			//factors.Add(new CollisionFactor(1f));
			//factors.Add(new DashFactor(1f));
			//factors.Add(new AngleFactor(1f));
			//factors.Add(new SafeDistanceFactor(1f));
			//Calculate();

			// or 
			//factors.Add(CreateFactor(FactorType.HitCollision, 1f));
			//factors.Add(CreateFactor(FactorType.Ability, 1f));
			//factors.Add(CreateFactor(FactorType.Angle, 1f));
			//factors.Add(CreateFactor(FactorType.SafeDistance, 1f));
			//Calculate();
		}
	}

	public class Tempo : Aspect
	{
		public Tempo(SegmentGameplayData data, ConstraintSet constraints): base(data, constraints) 
		{ 
			Factors.Add(new Factor(FactorType.ActionIntensity, 1f));
			Factors.Add(new Factor(FactorType.ActionDensity, 1f));
			Factors.Add(new Factor(FactorType.BotsDistance, 1f));
			Factors.Add(new Factor(FactorType.Velocity, 1f));
			Calculate();
		}
	}

	// ==========================================================
	// FACTOR CLASSES
	// ==========================================================

	public class Factor {
		public FactorType Type { get; private set; }
		public float Weight { get; private set; }
		public Factor(FactorType type, float weight) {
			Type = type;
			Weight = weight;
		}
		public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
			float score = 0f;

			if (Type == FactorType.HitCollision)
				score = constraints.Collision.Normalize(data.Collisions.Count);
			else if (Type == FactorType.Ability)
				score = constraints.DashSkill.Normalize(data.Actions.Count(a => a == ActionType.Dash || a == ActionType.SkillBoost));
			else if (Type == FactorType.Angle)
				score = data.Angles.Count > 0 ? constraints.Angle.Normalize(data.Angles.Average()) : 0f;
			else if (Type == FactorType.SafeDistance)
				score = data.SafeDistances.Count > 0 ? 1f - constraints.SafeDistance.Normalize(data.SafeDistances.Average()) : 0f;
			else if (Type == FactorType.ActionIntensity)
				score = constraints.ActionIntensity.Normalize(data.Actions.Count);
			else if (Type == FactorType.ActionDensity) {
				if (data.Actions.Count == 0) return 0f;
				var counts = data.GetActionCounts();
				float total = data.Actions.Count;
				float entropy = 0f;
				foreach (var c in counts.Values) {
					float p = c / total;
					entropy -= p * Mathf.Log(p);
				}
				score = constraints.ActionDensity.Normalize(entropy);
			} else if (Type == FactorType.BotsDistance)
				score = data.BotsDistances.Count > 0 ? 1f - constraints.BotsDistance.Normalize(data.BotsDistances.Average()) : 0f;
			else if (Type == FactorType.Velocity)
				score = data.Velocities.Count > 0 ? constraints.Velocity.Normalize(data.Velocities.Average()) : 0f;
			else
				throw new ArgumentException("Invalid factor type");

			return score;
		}
		
		// [Todo] Define all factor evaluation functions here. 
	}

	//public class CollisionFactor : IFactor
	//{
	//	public FactorType Type => FactorType.HitCollision;
	//	public float Weight { get; private set; }

	//	public CollisionFactor(float weight) {
	//		Weight = weight;
	//	}

	//	public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
	//		return constraints.Collision.Normalize(data.Collisions.Count);
	//	}
	//}

	//public class DashFactor: IFactor 
	//{
	//	public FactorType Type => FactorType.Ability;
	//	public float Weight { get; private set; }
	//	public DashFactor(float weight) {
	//		Weight = weight;
	//	}
	//	public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
	//		//return constraints.DashSkill.Normalize(data.Actions.Count(a => a == ActionType.Dash || a == ActionType.SkillBoost));
	//	}
	//}

	//public class AngleFactor : IFactor
	//{
	//	public FactorType Type => FactorType.Angle;
	//	public float Weight { get; private set; }
	//	public AngleFactor(float weight) {
	//		Weight = weight;
	//	}
	//	public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
	//		//return data.Angles.Count > 0 ? constraints.Angle.Normalize(data.Angles.Average()) : 0f;
	//	}
	//}

	//public class SafeDistanceFactor : IFactor
	//{
	//	public FactorType Type => FactorType.SafeDistance;
	//	public float Weight { get; private set; }
	//	public SafeDistanceFactor(float weight) {
	//		Weight = weight;
	//	}
	//	public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
	//		//return data.SafeDistances.Count > 0 ? 1f - constraints.SafeDistance.Normalize(data.SafeDistances.Average()) : 0f;
	//	}
	//}

	//public class ActionIntensityFactor : IFactor
	//{
	//	public FactorType Type => FactorType.ActionIntensity;
	//	public float Weight { get; private set; }
	//	public ActionIntensityFactor(float weight) {
	//		Weight = weight;
	//	}
	//	public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
	//		//return constraints.ActionIntensity.Normalize(data.Actions.Count);
	//	}
	//}

	//public class ActionDensityFactor : IFactor
	//{
	//	public FactorType Type => FactorType.ActionDensity;
	//	public float Weight { get; private set; }
	//	public ActionDensityFactor(float weight) {
	//		Weight = weight;
	//	}
	//	public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
	//		// [Todo] Implement the logic to calculate action density based on the distribution of action types. 
	//	}
	//}

	//public class BotsDistanceFactor : IFactor
	//{
	//	public FactorType Type => FactorType.BotsDistance;
	//	public float Weight { get; private set; }
	//	public BotsDistanceFactor(float weight) {
	//		Weight = weight;
	//	}
	//	public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
	//		//return data.BotsDistances.Count > 0 ? 1f - constraints.BotsDistance.Normalize(data.BotsDistances.Average()) : 0f;
	//	}
	//}

	//public class VelocityFactor : IFactor
	//{
	//	public FactorType Type => FactorType.Velocity;
	//	public float Weight { get; private set; }
	//	public VelocityFactor(float weight) {
	//		Weight = weight;
	//	}
	//	public float Evaluate(SegmentGameplayData data, ConstraintSet constraints) {
	//		//return data.Velocities.Count > 0 ? constraints.Velocity.Normalize(data.Velocities.Average()) : 0f;
	//	}
	//}

	// ==========================================================
	// BASE CLASSES
	// ==========================================================

	public abstract class Aspect
	{
		protected SegmentGameplayData Data;
		protected ConstraintSet Constraints;
		protected List<Factor> Factors = new();

		public float Value { get; protected set; }
		// [Todo]
		//Weight: float 

		public Aspect(SegmentGameplayData data, ConstraintSet constraints) {
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

		//public static IFactor CreateFactor(FactorType type, float weight) {
		//	return type switch {
		//		FactorType.HitCollision => new CollisionFactor(weight),
		//		FactorType.Ability => new DashFactor(weight),
		//		FactorType.Angle => new AngleFactor(weight),
		//		FactorType.SafeDistance => new SafeDistanceFactor(weight),
		//		FactorType.ActionIntensity => new ActionIntensityFactor(weight),
		//		FactorType.ActionDensity => new ActionDensityFactor(weight),
		//		FactorType.BotsDistance => new BotsDistanceFactor(weight),
		//		FactorType.Velocity => new VelocityFactor(weight),
		//		_ => throw new ArgumentException("Invalid factor type")
		//	};
		//}

		// [Todo] Implement factor weights and GetWeightByType(FactorType) to get the weight of each factor. 
	}

	public interface IFactor 
	{ 
		FactorType Type { get; }
		float Weight { get; }
		float Evaluate(SegmentGameplayData data, ConstraintSet constraints);
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

