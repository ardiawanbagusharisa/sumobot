using UnityEngine;

namespace PacingFramework 
{
	public class PacingController : MonoBehaviour
	{
		// Fields ==========
		//Pacing History: GamePacing
		//Pacing Target: GamePacingTarget

		// Runtime Configs ==========
		//SegmentDuration: float  
		//API: SumoAPI
		//Original Actions: <SumoAction> 
		//Paced Actions: <SumoAction> 

		// Temporary holder ==========
		//Current Gameplay Data: SegmentGameplayData
		//Current Segment Pacing: SegmentPacing

		// Methods ==========
		//Init(): void // Initialize all fields
		//Register functions of GameplayData -> few Register functions to register the raw data into currentGameplayData.

		//Tick(): void // Act like update for each game tick to allow register functions calling on currentGameplayData.
		//If a segment is reached, finalize the segment by populating the variables in segmentgameplaydata, compute pacing factors, and compute pacing aspect using finalize segment methods.
		//Then debug/ visualize it using debugpacing().
		//Later on we can also evaluate the pacing and actions using evaluation methods. 

		// Finalize Segment Methods ==========
		//PopulateSegmentGameplayData(): void // After currentSegmentData is registered and a segment is reached, ready for factor calculations. 
		//ComputeFactors(): void // Compute the pacing factors in currentsegmentpacing.Factors. Here, we need constraints for normalization and some helper functions from constraints: blending between global and local, or for each bot. 
		//ComputePacingAspect(): void // Compute the pacing aspect threat and tempo in currentsegmentpacing.Aspects and add data into pacinghistory.segmentgameplaydata and pacinghistory.pacing. 
		//DebugPacing(): void // Debug or visualize to console the details of pacing and segmentgameplaydata. 

		// Evaluation methods =========
		//EvaluatePacing(): void // Compare the actual latest pacing in pacinghistory with the pacingtarget according to the index. 
		//EvaluateAction(): void // Filtered out the original actions into paced actions. This requires rules on how to filter the action based on the pacing values. 

	}

	// ========================================== 
	public struct GamePacing 
	{
		// Fields ==========
		//List of Segment GameplayData: <SegmentGameplayData> 
		//List of Segment Pacing: <SegmentPacing> 
		//Curve of Pacing: PacingCurve // Custom curve for visualization 
	}

	public struct GamePacingTarget 
	{
		// Fields ==========
		//Global Constraints: Constraints 
		//List of Local Constraints: <Constraints>  
		//List of Target Pacing: <Segment Pacing>
		//Curve of Pacing: PacingCurve // Custom curve for visualization 
	}

	// ====================================================================================
	// Pacing.cs
	// ====================================================================================
	public enum AspectType
	{
		Threat,
		Tempo
	}

	public enum CollisionType
	{
		Hit,
		Struck,
		Tie
	}

	public enum ActionType
	{
		Accelerate,
		TurnLeft,
		TurnRight,
		Dash,
		SkillBoost,
		SkillStone,
		Idle
	}

	public enum FactorType
	{
		// Threat factors
		HitCollision,
		Ability,
		Angle,
		SafeDistance,

		//Tempo factors
		ActionIntensity,
		ActionDensity,
		BotsDistance,
		Velocity
	}

	public class Constraint { 
		// Fields ==========
		//Min: float
		//Max: float

		// Methods =========
		//Constructor(min, max) 
		//IsInRange(value): bool // return if the value is in constraint range. 
	}

	public struct ConstraintsSet {
		// Fields ==========
		//Threat fields
		//AvgHitCollision: constraint // factor avghitcollision: return 1 if hitcollision closer to max, return 0 if closer to min
		//AvgAngle: constraint // factor avgangle: return 1 if closer to min, return 0 if closer to max 
		//AvgSafeDistance: constraint // factor avgsafedistance: return 1 if closer to max, return 0 if closer to min 
		//AvgDashSkill: constraint // factor avgdashskill: return 1 if skill count closer to max, return 0 if closer to min 

		//Tempo fields
		//AvgAction: constraint // factor acvaction: return 1 if closer to max, return 0 if closer to min 
		//AvgBotDistance: constraint // factor avgbotdistance: return 1 if closer to min, return 0 if closer to max
		//AvgVelocity: constraint // factor avgvelocity: return 1 if closer to max, return 0 if closer to min 
		//AvgActionDensity: constraint // factor avgactiondensity: return 1 if closer to max, retunr 0 if closer to min; use shannon entropy

		// Methods ==========
		//NormalizedClamped(value, constraint): float // return the normalized and clamped value using this constraint. 
		//GetConstraintsByType(aspect type): dict<string, constraints> // return dictionary of constraint based on its aspect type related. 
	}

	public class SegmentGameplayData {
		// Fields ==========
		//Threat fields
		//List of collisions: <CollisionType>
		//List of agent angles: <float>
		//List of safe distances: <float> 

		//Tempo fields
		//List of list of actions: <<actiontype>> // This will be used in factor of action and action variation. 
		//List of bots distances: <float>
		//List of agent velocities: <float>

		// Methods ==========
		//Constructor()
		//Constructor(SegmentGameplayData)
		//Reset()

		// Register methods 
		//RegisterCollision(bounceevent or collisiontype, api): void 
		//RegisterAngle(api): void
		//RegisterSafeDistance(api): void 
		//RegisterActions(<actiontype> or api): void // Register list of actions from api 
		//RegisterBotsDistance(api): void

		// Helper methods 
		//GetCollisionsCounts(): dict<collisiontype, int>
		//GetActionsCounts(): dict<actiontype, int>
		//SafeDistance(pos, battleinfo): float
	}

	public class SegmentPacing 
	{
		// Fields ==========
		//Threat aspect: Aspect 
		//Tempo aspect: Aspect 
		//Constraints: Constraintsset [optional]

		// Methods ==========
		//GetOverallPacing(): float
	}

	public abstract class Aspect {
		// Fields ==========
		//Type: AspectType
		//Weight: float 
		//Value: float getter; call CalculatePacingAspect()
		
		// Private fields
		//GameplayData: SegmentGameplayData
		//Constraints: Constraintsset [optional]

		// Methods ==========	
		//Constructor(SegmentGameplayData) 
		//CalculateAspect(): float // Requires function calling from factors 
		//Reset(): void 

		// Helper methods
		//GetWeightByType(factor type): float
	}

	public class Threat : Aspect {
		// Additional Fields ==========
		//ThreatFactors

		// Additional Methods ==========	
	}
	public class Tempo : Aspect {
		// Additional Fields ==========
		//TempoFactors

		// Additional Methods ==========
	}

	// Option 1 implementation of factors
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

