using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoHelper;
using SumoInput;
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

		[HideInInspector]
		public PacingTargetConfig PacingTarget;


		private int tickCount;

		private SegmentPacing currentSegmentPacing;

		private GamePacing pacingHistory = new GamePacing();

		private SumoController controller;

		// Filtered Actions Storage (Testing)
		private List<ISumoAction> filteredActions = new List<ISumoAction>();

		// Progressive improvement tracking
		private List<float> improvementHistory = new List<float>();
		private float cumulativeClosenessPercent = 0f;
		private int evalCount = 0;

		private int segmentIndex = 0;

		// ================================
		// Unity
		// ================================

		void Start()
		{
			if (PacingFileName.Count() == 0)
			{
				Logger.Warning($"[{controller.Side}] PacingFileName not set. Default constraints is used");
				PacingTarget = new PacingTargetConfig();
				return;
			}

			string pacingConfigPath = $"Pacing/Constraints/{PacingFileName}";
			TextAsset pacingConfigAsset = Resources.Load<TextAsset>(pacingConfigPath);
			if (pacingConfigAsset == null)
			{
				Logger.Error($"pacingConfigPath {PacingFileName} JSON not found in Resources!");
				return;
			}

			PacingTarget = JsonUtility.FromJson<PacingTargetConfig>(pacingConfigAsset.text);
			Logger.Info($"[{controller.Side}] PacingConfig {PacingFileName} loaded");
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
			pacingHistory.InitBattle();
		}

		public void Tick()
		{
			RunEval();
			tickCount += 1;
			if ((tickCount / 10) < segmentDuration)
				return;

			FinalizeSegment();
			tickCount = 0;
			currentGameplayData.Reset();
		}

		private void RunEval()
		{
			PacingEvaluation eval = EvaluatePacing();
			if (eval == null) return;

			// Calculate original prediction
			var (origThreat, origTempo) = CalculatePredictedPacing(currentGameplayData.Actions, eval);

			// Filter actions and store them
			filteredActions = EvaluateAction(currentGameplayData.Actions, eval);

			// Calculate filtered prediction
			var (filtThreat, filtTempo) = CalculatePredictedPacing(filteredActions, eval);

			// Calculate improvements and percentage closer to target
			float origThreatDelta = Mathf.Abs(eval.ThreatDelta);
			float filtThreatDelta = Mathf.Abs(filtThreat - eval.TargetThreat);
			float threatImprovement = origThreatDelta - filtThreatDelta;
			float threatClosenessPercent = origThreatDelta > 0 ? (threatImprovement / origThreatDelta) * 100f : 0f;

			float origTempoDelta = Mathf.Abs(eval.TempoDelta);
			float filtTempoDelta = Mathf.Abs(filtTempo - eval.TargetTempo);
			float tempoImprovement = origTempoDelta - filtTempoDelta;
			float tempoClosenessPercent = origTempoDelta > 0 ? (tempoImprovement / origTempoDelta) * 100f : 0f;

			// Calculate averages
			float origAverage = (origThreat + origTempo) / 2f;
			float filtAverage = (filtThreat + filtTempo) / 2f;
			float targetAverage = (eval.TargetThreat + eval.TargetTempo) / 2f;
			float overallClosenessPercent = (threatClosenessPercent + tempoClosenessPercent) / 2f;

			// Count changed actions
			int changedCount = 0;
			for (int i = 0; i < Mathf.Min(currentGameplayData.Actions.Count, filteredActions.Count); i++)
			{
				if (currentGameplayData.Actions[i].Type != filteredActions[i].Type ||
					Mathf.Abs(currentGameplayData.Actions[i].Duration - filteredActions[i].Duration) > 0.01f)
					changedCount++;
			}

			// Track progressive improvement
			improvementHistory.Add(overallClosenessPercent);
			evalCount++;
			cumulativeClosenessPercent = 0f;
			foreach (float improvement in improvementHistory)
			{
				cumulativeClosenessPercent += improvement;
			}
			float progressiveAvgCloseness = cumulativeClosenessPercent / evalCount;


			// Multi-line log with all info including progressive average
			Logger.Info($"[{controller.Side}] PACING FILTER\n\nOriginal:\t\tThreat={origThreat:F3}({eval.ThreatDelta:F3}), Tempo={origTempo:F3}({eval.TempoDelta:F3}), Avg={origAverage:F3}\nFiltered:\t\tThreat={filtThreat:F3}({filtThreatDelta:F3}), Tempo={filtTempo:F3}({filtTempoDelta:F3}), Avg={filtAverage:F3}\nTarget:\t\tThreat={eval.TargetThreat:F3}, Tempo={eval.TargetTempo:F3}, Avg={targetAverage:F3}\nImprove:\t\tThreat={threatImprovement:F3} ({threatClosenessPercent:F1}%), Tempo={tempoImprovement:F3} ({tempoClosenessPercent:F1}%), Closeness={overallClosenessPercent:F1}%\nAct Changed:\t{changedCount}/{currentGameplayData.Actions.Count} ({changedCount * 100f / currentGameplayData.Actions.Count:F0}%)\nProgressive Avg=\t{progressiveAvgCloseness:F1}% (n={evalCount})");
		}

		/// <summary>
		/// Calculates the predicted pacing values from a set of actions.
		/// Returns (predictedThreat, predictedTempo).
		/// </summary>
		private (float, float) CalculatePredictedPacing(List<ISumoAction> actions, PacingEvaluation eval)
		{
			if (actions == null || actions.Count == 0)
				return (0f, 0f);

			// Simulate all actions and collect predicted factors
			var predictedSegmentData = new SegmentData();
			SumoAPI api = controller.InputProvider.API;

			for (int i = 0; i < actions.Count; i++)
			{
				var previousActions = actions.GetRange(0, i);
				var (predictedPos, predictedRot) = previousActions.Count > 0
					? api.Simulate(previousActions)
					: (api.MyRobot.Position, api.MyRobot.Rotation);

				var factors = PredictPacingFactors(predictedPos, predictedRot, actions[i]);

				// Add to predicted segment data
				predictedSegmentData.RegisterAngle(factors[FactorType.Angle]);
				predictedSegmentData.RegisterSafeDistance(factors[FactorType.SafeDistance]);
				predictedSegmentData.RegisterBotsDistance(factors[FactorType.BotsDistance]);
				predictedSegmentData.RegisterAction(actions[i]);
			}

			// Calculate predicted pacing
			var predictedPacing = new SegmentPacing(predictedSegmentData, PacingTarget.GlobalConstraints);

			return (predictedPacing.Threat.Value, predictedPacing.Tempo.Value);
		}

		private void FinalizeSegment()
		{


			// [Todo] Handle segment's local constraints if needed. 
			currentSegmentPacing = new SegmentPacing(currentGameplayData, PacingTarget.GlobalConstraints);
			pacingHistory.CurrentHistory().SegmentGameplayDatas.Add(new SegmentData(currentGameplayData));
			pacingHistory.CurrentHistory().SegmentPacings.Add(currentSegmentPacing);

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
			return PacingTarget.GlobalConstraints;
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
		}

		private void OnAction(EventParameter parameter)
		{
			if (!parameter.Bool) // !isExecuted 
			{
				SumoAPI api = controller.InputProvider.API;

				float arenaRadius = api.BattleInfo.ArenaRadius;
				var safeDist = Mathf.Abs((arenaRadius - api.DistanceNormalized(targetPos: api.BattleInfo.ArenaPosition)) / arenaRadius);

				currentGameplayData.RegisterAction(parameter.Action);
				currentGameplayData.RegisterBotsDistance(api.DistanceNormalized());
				currentGameplayData.RegisterSafeDistance(safeDist);
				currentGameplayData.RegisterVelocity(controller.RigidBody.linearVelocity.magnitude);
			}
		}

		// ================================
		// Evaluation methods
		// ================================

		/// <summary>
		/// Compare the actual latest pacing in pacingHistory with the pacingTarget according to the index.
		/// Returns the pacing evaluation results for both Threat and Tempo aspects.
		/// </summary>
		public PacingEvaluation EvaluatePacing()
		{
			if (pacingHistory.CurrentHistory().SegmentPacings.Count == 0)
			{
				Logger.Warning($"[{controller.Side}] No pacing history available for evaluation");
				return null;
			}

			int segmentIndex = pacingHistory.CurrentHistory().SegmentPacings.Count - 1;
			SegmentPacing latestPacing = pacingHistory.CurrentHistory().SegmentPacings[segmentIndex];

			// Get target values for current segment index (with bounds checking)
			float threatTarget = GetTargetValue(PacingTarget.ThreatTargets, segmentIndex);
			float tempoTarget = GetTargetValue(PacingTarget.TempoTargets, segmentIndex);

			// Calculate deltas
			float threatDelta = latestPacing.Threat.Value - threatTarget;
			float tempoDelta = latestPacing.Tempo.Value - tempoTarget;

			PacingEvaluation evaluation = new PacingEvaluation
			{
				SegmentIndex = segmentIndex,
				ActualThreat = latestPacing.Threat.Value,
				TargetThreat = threatTarget,
				ThreatDelta = threatDelta,
				ActualTempo = latestPacing.Tempo.Value,
				TargetTempo = tempoTarget,
				TempoDelta = tempoDelta
			};

			return evaluation;
		}

		/// <summary>
		/// Helper method to get target value from list with bounds checking.
		/// If index exceeds list size, returns the last available value.
		/// If list is empty, returns 0.5 as default target (middle of normalized range).
		/// </summary>
		private float GetTargetValue(List<float> targetList, int index)
		{
			if (targetList.Count == 0)
				return 0.5f; // Default to middle of normalized range

			if (index >= targetList.Count)
				return targetList[^1]; // Use last available target

			return targetList[index];
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
					currentGameplayData.RegisterAction(new AccelerateAction(InputType.Script, duration: 0.1f));
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
				currentGameplayData.RegisterAction(new AccelerateAction(InputType.Script, duration: 0.1f));
			}
		}

		// ================================
		// Action Evaluation & Filtering
		// ================================

		/// <summary>
		/// Evaluates and filters actions to better match target pacing values.
		/// Uses simulation to predict outcomes and selects actions that minimize pacing deltas.
		/// </summary>
		/// <param name="originalActions">The original sequence of actions from the bot</param>
		/// <param name="evaluation">Current pacing evaluation with threat/tempo deltas</param>
		/// <returns>Filtered/modified action sequence that better matches target pacing</returns>
		public List<ISumoAction> EvaluateAction(List<ISumoAction> originalActions, PacingEvaluation evaluation)
		{
			if (originalActions == null || originalActions.Count == 0)
				return new List<ISumoAction>();

			if (evaluation == null)
			{
				Logger.Warning($"[{controller.Side}] No evaluation provided, returning original actions");
				return new List<ISumoAction>(originalActions);
			}

			var pacedActions = new List<ISumoAction>();

			// Track simulated state as we build the action sequence
			var simulatedActions = new List<ISumoAction>();

			for (int i = 0; i < originalActions.Count; i++)
			{
				ISumoAction originalAction = originalActions[i];
				ISumoAction bestAction = originalAction;
				float bestScore = float.MaxValue;

				// Generate candidates dynamically based on current state and pacing needs
				var candidateActions = GenerateCandidateActions(evaluation, simulatedActions);

				// Evaluate original action
				float originalScore = ScoreAction(originalAction, simulatedActions, evaluation);
				bestScore = originalScore;

				// Try alternative actions
				foreach (var candidateAction in candidateActions)
				{
					float score = ScoreAction(candidateAction, simulatedActions, evaluation);

					if (score < bestScore)
					{
						bestScore = score;
						bestAction = candidateAction;
					}
				}

				// Add best action to sequence
				pacedActions.Add(bestAction);
				simulatedActions.Add(bestAction);
			}

			return pacedActions;
		}

		/// <summary>
		/// Generates a set of candidate actions optimized for the current pacing target.
		/// Calculates durations based on desired distance/angle changes using API measurements.
		/// </summary>
		private List<ISumoAction> GenerateCandidateActions(PacingEvaluation evaluation, List<ISumoAction> previousActions)
		{
			var candidates = new List<ISumoAction>();
			SumoAPI api = controller.InputProvider.API;
			SumoBotAPI myState = api.MyRobot;

			// Get current simulated state
			var (currentPos, currentRot) = previousActions.Count > 0
				? api.Simulate(previousActions)
				: (myState.Position, myState.Rotation);

			// Calculate current state metrics using API
			float angleToEnemy = api.Angle(currentPos, currentRot, api.EnemyRobot.Position);
			Vector2 distanceVector = api.Distance(currentPos, api.EnemyRobot.Position);
			float distanceToEnemy = distanceVector.magnitude;

			// Determine what we need based on pacing deltas
			bool needHigherThreat = evaluation.ThreatDelta < 0;
			bool needHigherTempo = evaluation.TempoDelta < 0;

			// Generate Turn actions based on angle needs
			if (needHigherThreat)
			{
				// Higher threat = better angle alignment (face enemy)
				// Calculate exact turn duration to face enemy
				float angleInDur = Mathf.Abs(angleToEnemy) / myState.RotateSpeed;
				angleInDur = Mathf.Clamp(angleInDur, ISumoAction.MinDuration, 1.0f);

				// Determine turn direction
				ActionType turnDirection = angleToEnemy > 0 ? ActionType.TurnLeft : ActionType.TurnRight;

				candidates.Add(new TurnAction(InputType.Script, turnDirection, angleInDur)); // Exact turn to face
				candidates.Add(new TurnAction(InputType.Script, turnDirection, angleInDur * 0.5f)); // Partial turn
				candidates.Add(new TurnAction(InputType.Script, turnDirection, angleInDur * 0.25f)); // Quarter turn
			}
			else
			{
				// Lower threat = turn away or maintain poor angle
				ActionType turnAwayDirection = angleToEnemy > 0 ? ActionType.TurnRight : ActionType.TurnLeft;

				// Generate turns that worsen angle
				float turnAwayDuration = Mathf.Abs(angleToEnemy) / myState.RotateSpeed;
				turnAwayDuration = Mathf.Clamp(turnAwayDuration * 0.5f, ISumoAction.MinDuration, 0.5f);

				candidates.Add(new TurnAction(InputType.Script, turnAwayDirection, turnAwayDuration));
				candidates.Add(new TurnAction(InputType.Script, turnAwayDirection, ISumoAction.MinDuration));
			}

			// Generate Accelerate actions with calculated durations based on distance
			if (needHigherTempo)
			{
				// Higher tempo = get closer to enemy
				// Calculate duration to close distance significantly
				float closeDistanceDur = CalculateAccelerateDuration(distanceToEnemy * 0.3f, myState.MoveSpeed); // Close 30% gap
				float aggressiveDur = CalculateAccelerateDuration(distanceToEnemy * 0.5f, myState.MoveSpeed); // Close 50% gap

				candidates.Add(new AccelerateAction(InputType.Script, duration: closeDistanceDur));
				candidates.Add(new AccelerateAction(InputType.Script, duration: aggressiveDur));
			}
			else
			{
				// Lower tempo = minimal movement
				float minimalDur = CalculateAccelerateDuration(distanceToEnemy * 0.05f, myState.MoveSpeed); // Move only 5%
				float moderateDur = CalculateAccelerateDuration(distanceToEnemy * 0.1f, myState.MoveSpeed); // Move 10%

				candidates.Add(new AccelerateAction(InputType.Script, duration: minimalDur));
				candidates.Add(new AccelerateAction(InputType.Script, duration: moderateDur));
			}

			// Always include minimal actions as fallback
			candidates.Add(new AccelerateAction(InputType.Script, duration: ISumoAction.MinDuration));
			candidates.Add(new TurnAction(InputType.Script, ActionType.TurnLeft, ISumoAction.MinDuration));
			candidates.Add(new TurnAction(InputType.Script, ActionType.TurnRight, ISumoAction.MinDuration));

			// Add abilities based on need
			if (needHigherThreat || needHigherTempo)
			{
				candidates.Add(new DashAction(InputType.Script)); // High threat & tempo
				candidates.Add(new SkillAction(InputType.Script, ActionType.SkillBoost)); // High threat & tempo
			}

			// Stone skill for defensive/low tempo play
			if (!needHigherTempo)
			{
				candidates.Add(new SkillAction(InputType.Script, ActionType.SkillStone));
			}

			return candidates;
		}

		/// <summary>
		/// Calculates the duration needed to travel a specific distance at given speed.
		/// </summary>
		private float CalculateAccelerateDuration(float distance, float moveSpeed)
		{
			if (moveSpeed <= 0) return ISumoAction.MinDuration;

			float duration = distance / moveSpeed;
			return Mathf.Clamp(duration, ISumoAction.MinDuration, 1.0f);
		}

		/// <summary>
		/// Scores an action based on how well it helps achieve target pacing.
		/// Lower score is better (closer to target).
		/// </summary>
		/// <param name="action">The action to evaluate</param>
		/// <param name="previousActions">Actions that have been simulated so far</param>
		/// <param name="evaluation">Current pacing evaluation with deltas</param>
		/// <returns>Score where lower is better</returns>
		private float ScoreAction(ISumoAction action, List<ISumoAction> previousActions, PacingEvaluation evaluation)
		{
			// Simulate the action to get predicted position/rotation
			var testActions = new List<ISumoAction>(previousActions) { action };
			var (predictedPos, predictedRot) = controller.InputProvider.API.Simulate(testActions);

			// Calculate predicted pacing factors
			var predictedFactors = PredictPacingFactors(predictedPos, predictedRot, action);

			// Calculate how this action affects threat and tempo
			float threatImpact = CalculateThreatImpact(predictedFactors, evaluation);
			float tempoImpact = CalculateTempoImpact(predictedFactors, evaluation);

			// Weight the impacts (can be tuned later)
			// Prioritize fixing the larger delta
			float threatWeight = Mathf.Abs(evaluation.ThreatDelta) > Mathf.Abs(evaluation.TempoDelta) ? 1.5f : 1.0f;
			float tempoWeight = Mathf.Abs(evaluation.TempoDelta) > Mathf.Abs(evaluation.ThreatDelta) ? 1.5f : 1.0f;

			return threatImpact * threatWeight + tempoImpact * tempoWeight;
		}

		/// <summary>
		/// Predicts pacing factors based on simulated position and rotation.
		/// </summary>
		private Dictionary<FactorType, float> PredictPacingFactors(Vector2 predictedPos, float predictedRot, ISumoAction action)
		{
			var factors = new Dictionary<FactorType, float>();
			SumoAPI api = controller.InputProvider.API;

			// Calculate angle to enemy from predicted position
			float angleToEnemy = Mathf.Abs(api.Angle(predictedPos, predictedRot, api.EnemyRobot.Position));
			angleToEnemy = Mathf.Min(angleToEnemy, 360 - angleToEnemy);
			factors[FactorType.Angle] = angleToEnemy;

			// Calculate safe distance (distance from arena edge)
			float arenaRadius = api.BattleInfo.ArenaRadius;
			float distFromCenter = Vector2.Distance(predictedPos, api.BattleInfo.ArenaPosition);
			float safeDist = Mathf.Abs((arenaRadius - distFromCenter) / arenaRadius);
			factors[FactorType.SafeDistance] = safeDist;

			// Calculate bots distance
			float botsDistance = api.DistanceNormalized(predictedPos, api.EnemyRobot.Position);
			factors[FactorType.BotsDistance] = botsDistance;

			// Action intensity contribution (1 action)
			factors[FactorType.ActionIntensity] = 1f;

			// Ability usage
			bool isAbility = action.Type == ActionType.Dash ||
							 action.Type == ActionType.SkillBoost ||
							 action.Type == ActionType.SkillStone;
			factors[FactorType.Ability] = isAbility ? 1f : 0f;

			return factors;
		}

		/// <summary>
		/// Calculates how the predicted factors impact threat relative to target.
		/// Returns penalty score (lower is better = moves threat toward target).
		/// </summary>
		private float CalculateThreatImpact(Dictionary<FactorType, float> predictedFactors, PacingEvaluation evaluation)
		{
			// Normalize factors using constraints
			float angleNorm = PacingTarget.GlobalConstraints.Angle.Normalize(predictedFactors[FactorType.Angle]);
			float safeDistNorm = PacingTarget.GlobalConstraints.SafeDistance.Normalize(predictedFactors[FactorType.SafeDistance]);
			float abilityNorm = predictedFactors[FactorType.Ability]; // Already 0 or 1

			// Predict this action's contribution to threat
			// Higher values = more threatening actions
			float actionThreatContribution = (angleNorm + safeDistNorm + abilityNorm) / 3f;

			// Determine if we need to increase or decrease threat
			float currentThreatDelta = evaluation.ThreatDelta; // Positive = too high, Negative = too low

			// Calculate penalty based on whether this action moves us toward target
			if (currentThreatDelta > 0)
			{
				// Threat is too high, we want LOWER threat actions
				// Penalize high-threat actions, reward low-threat actions
				return actionThreatContribution;
			}
			else
			{
				// Threat is too low, we want HIGHER threat actions
				// Penalize low-threat actions, reward high-threat actions
				return 1f - actionThreatContribution;
			}
		}

		/// <summary>
		/// Calculates how the predicted factors impact tempo relative to target.
		/// Returns penalty score (lower is better = moves tempo toward target).
		/// </summary>
		private float CalculateTempoImpact(Dictionary<FactorType, float> predictedFactors, PacingEvaluation evaluation)
		{
			// Normalize factors
			float actionIntensityNorm = PacingTarget.GlobalConstraints.ActionIntensity.Normalize(predictedFactors[FactorType.ActionIntensity]);
			float botsDistanceNorm = 1f - PacingTarget.GlobalConstraints.BotsDistance.Normalize(predictedFactors[FactorType.BotsDistance]);

			// Predict this action's contribution to tempo
			// Higher values = higher tempo actions (more intense, closer engagement)
			float actionTempoContribution = (actionIntensityNorm + botsDistanceNorm) / 2f;

			// Determine if we need to increase or decrease tempo
			float currentTempoDelta = evaluation.TempoDelta; // Positive = too high, Negative = too low

			// Calculate penalty based on whether this action moves us toward target
			if (currentTempoDelta > 0)
			{
				// Tempo is too high, we want LOWER tempo actions
				// Penalize high-tempo actions, reward low-tempo actions
				return actionTempoContribution;
			}
			else
			{
				// Tempo is too low, we want HIGHER tempo actions
				// Penalize low-tempo actions, reward high-tempo actions
				return 1f - actionTempoContribution;
			}
		}
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
		public Dictionary<int, Dictionary<int, GamePacingItem>> PacingHistories = new() { };

		public void InitBattle()
		{
			var currGameIdx = LogManager.CurrentGameIndex;
			var roundIdx = LogManager.GetCurrentRound().Index;

			if (PacingHistories.TryGetValue(currGameIdx, out var _))
			{
				if (!PacingHistories[currGameIdx].TryGetValue(roundIdx, out var _))
					PacingHistories[currGameIdx].Add(roundIdx, new GamePacingItem());
			}
			else
				PacingHistories.Add(currGameIdx, new() { [roundIdx] = new GamePacingItem() });
		}

		public GamePacingItem CurrentHistory()
		{
			var currGameIdx = LogManager.CurrentGameIndex;
			var roundIdx = LogManager.GetCurrentRound().Index;
			if (PacingHistories.TryGetValue(currGameIdx, out Dictionary<int, GamePacingItem> round))
			{
				if (round.TryGetValue(roundIdx, out var item))
				{
					return round[roundIdx];
				}
				else
				{
					var newInst = new GamePacingItem();
					PacingHistories[currGameIdx].Add(roundIdx, newInst);
					return newInst;
				}

			}
			else
			{
				var newInst = new GamePacingItem();
				PacingHistories[currGameIdx].Add(roundIdx, newInst);
				return newInst;
			}
		}
	}

	public class GamePacingItem
	{
		public List<SegmentData> SegmentGameplayDatas = new();
		public List<SegmentPacing> SegmentPacings = new();
	}

	[Serializable]
	public class PacingEvaluation
	{
		public int SegmentIndex;
		public float ActualThreat;
		public float TargetThreat;
		public float ThreatDelta;
		public float ActualTempo;
		public float TargetTempo;
		public float TempoDelta;

		public bool IsThreatAboveTarget => ThreatDelta > 0;
		public bool IsTempoAboveTarget => TempoDelta > 0;

		public override string ToString()
		{
			return $"Segment {SegmentIndex}: Threat={ActualThreat:F3} (Target={TargetThreat:F3}, Delta={ThreatDelta:F3}), " +
				   $"Tempo={ActualTempo:F3} (Target={TargetTempo:F3}, Delta={TempoDelta:F3})";
		}
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
		public List<ISumoAction> Actions = new();
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

		public void RegisterCollision(CollisionType type) => Collisions.Add(type);
		public void RegisterAngle(float angle) => Angles.Add(angle);
		public void RegisterSafeDistance(float d) => SafeDistances.Add(d);
		public void RegisterAction(ISumoAction action) => Actions.Add(action);
		public void RegisterBotsDistance(float d) => BotsDistances.Add(d);
		public void RegisterVelocity(float v) => Velocities.Add(v);

		// Helper functions
		public Dictionary<ISumoAction, int> GetActionCounts()
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
			float abilityCount = data.Actions.Count(a => a.Type == ActionType.Dash || a.Type == ActionType.SkillBoost || a.Type == ActionType.SkillStone);
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