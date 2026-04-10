using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

/// Structure:
/// - Pacing: the flow of the gameplay in a specific time segment, which includes the pacing aspects, factors, constraints and segment data.
/// - Pacing Aspects: the high level aspects of pacing, which are threat (danger) and tempo (action intensity).
/// - Pacing Factors: the specific factors that contribute to the pacing aspects. For example, the effectiveness of our bot hit collision behaviour and skill usage contribute to threat, while action intensity and distance to enemy effectiveness contribute to tempo.
/// - Constraints: the expected range of the pacing factors, which can be used to normalize the factors and evaluate the pacing. These constraints can be set globally or locally for each segment, and can also be blended together.
/// - Segment Data: the raw gameplay data collected in a specific time segment, which is used to calculate the pacing factors.
/// - Pacing Handler: the main class that manages the pacing system in a bot, which collects gameplay data, calculates pacing factors and aspects, evaluates the pacing against the target, and provides the pacing information to other parts of the bot.
namespace PacingFramework
{
	public class PacingHandler
	{
		// ================================
		// Runtime Config
		// ================================
		public SegmentData currentGameplayData;
		public float segmentDuration = 2f;
		public string PacingFileName = "";

		public int collisionWindowSize = 2;

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
		// Constructor
		// ================================
		public PacingHandler(SumoController controller, string pacingFileName, float segmentDuration, int collisionWindowSize)
		{
			this.controller = controller;
			this.PacingFileName = pacingFileName;
			this.segmentDuration = segmentDuration;
			this.collisionWindowSize = collisionWindowSize;

			// Subscribe to events
			controller.Events[SumoController.OnBounce].Subscribe(OnBounce);
			controller.Events[SumoController.OnAction].Subscribe(OnAction);

			// Load pacing config
			LoadPacingConfig();
		}

		// ================================
		// Lifecycle
		// ================================
		private void LoadPacingConfig()
		{
			string pacingConfigPath = $"Pacing/Constraints/{PacingFileName}";
			TextAsset pacingConfigAsset = Resources.Load<TextAsset>(pacingConfigPath);
			if (pacingConfigAsset == null)
			{
				Logger.Error($"pacingConfigPath {PacingFileName} JSON not found in Resources!");
				return;
			}

			PacingTarget = JsonUtility.FromJson<PacingTargetConfig>(pacingConfigAsset.text);
			Debug.Log($"[{controller.Side}] PacingConfig {PacingFileName} loaded: ThreatTargets={PacingTarget.ThreatTargets.Count}, Angle Min={PacingTarget.GlobalConstraints.Angle.Min}, Max={PacingTarget.GlobalConstraints.Angle.Max}");
		}

		public void Dispose()
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
			var predictedPacing = new SegmentPacing(predictedSegmentData, PacingTarget.GlobalConstraints, pacingHistory.CurrentRound().SegmentGameplayDatas, collisionWindowSize);

			return (predictedPacing.Threat.Value, predictedPacing.Tempo.Value);
		}

		private void FinalizeSegment()
		{
			// [Todo] Handle segment's local constraints if needed.
			GamePacingItem currentPacing = pacingHistory.CurrentRound();
			currentPacing.SegmentGameplayDatas.Add(new SegmentData(currentGameplayData));
			currentSegmentPacing = new SegmentPacing(currentGameplayData, PacingTarget.GlobalConstraints, currentPacing.SegmentGameplayDatas, collisionWindowSize);
			currentPacing.SegmentPacings.Add(currentSegmentPacing);

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

		public SegmentPacing GetCurrentSegmentPacing()
		{
			return currentSegmentPacing;
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

			if (parameter.BounceEvent.MyInfo.IsTieBreaker)
				type = CollisionType.Tie;
			else if (parameter.BounceEvent.MyInfo.IsActor)
				type = CollisionType.Hit;
			else
				type = CollisionType.Struck;

			SumoAPI api = controller.InputProvider.API;

			float angle = Mathf.Clamp01(api.Angle(normalized: true));

			currentGameplayData.RegisterCollision(type);
			currentGameplayData.RegisterAngle(angle);
		}

		private void OnAction(EventParameter parameter)
		{
			if (!parameter.Bool) // !isExecuted
			{
				SumoAPI api = controller.InputProvider.API;

				var safeDist = 1 - (api.BattleInfo.ArenaPosition - api.MyRobot.Position).magnitude / api.BattleInfo.ArenaRadius;

				Debug.Log($"SafeDistance {safeDist}");

				currentGameplayData.RegisterBotsDistance(api.DistanceNormalized());
				currentGameplayData.RegisterSafeDistance(safeDist);

				float angle = Mathf.Clamp01(api.Angle(normalized: true));

				currentGameplayData.RegisterAngle(angle);
				currentGameplayData.RegisterVelocity(controller.CachedVelocity.magnitude);

				foreach (var action in parameter.ActionList)
					currentGameplayData.RegisterAction(action);
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
			if (pacingHistory.CurrentRound().SegmentPacings.Count == 0)
			{
				return null;
			}

			int segmentIndex = pacingHistory.CurrentRound().SegmentPacings.Count - 1;
			SegmentPacing latestPacing = pacingHistory.CurrentRound().SegmentPacings[segmentIndex];

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
}
