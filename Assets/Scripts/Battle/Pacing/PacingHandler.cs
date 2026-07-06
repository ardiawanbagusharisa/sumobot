using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework.Internal;
using SumoBot;
using SumoCore;
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

		public float collisionWindowDuration = 3f; // Time-based window in seconds (e.g., 3 seconds lookback)

		public PacingTargetConfig PacingTarget;

		// Enable/disable action filtering (toggleable for testing)
		public bool EnableActionFiltering = true;

		private int tickCount;

		// Time tracking for collision storage
		private int currentSecond = 0;  // Current game second
		private int ticksInCurrentSecond = 0;  // Ticks accumulated in current second
		private float actionInterval = 0f;  // Cached action interval

		// Persistent collision history (survives across segments)
		private Dictionary<int, List<CollisionType>> collisionHistory = new();

		private SegmentPacing currentSegmentPacing;

		private GamePacing pacingHistory;

		private SumoController controller;

		// Filtered Actions Storage (Testing)
		private List<ISumoAction> filteredActions = new();

		// Store original unfiltered actions for comparison in RunEval
		private List<ISumoAction> originalUnfilteredActions = new();

		// Heuristic-based pacing brain (rule-based - no training required)
		private PacingBrainHeuristic pacingBrainHeuristic;

		// Original bot NN for generating candidate actions
		private NeuralNetwork originalBotNN;
		public bool useNNCandidates = false; // Enable NN-based candidate generation
		public string NNModelPath = "ML/Models/NN/NN_Model";

		// Fixed action pool (inspired by MCTS approach for reliable candidate actions)
		// Balanced pool: more acceleration options to prevent excessive turning
		private static readonly List<ISumoAction> BaseActionPool = new()
        {
			new AccelerateAction(InputType.Script, 0.1f),
			new DashAction(InputType.Script),
			new SkillAction(InputType.Script),
		};

		// Progressive improvement tracking
		private List<float> improvementHistory = new();
		private float cumulativeClosenessPercent = 0f;
		private int evalCount = 0;

		private int segmentIndex = 0;

		// Percentile calibration range (set from PacingManager)
		public float MinPacing = 0.0f;
		public float MaxPacing = 0.43f;

		// ================================
		// Constructor
		// ================================
		public PacingHandler(SumoController controller, string pacingFileName, float segmentDuration, float collisionWindowDuration, GamePacing sharedPacingHistory, float minPacing, float maxPacing, PacingBrainHeuristic sharedPacingBrainHeuristic = null, bool useNNCandidates = false)
		{
			this.controller = controller;
			this.segmentDuration = segmentDuration;
			this.collisionWindowDuration = collisionWindowDuration;
			PacingFileName = pacingFileName;
			this.pacingHistory = sharedPacingHistory;
			this.MinPacing = minPacing;
			this.MaxPacing = maxPacing;
			this.useNNCandidates = useNNCandidates;
			pacingBrainHeuristic = sharedPacingBrainHeuristic;

			// Subscribe to events
			controller.Events[SumoController.OnBounce].Subscribe(OnBounce);
			controller.Events[SumoController.OnAction].Subscribe(OnAction);
			controller.Events[SumoController.OnBeforeActionsQueued].Subscribe(OnBeforeActionsQueued);

			// Load pacing config
			LoadPacingConfig();

			if (sharedPacingBrainHeuristic != null)
			{
				pacingBrainHeuristic = sharedPacingBrainHeuristic;
				pacingBrainHeuristic.UpdateController(controller);  // Update controller reference for new round
				Logger.Info($"[{controller.Side}] Using shared PacingBrain Heuristic (rule-based, no training)");
			}
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
			controller.Events[SumoController.OnBeforeActionsQueued].Unsubscribe(OnBeforeActionsQueued);

			// Note: Episode tracking is now per-segment, not per-round
			// Model saving is handled by PacingManager
		}

		// ================================
		// Core Methods
		// ================================
		public void Init()
		{
			// Reset segment tracking
			currentGameplayData = new SegmentData();
			tickCount = 0;
			segmentIndex = 0;

			// Reset time tracking
			currentSecond = 0;
			ticksInCurrentSecond = 0;
			actionInterval = controller.InputProvider.API.BattleInfo.ActionInterval;

			// Clear collision history for new round
			collisionHistory.Clear();

			// Load original bot NN for candidate generation
			if (useNNCandidates && originalBotNN == null)
			{
				try
				{
					originalBotNN = NeuralNetwork.Load(NNModelPath);
					Logger.Info($"[{controller.Side}] Loaded NN model from {NNModelPath} for candidate generation");
				}
				catch (Exception e)
				{
					Logger.Warning($"[{controller.Side}] Failed to load NN model: {e.Message}. Using heuristic candidates only.");
					useNNCandidates = false;
				}
			}

			// Initialize pacing history for current round
			// This creates a fresh GamePacingItem with empty SegmentGameplayDatas
			pacingHistory.InitBattle();

			pacingBrainHeuristic?.OnRoundStart();

			Logger.Info($"[{controller.Side}] PacingHandler.Init() completed for Game {LogManager.CurrentGameIndex}, Round {LogManager.GetCurrentRound()?.Index}");
		}

		public void Tick()
		{
			SumoAPI api = controller.InputProvider.API;

			var safeDist = 1 - (api.BattleInfo.ArenaPosition - api.MyRobot.Position).magnitude / api.BattleInfo.ArenaRadius;
			
			float angle = Mathf.Clamp01(api.Angle(normalized: true));

			currentGameplayData.RegisterSafeDistance(safeDist);
			currentGameplayData.RegisterBotsDistance(api.DistanceNormalized());
			currentGameplayData.RegisterAngle(angle);
			currentGameplayData.RegisterVelocity(controller.CachedVelocity.magnitude);

			RunEval();

			// Update time tracking - check if we've reached the next second
			ticksInCurrentSecond++;
			int ticksPerSecond = Mathf.CeilToInt(1f / actionInterval);
			if (ticksInCurrentSecond >= ticksPerSecond)
			{
				currentSecond++;
				ticksInCurrentSecond = 0;
				Debug.Log($"[{controller.Side}] Second advanced to {currentSecond} (ticksPerSecond={ticksPerSecond}, actionInterval={actionInterval})");
			}

			if (tickCount * api.BattleInfo.ActionInterval >= segmentDuration)
			{
				FinalizeSegment();
				tickCount = 0;
				currentGameplayData.Reset();
			}

			tickCount++;

		}

		private void RunEval()
		{
			PacingEvaluation eval = EvaluatePacing();
			if (eval == null) return;

			// pacingBrainHeuristic?.TrainFromExperience(eval);  // No-op for heuristic

			// Use originalUnfilteredActions if available, otherwise fall back to currentGameplayData.Actions
			// This ensures we compare the TRUE original actions vs filtered actions
			var actionsToEvaluate = originalUnfilteredActions.Count > 0
				? originalUnfilteredActions
				: currentGameplayData.Actions;

			// Calculate original prediction
			var (origThreat, origTempo) = CalculatePredictedPacing(actionsToEvaluate, eval);

			// Calculate filtered prediction
			var (filtThreat, filtTempo) = CalculatePredictedPacing(filteredActions, eval);

			// Calculate improvements and percentage closer to target
			// Use CURRENT predictions (not past eval) for fair comparison
			float origThreatDelta = origThreat - eval.TargetThreat; // Current original prediction delta
			float filtThreatDelta = filtThreat - eval.TargetThreat; // Current filtered prediction delta

			float origTempoDelta = origTempo - eval.TargetTempo; // Current original prediction delta
			float filtTempoDelta = filtTempo - eval.TargetTempo; // Current filtered prediction delta

			// Calculate absolute distances for improvement measurement
			float origThreatDistance = Mathf.Abs(origThreatDelta);
			float filtThreatDistance = Mathf.Abs(filtThreatDelta);
			float origTempoDistance = Mathf.Abs(origTempoDelta);
			float filtTempoDistance = Mathf.Abs(filtTempoDelta);

			// Calculate improvement (positive = better, negative = worse)
			float threatImprovement = origThreatDistance - filtThreatDistance;
			float tempoImprovement = origTempoDistance - filtTempoDistance;

			// Calculate closeness percentage relative to full pacing range (0-1)
			// This prevents extreme percentages when already close to target
			const float PACING_RANGE = 1.0f; // Normalized pacing values are 0-1
			float threatClosenessPercent = (threatImprovement / PACING_RANGE) * 100f;
			float tempoClosenessPercent = (tempoImprovement / PACING_RANGE) * 100f;

			// Calculate averages
			float origAverage = (origThreat + origTempo) / 2f;
			float filtAverage = (filtThreat + filtTempo) / 2f;
			float targetAverage = (eval.TargetThreat + eval.TargetTempo) / 2f;
			float overallClosenessPercent = (threatClosenessPercent + tempoClosenessPercent) / 2f;

			// Count changed actions
			int changedCount = 0;
			for (int i = 0; i < Mathf.Min(actionsToEvaluate.Count, filteredActions.Count); i++)
			{
				if (actionsToEvaluate[i].Type != filteredActions[i].Type ||
					Mathf.Abs(actionsToEvaluate[i].Duration - filteredActions[i].Duration) > 0.01f)
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
			Logger.Info($"[{controller.Side}][{BattleManager.Instance.ElapsedTime}][Segment {segmentIndex}] PACING FILTER\n\nPast Eval:\t\tThreat Δ={eval.ThreatDelta:F3}, Tempo Δ={eval.TempoDelta:F3}\nOriginal:\t\tThreat={origThreat:F3}({origThreatDelta:+0.000;-0.000;0.000}), Tempo={origTempo:F3}({origTempoDelta:+0.000;-0.000;0.000}), Avg={origAverage:F3}\nFiltered:\t\tThreat={filtThreat:F3}({filtThreatDelta:+0.000;-0.000;0.000}), Tempo={filtTempo:F3}({filtTempoDelta:+0.000;-0.000;0.000}), Avg={filtAverage:F3}\nTarget:\t\t\tThreat={eval.TargetThreat:F3}, Tempo={eval.TargetTempo:F3}, Avg={targetAverage:F3}\nImprove:\t\tThreat={threatImprovement:F3} ({threatClosenessPercent:F1}%), Tempo={tempoImprovement:F3} ({tempoClosenessPercent:F1}%), Closeness={overallClosenessPercent:F1}%\nAct Changed:\t\t{changedCount}/{actionsToEvaluate.Count} ({changedCount * 100f / actionsToEvaluate.Count:F0}%)\nProgressive Avg=\t{progressiveAvgCloseness:F1}% (n={evalCount})\nOrig Actions: [{string.Join(", ", actionsToEvaluate.Select((x) => $"{x.Name} ({x.Duration})").ToList())}]\nFilter Actions: [{string.Join(", ", filteredActions.Select((x) => $"{x.Name} ({x.Duration})").ToList())}]");
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

			Vector2 prevPos = api.MyRobot.Position;
			float actionInterval = api.BattleInfo.ActionInterval;

			// Collision detection constants (approximate bot radius + buffer)
			const float COLLISION_THRESHOLD = 2.0f;

			for (int i = 0; i < actions.Count; i++)
			{
				var previousActions = actions.GetRange(0, i);
				var (predictedPos, predictedRot) = previousActions.Count > 0
					? api.Simulate(previousActions)
					: (api.MyRobot.Position, api.MyRobot.Rotation);

				var factors = PredictPacingFactors(predictedPos, predictedRot, actions[i]);

				// Calculate predicted velocity from position change
				float predictedVelocity = (predictedPos - prevPos).magnitude / actionInterval;

				// Detect potential collision with enemy
				// Assuming enemy position stays relatively constant during prediction
				float distanceToEnemy = (predictedPos - api.EnemyRobot.Position).magnitude;
				if (distanceToEnemy <= COLLISION_THRESHOLD)
				{
					// Predict collision type based on velocity comparison
					float myPredictedSpeed = predictedVelocity;
					float enemySpeed = api.EnemyRobot.LinearVelocity.magnitude;

					// Register predicted collisions (simulated, not real)
					if (myPredictedSpeed > enemySpeed + 0.1f)
						predictedSegmentData.RegisterCollision(CollisionType.Hit);
					else if (enemySpeed > myPredictedSpeed + 0.1f)
						predictedSegmentData.RegisterCollision(CollisionType.Struck);
					else
						predictedSegmentData.RegisterCollision(CollisionType.Tie);
				}

				prevPos = predictedPos;

				// Add to predicted segment data
				predictedSegmentData.RegisterAngle(factors[FactorType.Angle]);
				predictedSegmentData.RegisterSafeDistance(factors[FactorType.SafeDistance]);
				predictedSegmentData.RegisterBotsDistance(factors[FactorType.BotsDistance]);
				predictedSegmentData.RegisterAction(actions[i]);

				predictedSegmentData.RegisterVelocity(predictedVelocity);
			}

			// Calculate predicted pacing
			var predictedPacing = new SegmentPacing(predictedSegmentData, PacingTarget.GlobalConstraints);

			return (predictedPacing.Threat.Value, predictedPacing.Tempo.Value);
		}

		private void FinalizeSegment()
		{
			// Get current round's pacing data
			GamePacingItem currentRound = pacingHistory.CurrentRound();

			// Create deep copy of current segment data to preserve it in history
			SegmentData segmentCopy = new SegmentData(currentGameplayData);

			// Add the copy to history BEFORE calculating collision window
			currentRound.SegmentGameplayDatas.Add(segmentCopy);

			// Calculate and store collision window for this segment (true time-based, per second)
			// Pass the persistent collision history
			segmentCopy.CollisionData.CalculateWindow(collisionHistory, currentSecond, collisionWindowDuration);

			currentSegmentPacing = new SegmentPacing(
				segmentCopy,
				PacingTarget.GlobalConstraints
			);

			// Store calculated pacing
			currentRound.SegmentPacings.Add(currentSegmentPacing);

			// Log for external viewers/debugging
			LogManager.LogPacing(segmentCopy, currentSegmentPacing, segmentIndex, controller.Side);

			// Debug output
			DebugPacing(currentSegmentPacing);
			DebugSegmentData(currentGameplayData);

			pacingBrainHeuristic?.OnEpisodeEnd();  // No-op for heuristic

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

			float overallRaw = pacing.GetOverallPacing();
			float overallPercentile = pacing.GetPercentilePacing(MinPacing, MaxPacing);

			Debug.Log($"PACING (RAW) --> Threat: {pacing.Threat.Value:F3}, Tempo: {pacing.Tempo.Value:F3}, Overall: {overallRaw:F3}");
			Debug.Log($"PACING (PERCENTILE) --> Overall: {overallPercentile:F1}th percentile in range [{MinPacing:F3}, {MaxPacing:F3}]");
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

			// Store in persistent collision history
			if (!collisionHistory.ContainsKey(currentSecond))
			{
				collisionHistory[currentSecond] = new List<CollisionType>();
			}
			collisionHistory[currentSecond].Add(type);

			// Also register in current segment for legacy tracking
			currentGameplayData.RegisterCollision(type);
		}

		private void OnAction(EventParameter parameter)
		{
			if (!parameter.Bool) // !isExecuted
			{
				foreach (var action in parameter.ActionList)
					currentGameplayData.RegisterAction(action);
			}
		}

		/// <summary>
		/// Directly filters actions based on pacing evaluation.
		/// Called synchronously from SumoController.FlushInput().
		/// Returns filtered actions or null if filtering is disabled/unavailable.
		/// </summary>
		public List<ISumoAction> FilterActions()
		{
			if (!BattleManager.Instance.BotManager.LeftEnabled && controller.Side == PlayerSide.Left)
				return originalUnfilteredActions;
			if (!BattleManager.Instance.BotManager.RightEnabled && controller.Side == PlayerSide.Right)
				return originalUnfilteredActions;

			// Skip if filtering is disabled or no actions to filter
			if (!EnableActionFiltering)
			{
				return originalUnfilteredActions;
			}

			// Perform synchronous evaluation
			PacingEvaluation eval = EvaluatePacing();
			if (eval == null)
			{
				// No evaluation available yet (e.g., first segment), use original actions
				return originalUnfilteredActions;
			}

			// Filter the actions
			List<ISumoAction> filtered = EvaluateAction(originalUnfilteredActions, eval);

			// Log the filtering result for debugging with frame number
			if (filtered != null && filtered.Count > 0)
			{
				Logger.Info($"[{controller.Side}][FRAME {Time.frameCount}][TIME {Time.time:F3}] ACTION FILTER: Original={originalUnfilteredActions.Count}, Filtered={filtered.Count}, " +
					$"ThreatDelta={eval.ThreatDelta:F3}, TempoDelta={eval.TempoDelta:F3}");
			}
			else
			{
				Logger.Warning($"[{controller.Side}][FRAME {Time.frameCount}][TIME {Time.time:F3}] ACTION FILTER FAILED: No filtered actions generated! Using original {originalUnfilteredActions.Count} actions. " +
					$"ThreatDelta={eval.ThreatDelta:F3}, TempoDelta={eval.TempoDelta:F3}");
			}

			return filtered;
		}

		/// <summary>
		/// Called before actions are queued to allow synchronous filtering (DEPRECATED - use FilterActions instead).
		/// Evaluates pacing and provides filtered actions if filtering is enabled.
		/// </summary>
		private void OnBeforeActionsQueued(EventParameter parameter)
		{
			// Store original unfiltered acti ons for comparison in RunEval
			// This must happen BEFORE filtering to capture the true original actions
			if (parameter.ActionList != null && parameter.ActionList.Count > 0)
			{
				originalUnfilteredActions = new List<ISumoAction>(parameter.ActionList);
			}

			filteredActions = FilterActions();

			if (filteredActions != null && filteredActions.Count > 0)
			{
				parameter.FilteredActionList = filteredActions;
			}
		}

		// ================================
		// Evaluation methods
		// ================================

		/// <summary>
		/// Compare the actual latest pacing in pacingHistory with the pacingTarget according to the index.
		/// Returns the pacing evaluation results for both Threat and Tempo aspects.
		/// Automatically uses calibrated targets when percentile mode is enabled and calibration is loaded.
		/// </summary>
		public PacingEvaluation EvaluatePacing()
		{
			if (pacingHistory.CurrentRound().SegmentPacings.Count == 0)
			{
				return null;
			}

			int segmentIndex = pacingHistory.CurrentRound().SegmentPacings.Count - 1;
			SegmentPacing latestPacing = pacingHistory.CurrentRound().SegmentPacings[segmentIndex];

			// Get calibrated targets (converts percentile→raw using linear interpolation)
			var threatTargets = PacingTarget.GetCalibratedThreatTargets(MinPacing, MaxPacing);
			var tempoTargets = PacingTarget.GetCalibratedTempoTargets(MinPacing, MaxPacing);

			// Get target values for current segment index (with bounds checking)
			float threatTarget = GetTargetValue(threatTargets, segmentIndex);
			float tempoTarget = GetTargetValue(tempoTargets, segmentIndex);

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

		// ================================
		// Action Evaluation & Filtering
		// ================================

		/// <summary>
		/// Evaluates and filters actions to better match target pacing values.
		/// Uses simulation to predict outcomes and selects actions that minimize pacing deltas.
		/// Can generate completely different action sequences (different count and types).
		/// </summary>
		/// <param name="originalActions">The original sequence of actions from the bot</param>
		/// <param name="evaluation">Current pacing evaluation with threat/tempo deltas</param>
		/// <returns>Filtered/modified action sequence that better matches target pacing</returns>
		public List<ISumoAction> EvaluateAction(List<ISumoAction> originalActions, PacingEvaluation evaluation)
		{
			if (evaluation == null)
			{
				Logger.Warning($"[{controller.Side}] No evaluation provided, returning original actions");
				return new List<ISumoAction>(originalActions);
			}

			var pacedActions = new List<ISumoAction>();

			// Calculate current predicted pacing to determine what needs fixing
			var (predictedThreat, predictedTempo) = CalculatePredictedPacing(originalActions, evaluation);
			float currentThreatDelta = predictedThreat - evaluation.TargetThreat;
			float currentTempoDelta = predictedTempo - evaluation.TargetTempo;

			// Track simulated state as we build the action sequence
			var simulatedActions = new List<ISumoAction>();

			// Generate action sequence independently of original count
			// Allow flexible sequence length (1-5 actions typical for most bots)
			int maxActions = Mathf.Max(originalActions.Count, 3); // At least 3 actions for flexibility

			for (int i = 0; i < maxActions; i++)
			{
				// Get original action if available, otherwise null
				ISumoAction originalAction = i < originalActions.Count ? originalActions[i] : null;
				ISumoAction bestAction = originalAction;

				// Generate candidates dynamically based on current state and pacing needs
				var candidateActions = GenerateCandidateActions(evaluation, simulatedActions, currentThreatDelta, currentTempoDelta);

				// If no candidates available, stop generating actions
				if (candidateActions.Count == 0)
					break;

				if (pacingBrainHeuristic != null)
				{
					if (originalAction != null)
						candidateActions.Add(originalAction);

					if (pacingBrainHeuristic != null)
					{
						bestAction = pacingBrainHeuristic.SelectBestAction(candidateActions, evaluation, controller.InputProvider.API, simulatedActions);
					}
					else
						// Fallback to original if available, otherwise use first candidate
						bestAction ??= originalAction ?? candidateActions[0];
				}
				else
				{
					// Use traditional heuristic scoring
					float bestScore = float.MaxValue;

					// Evaluate original action if available
					if (originalAction != null)
					{
						float originalScore = ScoreAction(originalAction, simulatedActions, evaluation, currentThreatDelta, currentTempoDelta);
						bestScore = originalScore;
						bestAction = originalAction;
					}

					// Try alternative actions
					foreach (var candidateAction in candidateActions)
					{
						float score = ScoreAction(candidateAction, simulatedActions, evaluation, currentThreatDelta, currentTempoDelta);

						if (score < bestScore)
						{
							bestScore = score;
							bestAction = candidateAction;
						}
					}

					// If no best action found, use first candidate
					if (bestAction == null && candidateActions.Count > 0)
						bestAction = candidateActions[0];
				}

				// If we couldn't find any action, stop
				if (bestAction == null)
					break;

				if (pacedActions.Any((x) => x.Type == bestAction.Type && (Mathf.Abs(x.Duration - bestAction.Duration) < 0.1f))) continue;

				// Add best action to sequence
				pacedActions.Add(bestAction);
				simulatedActions.Add(bestAction);

				// Recalculate deltas for next iteration based on current progress
				var (currentThreat, currentTempo) = CalculatePredictedPacing(simulatedActions, evaluation);
				currentThreatDelta = currentThreat - evaluation.TargetThreat;
				currentTempoDelta = currentTempo - evaluation.TargetTempo;

				// Early termination: if we're close enough to target, stop adding actions
				if (Mathf.Abs(currentThreatDelta) < 0.05f && Mathf.Abs(currentTempoDelta) < 0.05f)
				{
					Logger.Info($"[{controller.Side}] Early termination: Close to target (ThreatΔ={currentThreatDelta:F3}, TempoΔ={currentTempoDelta:F3}) after {pacedActions.Count} actions");
					break;
				}
			}

			return pacedActions;
		}

		/// <summary>
		/// Generates a set of candidate actions from base pool with only safety validation.
		/// Neural network will handle all scoring and selection - no heuristic filtering!
		/// </summary>
		private List<ISumoAction> GenerateCandidateActions(PacingEvaluation evaluation, List<ISumoAction> previousActions, float currentThreatDelta, float currentTempoDelta)
		{
			var candidates = new List<ISumoAction>();
			SumoAPI api = controller.InputProvider.API;

			// Get current simulated state
			var (currentPos, currentRot) = previousActions.Count > 0
				? api.Simulate(previousActions)
				: (api.MyRobot.Position, api.MyRobot.Rotation);

			float angleToEnemy = api.Angle(currentPos, currentRot, api.EnemyRobot.Position, normalized: true);
			bool needHigherThreat = currentThreatDelta < 0;
			bool needHigherTempo = currentTempoDelta < 0;

			foreach (var action in BaseActionPool)
			{
				bool shouldInclude = true;

				// CRITICAL: Validate that action doesn't go out of bounds (hard constraint)
				var testActions = new List<ISumoAction>(previousActions) { action };
				var (testPos, testRot) = api.Simulate(testActions);
				Vector2 distanceFromArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition, oriPos: testPos);
				if (distanceFromArena.magnitude >= api.BattleInfo.ArenaRadius)
				{
					// HARD REJECT: This action would take us out of bounds
					continue; // Skip this action entirely
				}

				// Validate skills - only include if executable
				if (action.Type == ActionType.SkillBoost || action.Type == ActionType.SkillStone)
				{
					if (!api.CanExecute(action))
					{
						shouldInclude = false;
					}
					else
					{
						// Only include boost/dash for aggressive play
						if (action.Type == ActionType.SkillBoost && !needHigherThreat && !needHigherTempo)
							shouldInclude = false;
						// Only include stone for defensive play
						if (action.Type == ActionType.SkillStone && (needHigherThreat || needHigherTempo))
							shouldInclude = false;
					}
				}

				// Validate dash - skip if on cooldown or unsafe
				if (action.Type == ActionType.Dash)
				{
					if (api.MyRobot.IsDashOnCooldown)
					{
						shouldInclude = false;
					}
					else
					{
						// Check if dashing in current direction leads toward arena edge
						float angleToCenter = api.Angle(currentPos, currentRot, api.BattleInfo.ArenaPosition, normalized: true);
						Vector2 distFromCenter = api.Distance(targetPos: api.BattleInfo.ArenaPosition, oriPos: currentPos);
						float normalizedDist = distFromCenter.magnitude / api.BattleInfo.ArenaRadius;
						bool nearEdge = normalizedDist > 0.6f; // Dash is more aggressive, use tighter threshold
						bool facingAwayFromCenter = angleToCenter < 0.5f; // > 60 degrees off from center

						if (nearEdge && facingAwayFromCenter)
						{
							// Skip dash when near edge and facing away from center (very dangerous)
							shouldInclude = false;
						}
						else if (!needHigherThreat && !needHigherTempo)
						{
							// Only include dash for aggressive play
							shouldInclude = false;
						}
					}
				}

				// Filter turns based on threat needs and arena safety
				// if (action is TurnAction turn)
				// {
				// 	// Determine if this turn helps or hurts angle alignment
				// 	bool turnTowardsEnemy = angleToEnemy > 0.75f;

				// 	// Check if turning makes us face away from arena center
				// 	// Higher angle value = better alignment with center = safer
				// 	float angleToCenter = api.Angle(currentPos, currentRot, api.BattleInfo.ArenaPosition, normalized: true);
				// 	float angleAfterTurn = api.Angle(testPos, testRot, api.BattleInfo.ArenaPosition, normalized: true);
				// 	bool turningAwayFromCenter = angleAfterTurn < angleToCenter; // Lower alignment = facing more toward edge

				// 	if (needHigherThreat && !turnTowardsEnemy)
				// 	{
				// 		// Skip turns that worsen angle when we need threat
				// 		shouldInclude = false;
				// 	}
				// 	else if (!needHigherThreat && turnTowardsEnemy)
				// 	{
				// 		// Skip turns toward enemy when we don't need threat
				// 		shouldInclude = false;
				// 	}
				// 	else if (!needHigherThreat && !turnTowardsEnemy && turningAwayFromCenter)
				// 	{
				// 		// When lowering threat by turning away from enemy,
				// 		// reject if this turn makes us face away from arena center (toward edge)
				// 		shouldInclude = false;
				// 	}
				// }

				// Filter accelerate actions based on tempo needs and arena safety
				if (action is AccelerateAction accel)
				{
					// Check if accelerating in current direction leads toward arena edge
					float angleToCenter = api.Angle(currentPos, currentRot, api.BattleInfo.ArenaPosition, normalized: true);

					// If facing away from center (angle < 0.5 means > 60 degrees away from center)
					// and we're already close to edge, skip acceleration
					Vector2 distFromCenter = api.Distance(targetPos: api.BattleInfo.ArenaPosition, oriPos: currentPos);
					float normalizedDist = distFromCenter.magnitude / api.BattleInfo.ArenaRadius;
					bool nearEdge = normalizedDist > 0.7f; // Within 30% of arena radius from edge
					bool facingAwayFromCenter = angleToCenter < 0.5f; // > 60 degrees off from center

					if (nearEdge && facingAwayFromCenter)
					{
						// Skip acceleration when near edge and facing away from center
						shouldInclude = false;
					}
					else if (accel.Duration >= 0.3f && !needHigherTempo)
					{
						// Skip long accelerates when we don't need tempo
						shouldInclude = false;
					}
				}

				// Avoid duplicates (check by action type and approximate duration)
				bool isDuplicate = candidates.Any(c =>
					c.Type == action.Type && (Mathf.Abs(c.Duration - action.Duration) < 0.5f));

				if (!isDuplicate && shouldInclude)
					candidates.Add(action);
			}

			// Merge with NN-based candidates for richer action pool
			if (useNNCandidates && originalBotNN != null)
			{
				var nnCandidates = GetNNCandidateActions(previousActions);
				Debug.Log($"[PacingHandler] Candidate pool: {candidates.Count} total (heuristic + NN)\nNN: {string.Join(", ", nnCandidates)}\nHeuristic: {string.Join(", ", candidates)}");

				// candidates.Clear();

				foreach (var nnAction in nnCandidates)
				{
					// Avoid duplicates (check by action type and approximate duration)
					bool isDuplicate = candidates.Any(c =>
						c.Type == nnAction.Type && (Mathf.Abs(c.Duration - nnAction.Duration) < 0.5f));

					if (!isDuplicate)
					{
						candidates.Add(nnAction);
					}
					// candidates.Add(nnAction);
				}

			}

			return candidates;
		}

		/// <summary>
		/// Generates candidate actions from the original bot NN model.
		/// Uses the NN's learned behavior to provide diverse, bot-style action candidates.
		/// </summary>
		private List<ISumoAction> GetNNCandidateActions(List<ISumoAction> previousActions)
		{
			var candidates = new List<ISumoAction>();

			if (!useNNCandidates || originalBotNN == null)
				return candidates;

			SumoAPI api = controller.InputProvider.API;

			// Get current simulated state
			var (currentPos, currentRot) = previousActions.Count > 0
				? api.Simulate(previousActions)
				: (api.MyRobot.Position, api.MyRobot.Rotation);

			// Prepare NN inputs (matching AIBot_NN.cs format)
			float posX = currentPos.x / api.BattleInfo.ArenaRadius;
			float posY = currentPos.y / api.BattleInfo.ArenaRadius;
			float angle = api.Angle();
			float distanceNormalized = api.DistanceNormalized();
			float isDashCD = api.MyRobot.IsDashOnCooldown ? 1f : 0f;
			float isSkillCD = api.MyRobot.Skill.IsSkillOnCooldown ? 1f : 0f;

			float[] inputs = new float[] { posX, posY, angle, distanceNormalized, isDashCD, isSkillCD };

			// Run NN inference
			float[] outputs = originalBotNN.Forward(inputs);

			// Convert NN outputs to candidate actions (matching AIBot_NN.cs logic)
			// outputs: [accelerate, turnLeft, turnRight, dash, skill]
			float accelerate = outputs[0];
			float turnLeft = outputs[1];
			float turnRight = outputs[2];
			float dash = outputs[3];
			float skill = outputs[4];

			float angleThreshold = 10f; // From AIBot_NN
			float angleInDur = Mathf.Abs(angle) / api.MyRobot.RotateSpeed;

			// Add turn actions if NN suggests them
			if (angle > 0 && turnLeft > 0.05f)
			{
				float duration = Mathf.Max(0.1f, Mathf.Clamp01(angleInDur));
				candidates.Add(new TurnAction(InputType.Script, ActionType.TurnLeft, duration));
			}

			if (angle < 0 && turnRight > 0.05f)
			{
				float duration = Mathf.Max(0.1f, Mathf.Clamp01(angleInDur));
				candidates.Add(new TurnAction(InputType.Script, ActionType.TurnRight, duration));
			}

			// Add accelerate if NN suggests it
			if (Mathf.Abs(angle) < angleThreshold && accelerate > 0.05f)
			{
				float duration = Mathf.Max(0.1f, Mathf.Clamp01(accelerate));
				candidates.Add(new AccelerateAction(InputType.Script, duration));
			}

			// Add dash if available and NN suggests it
			if (!api.MyRobot.IsDashOnCooldown && dash > 0.05f)
			{
				candidates.Add(new DashAction(InputType.Script));
			}

			// Add skill if available and NN suggests it
			if (!api.MyRobot.Skill.IsSkillOnCooldown && skill > 0.05f)
			{
				candidates.Add(new SkillAction(InputType.Script));
			}

			return candidates;
		}

		/// <summary>
		/// Scores an action based on how well it helps achieve target pacing.
		/// Includes MCTS-inspired bonus/penalty system for tactical awareness.
		/// Lower score is better (closer to target).
		/// </summary>
		/// <param name="action">The action to evaluate</param>
		/// <param name="previousActions">Actions that have been simulated so far</param>
		/// <param name="evaluation">Current pacing evaluation with deltas</param>
		/// <param name="currentThreatDelta">Current predicted threat delta (not past)</param>
		/// <param name="currentTempoDelta">Current predicted tempo delta (not past)</param>
		/// <returns>Score where lower is better</returns>
		private float ScoreAction(ISumoAction action, List<ISumoAction> previousActions, PacingEvaluation evaluation, float currentThreatDelta, float currentTempoDelta)
		{
			SumoAPI api = controller.InputProvider.API;

			// Simulate the action to get predicted position/rotation
			var testActions = new List<ISumoAction>(previousActions) { action };
			var (predictedPos, predictedRot) = api.Simulate(testActions);

			// Calculate predicted pacing factors
			var predictedFactors = PredictPacingFactors(predictedPos, predictedRot, action);

			// Calculate how this action affects threat and tempo
			float threatImpact = CalculateThreatImpact(predictedFactors, evaluation);
			float tempoImpact = CalculateTempoImpact(predictedFactors, evaluation);

			// Dynamic weighting based on CURRENT predicted deltas (not past eval)
			// Larger deltas get proportionally higher weights to prioritize fixing bigger problems
			float absThreatDelta = Mathf.Abs(currentThreatDelta);
			float absTempoDelta = Mathf.Abs(currentTempoDelta);
			float totalDelta = absThreatDelta + absTempoDelta;

			float threatWeight, tempoWeight;
			if (totalDelta < 0.001f) // Near-perfect pacing, equal weights
			{
				threatWeight = 1.0f;
				tempoWeight = 1.0f;
			}
			else
			{
				// Calculate base ratio (0-1 for each, sum = 1.0)
				float threatRatio = absThreatDelta / totalDelta;
				float tempoRatio = absTempoDelta / totalDelta;

				// Scale weights: 0.5 min (still consider both) to 3.0 max (strong priority)
				// Formula: lerp between min and max based on ratio, with exponential scaling
				float minWeight = 0.5f;
				float maxWeight = 3.0f;
				float exponent = 1.5f; // Controls aggressiveness (higher = more aggressive prioritization)

				threatWeight = Mathf.Lerp(minWeight, maxWeight, Mathf.Pow(threatRatio, exponent));
				tempoWeight = Mathf.Lerp(minWeight, maxWeight, Mathf.Pow(tempoRatio, exponent));

				// Boost tempo weight if it's the larger problem (helps tempo catch up faster)
				if (absTempoDelta > absThreatDelta)
				{
					tempoWeight *= 1.5f; // 50% boost for tempo when it's struggling
				}
			}

			float baseScore = threatImpact * threatWeight + tempoImpact * tempoWeight;

			// ===== MCTS-Inspired Bonus/Penalty System =====
			float bonusOrPenalty = 0f;

			// Get current angle for tactical checks
			float currentAngle = api.Angle(normalized: true);
			float predictedAngle = api.Angle(predictedPos, predictedRot, api.EnemyRobot.Position, normalized: true);

			// Penalty for going out of bounds (critical failure)
			Vector2 distanceFromArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition, oriPos: predictedPos);
			if (distanceFromArena.magnitude >= api.BattleInfo.ArenaRadius)
			{
				bonusOrPenalty += 10000f; // ABSOLUTELY MASSIVE penalty - this should NEVER happen
			}
			// Also penalize getting too close to edge (within 10% of radius)
			else if (distanceFromArena.magnitude >= api.BattleInfo.ArenaRadius * 0.9f)
			{
				bonusOrPenalty += 5.0f; // Strong penalty for danger zone
			}

			// Penalty for poor angle alignment (minimum engagement requirement)
			if (predictedAngle < 0.3f) // Less than 30% aligned = facing away
			{
				bonusOrPenalty += 10.0f; // Heavy penalty for ignoring enemy
			}

			// Tactical bonuses for smart plays (MCTS-inspired)
			if (action.Type == ActionType.SkillStone || action.Type == ActionType.SkillBoost)
			{
				if (api.CanExecute(action))
				{
					bonusOrPenalty -= 1.0f; // Bonus for using available skills
				}
				else
				{
					bonusOrPenalty += 0.5f; // Small penalty for trying unusable skills
				}
			}

			if (action is TurnAction)
			{
				// Penalty if turn worsens angle alignment
				if (predictedAngle < currentAngle)
				{
					bonusOrPenalty += 0.5f;
				}
			}

			if (action is AccelerateAction)
			{
				// Bonus for accelerating when well-aligned (smart aggressive play)
				if (predictedAngle > 0.85f)
				{
					bonusOrPenalty -= 0.3f;
				}
			}

			if (action.Type == ActionType.Dash && !api.MyRobot.IsDashOnCooldown)
			{
				// Bonus for dashing when perfectly aligned (MCTS uses 0.95 threshold)
				if (predictedAngle > 0.95f)
				{
					bonusOrPenalty -= 0.5f;
				}
				else
				{
					// Small penalty for dashing when not aligned
					bonusOrPenalty += 0.2f;
				}
			}

			return baseScore + bonusOrPenalty;
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
