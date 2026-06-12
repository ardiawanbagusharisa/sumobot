using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using UnityEngine;

namespace PacingFramework
{
	/// <summary>
	/// Neural network-based decision maker for pacing-aware action selection.
	/// Learns to select actions that achieve dynamic pacing targets (Threat/Tempo balance).
	/// Unlike AIBot_NN which learns to win, PacingBrain learns to achieve specific pacing goals.
	/// </summary>
	public class PacingBrain
	{
		#region Configuration
		public bool LoadModel = true;
		public bool SaveModel = false;
		public string ModelFileName = "PacingBrain_Model";
		public string CsvLogFileName = "PacingBrain_LearningLog.csv";
		public float LearningRate = 0.01f;

		// Network architecture
		// Inputs: posX, posY, distFromCenter, angle, distance, isDashCD, isSkillCD, threatDelta, tempoDelta, actualThreat, actualTempo
		private const int INPUT_SIZE = 11;
		private const int HIDDEN_SIZE = 24;
		// Outputs: accelerateBias, turnBias, dashDesirability, skillDesirability, aggressionLevel
		private const int OUTPUT_SIZE = 5;

		private PacingNeuralNetwork NN;
		private SumoController controller;
		private PacingHandler pacingHandler;

		// Experience tracking for training
		// Note: Each segment (default 2s) is one episode
		private PacingExperience previousExperience;
		private int episodeCount = 0;  // Increments per segment, persists across rounds
		private float cumulativeReward = 0f;  // Resets per segment
		#endregion

		#region Constructor & Initialization
		public PacingBrain(SumoController controller, PacingHandler pacingHandler, bool loadModel = true, bool saveModel = false)
		{
			this.controller = controller;
			this.pacingHandler = pacingHandler;
			this.LoadModel = loadModel;
			this.SaveModel = saveModel;

			InitializeNetwork();
		}

		/// <summary>
		/// Updates the controller reference. Used when PacingBrain is reused across rounds.
		/// </summary>
		public void UpdateController(SumoController newController)
		{
			this.controller = newController;
		}

		private void InitializeNetwork()
		{
			string path = "ML/Models/PacingBrain/" + ModelFileName;

			if (LoadModel)
			{
				try
				{
					NN = PacingNeuralNetwork.Load(path);
					Logger.Info($"[PacingBrain] Loaded model from {path}");
					return;
				}
				catch (Exception e)
				{
					Logger.Warning($"[PacingBrain] Failed to load model: {e.Message}. Creating new network.");
				}
			}

			NN = new PacingNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
			Logger.Info("[PacingBrain] Created new PacingNeuralNetwork");
		}
		#endregion

		#region Action Evaluation & Selection
		/// <summary>
		/// Evaluates a candidate action and returns a score indicating how well it aligns with pacing targets.
		/// Higher score = better for achieving pacing balance.
		/// </summary>
		public float EvaluateAction(ISumoAction action, PacingEvaluation currentPacing, SumoAPI api, List<ISumoAction> previousActions)
		{
			// Get current state inputs
			float[] inputs = BuildInputs(currentPacing, api, previousActions);

			// Get NN outputs (biases and desirabilities)
			float[] outputs = NN.Forward(inputs);

			// Parse outputs
			float accelerateBias = outputs[0];      // -1 to 1: negative = reduce speed, positive = increase
			float turnBias = outputs[1];            // -1 to 1: negative = prefer defensive angles, positive = aggressive
			float dashDesirability = outputs[2];    // 0 to 1: how much we want to dash
			float skillDesirability = outputs[3];   // 0 to 1: how much we want to use skill
			float aggressionLevel = outputs[4];     // 0 to 1: overall aggression (affects all actions)

			// Score the action based on NN's current policy
			float score = ScoreActionWithPolicy(action, accelerateBias, turnBias, dashDesirability, skillDesirability, aggressionLevel);

			return score;
		}

		/// <summary>
		/// Scores an action based on the NN's learned policy.
		/// </summary>
		private float ScoreActionWithPolicy(ISumoAction action, float accelBias, float turnBias, float dashDesire, float skillDesire, float aggression)
		{
			float score = 0.5f; // Neutral baseline

			switch (action.Type)
			{
				case ActionType.Accelerate:
					AccelerateAction accel = action as AccelerateAction;
					// Favor longer acceleration if accelBias is positive, shorter if negative
					score = 0.5f + accelBias * 0.3f;
					if (accel != null)
					{
						// Longer duration = more aggressive
						score += (accel.Duration - 0.1f) / 0.9f * aggression * 0.2f;
					}
					break;

				case ActionType.TurnLeft:
				case ActionType.TurnRight:
					TurnAction turn = action as TurnAction;
					// Favor turns if turnBias suggests we need positioning
					score = 0.5f + Mathf.Abs(turnBias) * 0.2f;
					if (turn != null)
					{
						// Quick turns for aggression, longer for repositioning
						float turnAggression = 1f - (turn.Duration / 0.3f);
						score += turnAggression * aggression * 0.15f;
					}
					break;

				case ActionType.Dash:
					// Dash is aggressive and increases tempo
					score = dashDesire * 0.7f + aggression * 0.3f;
					break;

				case ActionType.SkillBoost:
				case ActionType.SkillStone:
					// Skills are high threat actions
					score = skillDesire * 0.7f + aggression * 0.3f;
					break;

				default:
					score = 0.5f;
					break;
			}

			return Mathf.Clamp01(score);
		}

		/// <summary>
		/// Selects the best action from candidates based on pacing targets.
		/// </summary>
		public ISumoAction SelectBestAction(List<ISumoAction> candidates, PacingEvaluation currentPacing, SumoAPI api, List<ISumoAction> previousActions)
		{
			if (candidates.Count == 0)
				return null;

			float bestScore = float.MinValue;
			ISumoAction bestAction = candidates[0];

			foreach (var action in candidates)
			{
				float score = EvaluateAction(action, currentPacing, api, previousActions);

				if (score > bestScore)
				{
					bestScore = score;
					bestAction = action;
				}
			}

			return bestAction;
		}
		#endregion

		#region Training & Learning
		/// <summary>
		/// Trains the network based on the outcome of the previous decision.
		/// Uses reward shaping: reward = improvement in pacing closeness to target + boundary penalties.
		/// </summary>
		public void TrainFromExperience(PacingEvaluation currentPacing)
		{
			SumoAPI api = controller.InputProvider.API;
			Vector2 currentPos = api.MyRobot.Position;
			float arenaRadius = api.BattleInfo.ArenaRadius;
			float distanceFromCenter = currentPos.magnitude / arenaRadius;

			if (previousExperience == null)
			{
				// First call, just store current state
				previousExperience = new PacingExperience
				{
					Inputs = BuildInputs(currentPacing, api, new List<ISumoAction>()),
					PacingEvaluation = currentPacing,
					Position = currentPos,
					DistanceFromCenter = distanceFromCenter
				};
				return;
			}

			// Calculate reward components
			float pacingReward = CalculatePacingReward(previousExperience.PacingEvaluation, currentPacing);
			float boundaryPenalty = CalculateBoundaryPenalty(previousExperience.DistanceFromCenter, distanceFromCenter);
			float tacticalReward = CalculateTacticalReward(previousExperience.Position, currentPos, api, currentPacing);
			float reward = boundaryPenalty + pacingReward + tacticalReward;

			cumulativeReward += reward;

			// Build target outputs based on reward shaping
			float[] currentInputs = BuildInputs(currentPacing, api, new List<ISumoAction>());
			float[] targetOutputs = BuildTargetOutputs(previousExperience.PacingEvaluation, currentPacing, reward);

			// Train the network
			NN.Train(previousExperience.Inputs, targetOutputs, LearningRate);

			// Log training data
#if UNITY_EDITOR
			if (SaveModel)
			{
				float[] outputs = NN.Forward(previousExperience.Inputs);
				// LogTraining(previousExperience.Inputs, outputs, targetOutputs, reward, currentPacing,
				// 	pacingReward, boundaryPenalty, tacticalReward, distanceFromCenter);
			}
#endif

			// Update previous experience
			previousExperience.Inputs = currentInputs;
			previousExperience.PacingEvaluation = currentPacing;
			previousExperience.Position = currentPos;
			previousExperience.DistanceFromCenter = distanceFromCenter;
		}

		/// <summary>
		/// Calculates reward based on how much closer we got to pacing targets.
		/// Positive reward = improvement, negative = got worse.
		/// </summary>
		private float CalculatePacingReward(PacingEvaluation prev, PacingEvaluation curr)
		{
			// Calculate previous error (distance from target)
			float prevError = Mathf.Abs(prev.ThreatDelta) + Mathf.Abs(prev.TempoDelta);

			// Calculate current error
			float currError = Mathf.Abs(curr.ThreatDelta) + Mathf.Abs(curr.TempoDelta);

			// Reward = reduction in error (positive = improvement)
			float improvementReward = (prevError - currError) * 10f; // Scale up for learning

			// Bonus for achieving balance (both deltas close to 0)
			float balanceBonus = 0f;
			if (Mathf.Abs(curr.ThreatDelta) < 0.1f && Mathf.Abs(curr.TempoDelta) < 0.1f)
			{
				balanceBonus = 2.0f; // Significant bonus for perfect balance
			}
			else if (Mathf.Abs(curr.ThreatDelta) < 0.2f && Mathf.Abs(curr.TempoDelta) < 0.2f)
			{
				balanceBonus = 0.5f; // Small bonus for good balance
			}

			// Penalty for extreme imbalance
			float imbalancePenalty = 0f;
			if (Mathf.Abs(curr.ThreatDelta) > 0.5f || Mathf.Abs(curr.TempoDelta) > 0.5f)
			{
				imbalancePenalty = -1.0f;
			}

			return improvementReward + balanceBonus + imbalancePenalty;
		}

		/// <summary>
		/// Calculates penalty for getting too close to arena boundaries.
		/// HEAVILY penalizes movement toward the edge, rewards staying center.
		/// </summary>
		private float CalculateBoundaryPenalty(float prevDistance, float currDistance)
		{
			// Distance is normalized (0 = center, 1 = edge)
			// MASSIVELY increased penalties to prevent going out of bounds

			// Exponential position penalty (gets extreme near edge)
			float positionPenalty = 0f;
			if (currDistance > 0.5f)
			{
				// Start penalizing at 50% of radius
				float excessDist = (currDistance - 0.5f) / 0.5f; // 0 to 1 scale
				positionPenalty = -20f * Mathf.Pow(excessDist, 3); // Cubic penalty
			}
			if (currDistance > 0.7f)
			{
				// Getting very dangerous - additional penalty
				positionPenalty -= 50f * Mathf.Pow((currDistance - 0.7f) / 0.3f, 2);
			}
			if (currDistance > 0.85f)
			{
				// Critical danger zone - MASSIVE penalty
				positionPenalty -= 200f;
			}
			if (currDistance > 0.95f)
			{
				// About to die - CATASTROPHIC penalty
				positionPenalty -= 1000f;
			}

			// Movement direction penalty/reward
			float movementDelta = currDistance - prevDistance;
			float movementPenalty = 0f;

			if (movementDelta > 0)
			{
				// Moving toward edge - scale penalty based on current position
				float basePenalty = -50f * movementDelta; // Much higher base penalty

				// Exponential scaling based on how far out we are
				if (currDistance > 0.6f)
				{
					float distFactor = Mathf.Pow((currDistance - 0.6f) / 0.4f, 2);
					basePenalty *= (1f + 10f * distFactor); // Up to 11x multiplier
				}

				movementPenalty = basePenalty;
			}
			else if (movementDelta < 0)
			{
				// Moving toward center - BIG reward, especially from danger zones
				float baseReward = 20f * Mathf.Abs(movementDelta);

				if (prevDistance > 0.7f)
				{
					// HUGE reward for escaping danger zone
					baseReward *= (1f + 5f * (prevDistance - 0.7f));
				}

				movementPenalty = baseReward;
			}

			// Safe zone bonus (central area)
			float safeZoneBonus = 0f;
			if (currDistance < 0.4f)
			{
				// Strong bonus for staying very central
				safeZoneBonus = 10.0f * (0.4f - currDistance);
			}
			else if (currDistance < 0.6f)
			{
				// Medium bonus for staying reasonably central
				safeZoneBonus = 5.0f * (0.6f - currDistance);
			}

			return positionPenalty + movementPenalty + safeZoneBonus;
		}

		/// <summary>
		/// Calculates tactical reward for smart positioning.
		/// Rewards defensive play (staying central with distance from enemy) over fleeing.
		/// Penalizes "fleeing" behavior (moving away from both enemy AND center).
		/// </summary>
		private float CalculateTacticalReward(Vector2 prevPos, Vector2 currPos, SumoAPI api, PacingEvaluation pacing)
		{
			float reward = 0f;

			Vector2 enemyPos = api.EnemyRobot.Position;
			Vector2 arenaCenter = api.BattleInfo.ArenaPosition;
			float arenaRadius = api.BattleInfo.ArenaRadius;

			// Calculate distances
			float prevDistToEnemy = (prevPos - enemyPos).magnitude;
			float currDistToEnemy = (currPos - enemyPos).magnitude;
			float prevDistToCenter = (prevPos - arenaCenter).magnitude;
			float currDistToCenter = (currPos - arenaCenter).magnitude;

			// Movement vectors
			float movementTowardEnemy = prevDistToEnemy - currDistToEnemy; // Positive = moving toward enemy
			float movementTowardCenter = prevDistToCenter - currDistToCenter; // Positive = moving toward center

			// Check if we're in "fleeing" mode (moving away from enemy AND away from center)
			bool movingAwayFromEnemy = movementTowardEnemy < -0.1f; // Moving away
			bool movingAwayFromCenter = movementTowardCenter < -0.1f; // Also moving away from center
			bool isFleeing = movingAwayFromEnemy && movingAwayFromCenter;

			// === Anti-Fleeing Penalty ===
			if (isFleeing)
			{
				// MASSIVE penalty for fleeing (running away from both enemy and center)
				// This is the suicide behavior we want to completely eliminate
				float fleeingPenalty = -100.0f * Mathf.Abs(movementTowardEnemy) * Mathf.Abs(movementTowardCenter);

				// Extra penalty if already far from center
				if (currDistToCenter > arenaRadius * 0.6f)
				{
					fleeingPenalty *= 2f; // Double penalty when already dangerous
				}

				reward += fleeingPenalty;
			}

			// === Defensive Positioning Reward ===
			// Reward maintaining good distance from enemy while staying central
			float normalizedDistToCenter = currDistToCenter / arenaRadius;
			float normalizedDistToEnemy = Mathf.Clamp01(currDistToEnemy / (arenaRadius * 0.5f));

			// "Sweet spot" for defensive play:
			// - Not too close to enemy (distToEnemy > 0.3 normalized)
			// - Not too close to edge (distFromCenter < 0.5)
			bool inDefensiveZone = normalizedDistToEnemy > 0.3f && normalizedDistToCenter < 0.5f;

			if (inDefensiveZone)
			{
				// Reward for maintaining defensive position
				float defensiveBonus = 2.0f;

				// Extra bonus if threat/tempo targets are moderate (0.4-0.6) - balanced play
				bool moderateTargets = (pacing.TargetThreat > 0.4f && pacing.TargetThreat < 0.6f) &&
				                       (pacing.TargetTempo > 0.4f && pacing.TargetTempo < 0.6f);
				if (moderateTargets)
				{
					defensiveBonus += 2.0f; // Double bonus for defensive play when targets are moderate
				}

				reward += defensiveBonus;
			}

			// === Smart Retreat Reward ===
			// If need to reduce threat/tempo, reward moving toward center (not edge)
			bool needLowerThreat = pacing.ThreatDelta > 0.1f; // Too much threat
			bool needLowerTempo = pacing.TempoDelta > 0.1f; // Too much tempo

			if ((needLowerThreat || needLowerTempo) && movementTowardCenter > 0)
			{
				// Reward for retreating toward center (smart defensive move)
				reward += 3.0f * movementTowardCenter;
			}

			// === Controlled Aggression Reward ===
			// If need to increase threat/tempo, reward moving toward enemy FROM a safe position
			bool needHigherThreat = pacing.ThreatDelta < -0.1f;
			bool needHigherTempo = pacing.TempoDelta < -0.1f;
			bool inSafePosition = normalizedDistToCenter < 0.6f; // Safe from edge

			if ((needHigherThreat || needHigherTempo) && movementTowardEnemy > 0 && inSafePosition)
			{
				// Reward for controlled aggression (approaching from safe position)
				reward += 2.0f * movementTowardEnemy;
			}

			return reward;
		}

		/// <summary>
		/// Builds target outputs for training based on pacing deltas and reward.
		/// Uses reward shaping to guide the network toward pacing targets.
		/// </summary>
		private float[] BuildTargetOutputs(PacingEvaluation prev, PacingEvaluation curr, float reward)
		{
			float[] targets = new float[OUTPUT_SIZE];

			// Analyze what we need based on current pacing
			float threatDelta = curr.ThreatDelta; // Negative = need more threat, Positive = too much threat
			float tempoDelta = curr.TempoDelta;   // Negative = need more tempo, Positive = too much tempo

			// Target 0: accelerateBias (-1 to 1)
			// If tempo is low OR threat is low, encourage acceleration
			float needAccel = (-threatDelta - tempoDelta) / 2f;
			targets[0] = Mathf.Clamp(needAccel, -1f, 1f);

			// Target 1: turnBias (-1 to 1)
			// If threat is low, encourage aggressive positioning (positive)
			// If threat is high, encourage defensive positioning (negative)
			targets[1] = Mathf.Clamp(-threatDelta, -1f, 1f);

			// Target 2: dashDesirability (0 to 1)
			// Dash increases both threat and tempo
			float needDash = (-threatDelta - tempoDelta) / 2f;
			targets[2] = Mathf.Clamp01(needDash);

			// Target 3: skillDesirability (0 to 1)
			// Skills are high threat actions
			targets[3] = Mathf.Clamp01(-threatDelta);

			// Target 4: aggressionLevel (0 to 1)
			// Overall aggression based on how much we need threat/tempo
			float needAggression = (-threatDelta * 0.6f - tempoDelta * 0.4f);
			targets[4] = Mathf.Clamp01(needAggression);

			// Apply reward influence: if reward is positive, reinforce current behavior
			// if reward is negative, push away from current behavior
			if (reward > 0 && previousExperience != null)
			{
				// Good reward: keep similar targets
				float[] currentOutputs = NN.Forward(previousExperience.Inputs);
				for (int i = 0; i < OUTPUT_SIZE; i++)
				{
					targets[i] = Mathf.Lerp(targets[i], currentOutputs[i], 0.3f);
				}
			}

			return targets;
		}

		/// <summary>
		/// Builds input vector from current game state and pacing evaluation.
		/// </summary>
		private float[] BuildInputs(PacingEvaluation pacing, SumoAPI api, List<ISumoAction> previousActions)
		{
			// Simulate current position if there are previous actions
			var (currentPos, currentRot) = previousActions.Count > 0
				? api.Simulate(previousActions)
				: (api.MyRobot.Position, api.MyRobot.Rotation);

			float posX = currentPos.x / api.BattleInfo.ArenaRadius;
			float posY = currentPos.y / api.BattleInfo.ArenaRadius;

			// CRITICAL: Add explicit distance from center as input!
			// This makes it much easier for the network to learn boundary avoidance
			float distFromCenter = currentPos.magnitude / api.BattleInfo.ArenaRadius;  // 0 = center, 1 = edge

			float angle = api.Angle(currentPos, currentRot, api.EnemyRobot.Position, normalized: true);
			float distance = api.DistanceNormalized(currentPos, api.EnemyRobot.Position);
			float isDashCD = api.MyRobot.IsDashOnCooldown ? 1f : 0f;
			float isSkillCD = api.MyRobot.Skill.IsSkillOnCooldown ? 1f : 0f;

			// Pacing deltas (key inputs for learning pacing-aware behavior!)
			float threatDelta = pacing.ThreatDelta;
			float tempoDelta = pacing.TempoDelta;

			// Actual pacing values (for context)
			float actualThreat = pacing.ActualThreat;
			float actualTempo = pacing.ActualTempo;

			return new float[]
			{
				posX,
				posY,
				distFromCenter,
				angle,
				distance,
				isDashCD,
				isSkillCD,
				threatDelta,    // Critical: tells NN how much threat adjustment needed
				tempoDelta,     // Critical: tells NN how much tempo adjustment needed
				actualThreat,
				actualTempo
			};
		}
		#endregion

		#region Persistence & Logging
		public void SaveModelToDisk()
		{
#if UNITY_EDITOR
			if (SaveModel)
			{
				string path = "Assets/Resources/ML/Models/PacingBrain/" + ModelFileName + ".json";
				NN.Save(path);
				Logger.Info($"[PacingBrain] Saved model to {path}");
			}
#endif
		}

		private void LogTraining(float[] inputs, float[] outputs, float[] targets, float reward, PacingEvaluation pacing,
			float pacingReward, float boundaryPenalty, float tacticalReward, float distanceFromCenter)
		{
			string csvPath = "ML/Models/PacingBrain/" + CsvLogFileName;
			string path = Path.Combine(Application.streamingAssetsPath, csvPath);
			bool writeHeader = !File.Exists(path);

			using (StreamWriter sw = new StreamWriter(path, true))
			{
				if (writeHeader)
				{
					sw.WriteLine("Episode,CumulativeReward,TotalReward,PacingReward,BoundaryPenalty,TacticalReward,DistanceFromCenter,ThreatDelta,TempoDelta," +
					            "In_PosX,In_PosY,In_Angle,In_Dist,In_DashCD,In_SkillCD,In_ThreatDelta,In_TempoDelta,In_ActualThreat,In_ActualTempo," +
					            "Out_AccelBias,Out_TurnBias,Out_DashDesire,Out_SkillDesire,Out_Aggression," +
					            "Tgt_AccelBias,Tgt_TurnBias,Tgt_DashDesire,Tgt_SkillDesire,Tgt_Aggression," +
					            "Loss");
				}

				float loss = CalculateLoss(outputs, targets);

				sw.WriteLine($"{episodeCount},{cumulativeReward:F3},{reward:F3},{pacingReward:F3},{boundaryPenalty:F3},{tacticalReward:F3},{distanceFromCenter:F3},{pacing.ThreatDelta:F3},{pacing.TempoDelta:F3}," +
				            $"{inputs[0]:F4},{inputs[1]:F4},{inputs[2]:F4},{inputs[3]:F4},{inputs[4]:F4},{inputs[5]:F4},{inputs[6]:F4},{inputs[7]:F4},{inputs[8]:F4},{inputs[9]:F4}," +
				            $"{outputs[0]:F4},{outputs[1]:F4},{outputs[2]:F4},{outputs[3]:F4},{outputs[4]:F4}," +
				            $"{targets[0]:F4},{targets[1]:F4},{targets[2]:F4},{targets[3]:F4},{targets[4]:F4}," +
				            $"{loss:F6}");
			}
		}

		private float CalculateLoss(float[] outputs, float[] targets)
		{
			float loss = 0f;
			int len = Math.Min(outputs.Length, targets.Length);
			for (int i = 0; i < len; i++)
			{
				float diff = outputs[i] - targets[i];
				loss += diff * diff;
			}
			return loss / len;
		}

		/// <summary>
		/// Called when an episode (segment) ends.
		/// Increments episode counter and resets per-episode tracking.
		/// Note: Episode = one segment (default 2s of gameplay)
		/// IMPORTANT: We do NOT reset previousExperience here - learning must be continuous within a round!
		/// </summary>
		public void OnEpisodeEnd()
		{
			episodeCount++;
			cumulativeReward = 0f;
			// DO NOT reset previousExperience - it breaks learning continuity!
			// previousExperience = null;
		}

		/// <summary>
		/// Called when a new round starts. Resets experience for fresh start.
		/// </summary>
		public void OnRoundStart()
		{
			previousExperience = null;
			cumulativeReward = 0f;
		}

		public int GetEpisodeCount()
		{
			return episodeCount;
		}
		#endregion
	}

	#region Supporting Classes
	/// <summary>
	/// Stores experience for training.
	/// </summary>
	public class PacingExperience
	{
		public float[] Inputs;
		public PacingEvaluation PacingEvaluation;
		public Vector2 Position;  // Track position for boundary checking
		public float DistanceFromCenter; // Distance from arena center (normalized)
	}

	/// <summary>
	/// Simple neural network for pacing decisions.
	/// Similar to NeuralNetwork in AIBot_NN but separate for pacing-specific learning.
	/// </summary>
	public class PacingNeuralNetwork
	{
		private float[,] weights1; // Input to hidden
		private float[] bias1;
		private float[,] weights2; // Hidden to output
		private float[] bias2;

		private int inputSize;
		private int hiddenSize;
		private int outputSize;

		[Serializable]
		private class NetworkData
		{
			public int inputSize, hiddenSize, outputSize;
			public float[] w1, w2, b1, b2;
		}

		public PacingNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
		{
			this.inputSize = inputSize;
			this.hiddenSize = hiddenSize;
			this.outputSize = outputSize;

			weights1 = new float[inputSize, hiddenSize];
			bias1 = new float[hiddenSize];
			weights2 = new float[hiddenSize, outputSize];
			bias2 = new float[outputSize];

			Randomize(weights1);
			Randomize(weights2);
		}

		private void Randomize(float[,] weights)
		{
			System.Random rand = new System.Random();
			for (int i = 0; i < weights.GetLength(0); i++)
				for (int j = 0; j < weights.GetLength(1); j++)
					weights[i, j] = (float)(rand.NextDouble() * 2 - 1) * 0.5f; // Smaller initial weights
		}

		public float[] Forward(float[] input)
		{
			// Hidden layer with Tanh activation
			float[] hidden = new float[hiddenSize];
			for (int i = 0; i < hiddenSize; i++)
			{
				hidden[i] = bias1[i];
				for (int j = 0; j < inputSize; j++)
					hidden[i] += input[j] * weights1[j, i];
				hidden[i] = (float)Math.Tanh(hidden[i]);
			}

			// Output layer with Tanh activation
			float[] output = new float[outputSize];
			for (int i = 0; i < outputSize; i++)
			{
				output[i] = bias2[i];
				for (int j = 0; j < hiddenSize; j++)
					output[i] += hidden[j] * weights2[j, i];
				output[i] = (float)Math.Tanh(output[i]); // -1 to 1 range
			}

			return output;
		}

		public void Train(float[] input, float[] target, float learningRate)
		{
			// Forward pass
			float[] hidden = new float[hiddenSize];
			for (int i = 0; i < hiddenSize; i++)
			{
				hidden[i] = bias1[i];
				for (int j = 0; j < inputSize; j++)
					hidden[i] += input[j] * weights1[j, i];
				hidden[i] = (float)Math.Tanh(hidden[i]);
			}

			float[] output = new float[outputSize];
			for (int i = 0; i < outputSize; i++)
			{
				output[i] = bias2[i];
				for (int j = 0; j < hiddenSize; j++)
					output[i] += hidden[j] * weights2[j, i];
				output[i] = (float)Math.Tanh(output[i]);
			}

			// Backpropagation
			float[] dOutput = new float[outputSize];
			for (int i = 0; i < outputSize; i++)
				dOutput[i] = (output[i] - target[i]) * (1 - output[i] * output[i]);

			float[] dHidden = new float[hiddenSize];
			for (int i = 0; i < hiddenSize; i++)
			{
				float error = 0f;
				for (int j = 0; j < outputSize; j++)
					error += dOutput[j] * weights2[i, j];
				dHidden[i] = error * (1 - hidden[i] * hidden[i]);
			}

			// Update weights
			for (int i = 0; i < hiddenSize; i++)
				for (int j = 0; j < outputSize; j++)
					weights2[i, j] -= learningRate * dOutput[j] * hidden[i];

			for (int i = 0; i < outputSize; i++)
				bias2[i] -= learningRate * dOutput[i];

			for (int i = 0; i < inputSize; i++)
				for (int j = 0; j < hiddenSize; j++)
					weights1[i, j] -= learningRate * dHidden[j] * input[i];

			for (int i = 0; i < hiddenSize; i++)
				bias1[i] -= learningRate * dHidden[i];
		}

		public void Save(string path)
		{
			NetworkData d = new NetworkData
			{
				inputSize = inputSize,
				hiddenSize = hiddenSize,
				outputSize = outputSize,
				w1 = Flatten(weights1),
				w2 = Flatten(weights2),
				b1 = (float[])bias1.Clone(),
				b2 = (float[])bias2.Clone()
			};
			string json = JsonUtility.ToJson(d, true);
			File.WriteAllText(path, json);
		}

		public static PacingNeuralNetwork Load(string path)
		{
			TextAsset model = Resources.Load<TextAsset>(path);
			if (model == null)
				throw new FileNotFoundException($"PacingBrain model not found at {path}");

			NetworkData d = JsonUtility.FromJson<NetworkData>(model.text);
			PacingNeuralNetwork nn = new PacingNeuralNetwork(d.inputSize, d.hiddenSize, d.outputSize);

			nn.Unflatten(d.w1, nn.weights1);
			nn.Unflatten(d.w2, nn.weights2);
			nn.bias1 = (float[])d.b1.Clone();
			nn.bias2 = (float[])d.b2.Clone();
			return nn;
		}

		private float[] Flatten(float[,] m)
		{
			int rows = m.GetLength(0), cols = m.GetLength(1);
			float[] flat = new float[rows * cols];
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					flat[i * cols + j] = m[i, j];
			return flat;
		}

		private void Unflatten(float[] flat, float[,] m)
		{
			int rows = m.GetLength(0), cols = m.GetLength(1);
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					m[i, j] = flat[i * cols + j];
		}
	}
	#endregion
}
