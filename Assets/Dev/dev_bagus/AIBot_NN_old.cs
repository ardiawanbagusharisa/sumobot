//using SumoCore;
//using SumoInput;
//using SumoManager;
//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using UnityEngine;

//namespace SumoBot
//{
//    public class AIBot_NN : Bot
//    {
//        #region Runtime Properties
//        public override string ID => "Bot_NN";
//        public override SkillType SkillType => SkillType.Boost;
//        #endregion

//        #region NN Properties
//        [Header("Neural Network Hyperparameters")]
//        public float learningRate = 0.005f; // Increased from 0.001f for faster learning
//        public float learningRateDecay = 0.999f; // NEW: Decay per episode
//        public float discountFactor = 0.99f; // Gamma for future rewards
//        public float epsilon = 0.05f; // Reduced from 1.0f for less random actions
//        public float epsilonDecay = 0.999f;
//        public float minEpsilon = 0.01f;
//        public float trainingInterval = 0.5f;
//        public float decisionInterval = 0.2f;
//        public float explorationNoise = 0.1f; // NEW: Noise for Q-value exploration
//        public int maxTrainingEpisodes = 1000; // NEW: Stop training after this many episodes
//        public float minQValueVariance = 0.1f; // NEW: Stop training if Q-value variance is too low

//        [Header("Replay Buffer")]
//        public int replayBufferSize = 10000;
//        public int batchSize = 64;
//        public int minExpLearn = 500;
//        #endregion

//        #region Model Properties
//        [Header("Model Saving/Loading")]
//        public bool loadModel = true;
//        public bool trainModel = true;
//        public string modelFileName = "NN_Model.json";
//        private string modelFilePath;
//        private string csvFilePath;
//        #endregion

//        #region Private Properties
//        private const int INPUT_SIZE = 5; // x, y, distance to enemy, enemy x, enemy y
//        private const int OUTPUT_SIZE = 5; // Accelerate, TurnLeft, TurnRight, Dash, Skill
//        private const int HIDDEN_SIZE = 128;

//        private NeuralNetwork qNetwork;
//        private ReplayBuffer replayBuffer;
//        private SumoAPI api;
//        private PlayerSide mySide;
//        private float[] lastStates;
//        private bool[] lastActions;
//        private float tempReward = 0f;
//        private float episodeReward = 0f;
//        private int episodeCounter = 0;
//        private float decisionTimer = 0f;
//        private float trainingTimer = 0f;
//        private float currentLearningRate;
//        private float[] avgQValues = new float[OUTPUT_SIZE]; // NEW: Track average Q-values
//        private int actionCount = 0;
//        #endregion

//        #region Bot Methods
//        public override void OnBotInit(SumoAPI botAPI)
//        {
//            api = botAPI;
//            mySide = api.MyRobot.Side;
//            replayBuffer = new ReplayBuffer(replayBufferSize);
//            modelFilePath = Path.Combine(Application.persistentDataPath, modelFileName);
//            csvFilePath = Path.Combine(Application.persistentDataPath, "training_log.csv");

//            if (loadModel && File.Exists(modelFilePath))
//            {
//                LoadModel();
//                Debug.Log($"Loaded pre-trained model from: {modelFilePath}");
//                epsilon = minEpsilon; // Start with low exploration for loaded model
//            }
//            else
//            {
//                qNetwork = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
//                Debug.Log("Initialized new model. Starting training from scratch.");
//                epsilon = 0.05f; // Lower initial epsilon
//            }

//            lastStates = null;
//            currentLearningRate = learningRate;
//            Debug.Log($"NN Bot Initialized. Side: {mySide}, Epsilon: {epsilon:F4}");
//        }

//        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
//        {
//            if (state == BattleState.Battle_End)
//            {
//                // Log episode data
//                int win = (winner.ToString() == mySide.ToString()) ? 1 : 0;
//                LogEpisodeData(episodeCounter, episodeReward, win);
//                episodeCounter++;
//                episodeReward = 0f;

//                // Extra reinforcement for winning
//                if (win == 1 && replayBuffer.Count > minExpLearn)
//                {
//                    List<Experience> extraBatch = replayBuffer.Sample(batchSize);
//                    for (int i = 0; i < 5; i++)
//                    {
//                        TrainNetwork(extraBatch);
//                    }
//                    Debug.Log($"[Extra Training] Reinforced win in episode {episodeCounter - 1} with {extraBatch.Count} samples.");
//                }

//                // Regular training
//                if (replayBuffer.Count > minExpLearn)
//                {
//                    List<Experience> batch = replayBuffer.Sample(batchSize);
//                    if (batch.Count > 0)
//                        TrainNetwork(batch);
//                }

//                if (trainModel)
//                    SaveModel();

//                lastStates = null;
//                epsilon = Mathf.Max(minEpsilon, epsilon * epsilonDecay);
//                currentLearningRate *= learningRateDecay; // Decay learning rate
//                Debug.Log($"Episode {episodeCounter}, New Epsilon: {epsilon:F4}, New Learning Rate: {currentLearningRate:F6}");

//                // Reset average Q-values periodically
//                if (episodeCounter % 100 == 0)
//                {
//                    for (int i = 0; i < avgQValues.Length; i++)
//                        avgQValues[i] = 0f;
//                    actionCount = 0;
//                }
//            }
//        }

//        public override void OnBotCollision(BounceEvent param)
//        {
//            float speedColThreshold = 0.05f;
//            SumoBotAPI myRobot = api.MyRobot;
//            SumoBotAPI enemyRobot = api.EnemyRobot;

//            if (myRobot.LinearVelocity.magnitude - speedColThreshold > enemyRobot.LinearVelocity.magnitude)
//                tempReward += 5f;
//            else if (myRobot.LinearVelocity.magnitude < enemyRobot.LinearVelocity.magnitude - speedColThreshold)
//                tempReward -= 1f;
//            else
//                tempReward += 0f;

//            Debug.Log($"Collision detected! Temp Reward: {tempReward:F2} (My Speed: {myRobot.LinearVelocity.magnitude:F2}, Enemy Speed: {enemyRobot.LinearVelocity.magnitude:F2})");
//        }

//        public override void OnBotUpdate()
//        {
//            ClearCommands();
//            SumoBotAPI myRobot = api.MyRobot;
//            SumoBotAPI enemyRobot = api.EnemyRobot;

//            tempReward += CalculateReward(myRobot, enemyRobot);
//            decisionTimer += Time.deltaTime;

//            if (decisionTimer >= decisionInterval)
//            {
//                decisionTimer = 0f;
//                float[] currentStates = GatherInputs(myRobot, enemyRobot);

//                if (lastStates != null)
//                {
//                    bool done = IsEpisodeDone();
//                    if (done)
//                    {
//                        tempReward += (myRobot.Position - api.BattleInfo.ArenaPosition).magnitude <
//                            (enemyRobot.Position - api.BattleInfo.ArenaPosition).magnitude ? 10f : -10f; // Increased win/loss reward
//                    }

//                    replayBuffer.AddExperience(new Experience
//                    {
//                        state = lastStates,
//                        actionsTaken = lastActions,
//                        reward = tempReward,
//                        nextState = currentStates,
//                        done = done
//                    });

//                    episodeReward += tempReward;
//                    tempReward = 0f;
//                }

//                bool[] currentActions = ChooseActions(currentStates);
//                lastActions = currentActions;
//                lastStates = currentStates;
//            }

//            if (lastActions != null)
//            {
//                ExecuteActions(lastActions, myRobot);
//            }

//            trainingTimer += Time.deltaTime;
//            if (trainingTimer >= trainingInterval && replayBuffer.Count > minExpLearn)
//            {
//                trainingTimer = 0f;
//                List<Experience> batch = replayBuffer.Sample(batchSize);
//                if (batch.Count > 0)
//                    TrainNetwork(batch);
//            }

//            Submit();

//            // Debug logging
//            string actionsStr = lastActions != null ? string.Join(", ", lastActions.Select((a, i) => $"{i}: {(a ? "1" : "0")}")) : "None";
//            float[] qValues = lastStates != null ? qNetwork.FeedForward(lastStates) : new float[OUTPUT_SIZE];
//            float qVariance = CalculateQValueVariance(qValues);
//            string qValuesStr = string.Join(", ", qValues.Select(q => q.ToString("F3")));
//            Debug.Log($"Episode: {episodeCounter}, Reward: {episodeReward:F2}, Epsilon: {epsilon:F4}, Buffer Size: {replayBuffer.Count}, Actions: {actionsStr}, Q-Values: {qValuesStr}, Q-Variance: {qVariance:F3}");

//            // Warn if Q-value variance is too low
//            if (qVariance < minQValueVariance && episodeCounter > 100)
//                Debug.LogWarning($"Q-value variance {qVariance:F3} below threshold {minQValueVariance}. Consider stopping training to avoid overfitting.");
//        }

//        private float CalculateReward(SumoBotAPI myRobot, SumoBotAPI enemyRobot)
//        {
//            Vector2 center = api.BattleInfo.ArenaPosition;
//            float arenaRadius = api.BattleInfo.ArenaRadius;
//            float distToCenter = (myRobot.Position - center).magnitude / arenaRadius;
//            float distToEnemy = (enemyRobot.Position - myRobot.Position).magnitude / arenaRadius;
//            float lastDistToEnemy = lastStates != null ? lastStates[2] * 2 * arenaRadius : distToEnemy;
//            float enemyDistToCenter = (enemyRobot.Position - center).magnitude / arenaRadius;
//            float angleToEnemy = api.Angle(myRobot.Position, myRobot.Rotation, enemyRobot.Position, false);

//            float reward = 0f;
//            reward += (lastDistToEnemy - distToEnemy) * 15f; // Increased from 0.01f to strongly favor closing distance
//            reward += (1f - distToCenter) * 0.1f; // Increased from 0.05f for staying near center
//            reward += enemyDistToCenter * 0.1f; // Increased from 0.05f for pushing enemy to edge
//            if (distToCenter > 0.9f)
//                reward -= 0.2f; // Increased penalty for being near edge
//            if (!myRobot.IsDashOnCooldown && angleToEnemy < 10f)
//                reward += 0.2f; // Increased Dash bonus
//            if (!myRobot.Skill.IsSkillOnCooldown && angleToEnemy < 10f)
//                reward += 0.2f; // Increased Skill bonus
//            reward += (myRobot.LinearVelocity.magnitude - enemyRobot.LinearVelocity.magnitude) * 0.05f; // Increased velocity reward

//            Vector2 distToEnemyNormalized = (enemyRobot.Position - myRobot.Position).normalized;
//            Vector2 myForward = new Vector2(Mathf.Cos(myRobot.Rotation * Mathf.Deg2Rad), Mathf.Sin(myRobot.Rotation * Mathf.Deg2Rad));
//            float facingDot = Vector2.Dot(distToEnemyNormalized, myForward);
//            if (facingDot > 0.9f && distToEnemy < 0.4f)
//                reward += 0.5f; // Increased chasing reward

//            return reward;
//        }
//        #endregion

//        #region NN and RL Methods
//        private float[] GatherInputs(SumoBotAPI myRobot, SumoBotAPI enemyRobot)
//        {
//            float[] inputs = new float[INPUT_SIZE];
//            float arenaRadius = api.BattleInfo.ArenaRadius;
//            float myRotationRad = myRobot.Rotation * Mathf.Deg2Rad;
//            Vector2 myForward = new Vector2(Mathf.Cos(myRotationRad), Mathf.Sin(myRotationRad));
//            Vector2 toEnemy = (enemyRobot.Position - myRobot.Position).normalized;
//            float angleToEnemy = Vector2.SignedAngle(myForward, toEnemy) * Mathf.Deg2Rad;

//            inputs[0] = myRobot.Position.x / arenaRadius;
//            inputs[1] = myRobot.Position.y / arenaRadius;
//            inputs[2] = Vector2.Distance(myRobot.Position, enemyRobot.Position) / (2 * arenaRadius);
//            inputs[3] = angleToEnemy;
//            inputs[4] = enemyRobot.Position.x / arenaRadius;

//            return inputs;
//        }

//        private bool[] ChooseActions(float[] state)
//        {
//            bool[] actions = new bool[OUTPUT_SIZE];
//            float[] qValues = qNetwork.FeedForward(state);

//            // Add exploration noise to Q-values
//            for (int i = 0; i < qValues.Length; i++)
//                qValues[i] += UnityEngine.Random.Range(-explorationNoise, explorationNoise);

//            if (UnityEngine.Random.value < epsilon)
//            {
//                for (int i = 0; i < OUTPUT_SIZE; i++)
//                    actions[i] = UnityEngine.Random.value > 0.5f;
//            }
//            else
//            {
//                for (int i = 0; i < OUTPUT_SIZE; i++)
//                    actions[i] = qValues[i] > 0f;
//            }

//            // Update average Q-values
//            for (int i = 0; i < qValues.Length; i++)
//                avgQValues[i] = (avgQValues[i] * actionCount + qValues[i]) / (actionCount + 1);
//            actionCount++;

//            // Prevent conflicting turn actions
//            if (actions[1] && actions[2])
//            {
//                if (qValues[1] > qValues[2])
//                    actions[2] = false;
//                else
//                    actions[1] = false;
//            }

//            return actions;
//        }

//        private void ExecuteActions(bool[] actions, SumoBotAPI myRobot)
//        {
//            if (actions[0])
//                Enqueue(new AccelerateAction(InputType.Script));
//            if (actions[1])
//                Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft));
//            if (actions[2])
//                Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight));
//            if (actions[3] && !myRobot.IsDashOnCooldown)
//                Enqueue(new DashAction(InputType.Script));
//            if (actions[4] && !myRobot.Skill.IsSkillOnCooldown)
//                Enqueue(new SkillAction(InputType.Script));
//        }

//        private bool IsEpisodeDone()
//        {
//            return api.BattleInfo.CurrentState == BattleState.Battle_End;
//        }

//        private void TrainNetwork(List<Experience> batch)
//        {
//            if (episodeCounter >= maxTrainingEpisodes)
//            {
//                Debug.LogWarning($"Training stopped: Reached max episodes {maxTrainingEpisodes}");
//                return;
//            }

//            foreach (Experience exp in batch)
//            {
//                float[] currentQ = qNetwork.FeedForward(exp.state);
//                float[] targetQ = (float[])currentQ.Clone();

//                for (int i = 0; i < OUTPUT_SIZE; i++)
//                {
//                    if (exp.actionsTaken[i])
//                    {
//                        float target = exp.reward;
//                        if (!exp.done)
//                        {
//                            float[] nextQ = qNetwork.FeedForward(exp.nextState);
//                            target += discountFactor * nextQ.Max();
//                        }
//                        targetQ[i] = target;
//                    }
//                }

//                qNetwork.Train(exp.state, targetQ, currentLearningRate);
//            }

//            // Check Q-value variance for early stopping
//            float qVariance = CalculateQValueVariance(avgQValues);
//            if (qVariance < minQValueVariance && episodeCounter > 100)
//                Debug.LogWarning($"Low Q-value variance {qVariance:F3}. Training may be overfitting.");
//        }

//        private float CalculateQValueVariance(float[] qValues)
//        {
//            float mean = qValues.Average();
//            float variance = qValues.Sum(q => (q - mean) * (q - mean)) / qValues.Length;
//            return variance;
//        }

//        private void SaveModel()
//        {
//            try
//            {
//                SerializableNeuralNetworkData dataToSave = qNetwork.ToSerializableData();
//                string json = JsonUtility.ToJson(dataToSave);
//                File.WriteAllText(modelFilePath, json);
//                Debug.Log($"Model saved successfully to: {modelFilePath}");
//            }
//            catch (Exception e)
//            {
//                Debug.LogError($"Failed to save model: {e.Message}");
//            }
//        }

//        private void LoadModel()
//        {
//            try
//            {
//                string json = File.ReadAllText(modelFilePath);
//                SerializableNeuralNetworkData loadedData = JsonUtility.FromJson<SerializableNeuralNetworkData>(json);

//                if (loadedData != null)
//                {
//                    qNetwork = new NeuralNetwork(loadedData.inputSize, loadedData.hiddenSize, loadedData.outputSize);
//                    qNetwork.LoadFromSerializableData(loadedData);
//                    Debug.Log($"Model loaded successfully from: {modelFilePath}");
//                }
//                else
//                {
//                    throw new Exception("JsonUtility returned null during deserialization.");
//                }
//            }
//            catch (Exception e)
//            {
//                Debug.LogError($"Failed to load model: {e.Message}");
//                qNetwork = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
//            }
//        }

//        private void LogEpisodeData(int episode, float reward, int win)
//        {
//            string extendedCsvPath = Path.Combine(Application.persistentDataPath, "training_log_detailed.csv");
//            bool writeHeader = !File.Exists(extendedCsvPath);

//            try
//            {
//                using (StreamWriter sw = new StreamWriter(extendedCsvPath, true))
//                {
//                    if (writeHeader)
//                    {
//                        sw.WriteLine("Episode,Reward,Win,Step,ActionTaken,QValues,State,NextState,Done");
//                    }

//                    int step = 0;
//                    foreach (Experience exp in replayBuffer.Sample(replayBuffer.Count))
//                    {
//                        string actionTaken = string.Join("|", exp.actionsTaken.Select(a => a ? "1" : "0"));
//                        string qVals = string.Join("|", qNetwork.FeedForward(exp.state).Select(q => q.ToString("F4")));
//                        string stateStr = string.Join("|", exp.state.Select(s => s.ToString("F3")));
//                        string nextStateStr = exp.nextState != null ? string.Join("|", exp.nextState.Select(s => s.ToString("F3"))) : "null";
//                        string doneStr = exp.done ? "1" : "0";

//                        sw.WriteLine($"{episode},{reward:F2},{win},{step},{actionTaken},{qVals},{stateStr},{nextStateStr},{doneStr}");
//                        step++;
//                    }
//                }
//            }
//            catch (Exception e)
//            {
//                Debug.LogError($"Failed to write extended CSV log: {e.Message}");
//            }
//        }
//        #endregion

//        #region Neural Network Core Components
//        public static class Activation
//        {
//            public static float ReLU(float x) => MathF.Max(0, x);
//            public static float ReluDerivative(float x) => x > 0 ? 1 : 0;
//            public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
//            public static float SigmoidDerivative(float x) => Sigmoid(x) * (1 - Sigmoid(x));
//        }

//        public class SerializableNeuralNetworkData
//        {
//            public int inputSize;
//            public int hiddenSize;
//            public int outputSize;
//            public float[] weightsInputHiddenFlat;
//            public float[] biasesHidden;
//            public float[] weightsHiddenOutputFlat;
//            public float[] biasesOutput;
//        }

//        [Serializable]
//        public class NeuralNetwork
//        {
//            private int inputSize;
//            private int hiddenSize;
//            private int outputSize;
//            private float[,] weightsInput;
//            private float[,] weightsOutput;
//            private float[] biasesHidden;
//            private float[] biasesOutput;

//            public NeuralNetwork(int input, int hidden, int output)
//            {
//                inputSize = input;
//                hiddenSize = hidden;
//                outputSize = output;
//                weightsInput = new float[inputSize, hiddenSize];
//                weightsOutput = new float[hiddenSize, outputSize];
//                biasesHidden = new float[hiddenSize];
//                biasesOutput = new float[outputSize];
//                InitializeWeights();
//            }

//            private void InitializeWeights()
//            {
//                float limitInput = Mathf.Sqrt(2f / inputSize);
//                float limitHidden = Mathf.Sqrt(2f / hiddenSize);

//                for (int i = 0; i < inputSize; i++)
//                    for (int j = 0; j < hiddenSize; j++)
//                        weightsInput[i, j] = UnityEngine.Random.Range(-limitInput, limitInput);

//                for (int i = 0; i < hiddenSize; i++)
//                    biasesHidden[i] = 0.01f;

//                for (int i = 0; i < hiddenSize; i++)
//                    for (int j = 0; j < outputSize; j++)
//                        weightsOutput[i, j] = UnityEngine.Random.Range(-limitHidden, limitHidden);

//                for (int i = 0; i < outputSize; i++)
//                    biasesOutput[i] = 0.01f;
//            }

//            public float[] FeedForward(float[] inputs)
//            {
//                if (inputs.Length != inputSize)
//                {
//                    Debug.LogError($"Input array size mismatch! Expected {inputSize}, got {inputs.Length}");
//                    return new float[outputSize];
//                }

//                float[] hiddenOutputs = new float[hiddenSize];
//                for (int i = 0; i < hiddenSize; i++)
//                {
//                    float sum = biasesHidden[i];
//                    for (int j = 0; j < inputSize; j++)
//                        sum += inputs[j] * weightsInput[j, i];
//                    hiddenOutputs[i] = Activation.ReLU(sum);
//                }

//                float[] finalOutputs = new float[outputSize];
//                for (int i = 0; i < outputSize; i++)
//                {
//                    float sum = biasesOutput[i];
//                    for (int j = 0; j < hiddenSize; j++)
//                        sum += hiddenOutputs[j] * weightsOutput[j, i];
//                    finalOutputs[i] = sum; // Linear output
//                }

//                return finalOutputs;
//            }

//            public void Train(float[] inputs, float[] targetOutputs, float learningRate)
//            {
//                float[] hiddenInputsRaw = new float[hiddenSize];
//                float[] hiddenOutputs = new float[hiddenSize];

//                for (int i = 0; i < hiddenSize; i++)
//                {
//                    float sum = biasesHidden[i];
//                    for (int j = 0; j < inputSize; j++)
//                        sum += inputs[j] * weightsInput[j, i];
//                    hiddenInputsRaw[i] = sum;
//                    hiddenOutputs[i] = Activation.ReLU(sum);
//                }

//                float[] outputInputsRaw = new float[outputSize];
//                float[] actualOutputs = new float[outputSize];
//                for (int i = 0; i < outputSize; i++)
//                {
//                    float sum = biasesOutput[i];
//                    for (int j = 0; j < hiddenSize; j++)
//                        sum += hiddenOutputs[j] * weightsOutput[j, i];
//                    outputInputsRaw[i] = sum;
//                    actualOutputs[i] = sum;
//                }

//                float[] outputErrors = new float[outputSize];
//                for (int i = 0; i < outputSize; i++)
//                    outputErrors[i] = targetOutputs[i] - actualOutputs[i];

//                float[] hiddenErrors = new float[hiddenSize];
//                for (int i = 0; i < hiddenSize; i++)
//                {
//                    float error = 0f;
//                    for (int j = 0; j < outputSize; j++)
//                        error += outputErrors[j] * weightsOutput[i, j];
//                    hiddenErrors[i] = error * Activation.ReluDerivative(hiddenInputsRaw[i]);
//                }

//                // Gradient clipping
//                for (int i = 0; i < hiddenSize; i++)
//                    for (int j = 0; j < outputSize; j++)
//                        weightsOutput[i, j] -= learningRate * Mathf.Clamp(outputErrors[j] * hiddenOutputs[i], -0.1f, 0.1f);

//                for (int i = 0; i < outputSize; i++)
//                    biasesOutput[i] -= learningRate * Mathf.Clamp(outputErrors[i], -0.1f, 0.1f);

//                for (int i = 0; i < inputSize; i++)
//                    for (int j = 0; j < hiddenSize; j++)
//                        weightsInput[i, j] -= learningRate * Mathf.Clamp(hiddenErrors[j] * inputs[i], -0.1f, 0.1f);

//                for (int i = 0; i < hiddenSize; i++)
//                    biasesHidden[i] -= learningRate * Mathf.Clamp(hiddenErrors[i], -0.1f, 0.1f);
//            }

//            public SerializableNeuralNetworkData ToSerializableData()
//            {
//                return new SerializableNeuralNetworkData
//                {
//                    inputSize = this.inputSize,
//                    hiddenSize = this.hiddenSize,
//                    outputSize = this.outputSize,
//                    weightsInputHiddenFlat = Flatten(weightsInput),
//                    biasesHidden = this.biasesHidden,
//                    weightsHiddenOutputFlat = Flatten(weightsOutput),
//                    biasesOutput = this.biasesOutput
//                };
//            }

//            public void LoadFromSerializableData(SerializableNeuralNetworkData data)
//            {
//                if (data.inputSize != this.inputSize || data.hiddenSize != this.hiddenSize || data.outputSize != this.outputSize)
//                {
//                    Debug.LogError("Model architecture mismatch during loading! Reinitializing with new architecture.");
//                    this.inputSize = data.inputSize;
//                    this.hiddenSize = data.hiddenSize;
//                    this.outputSize = data.outputSize;
//                    weightsInput = new float[inputSize, hiddenSize];
//                    weightsOutput = new float[hiddenSize, outputSize];
//                    biasesHidden = new float[hiddenSize];
//                    biasesOutput = new float[outputSize];
//                }

//                weightsInput = Reshape(data.weightsInputHiddenFlat, inputSize, hiddenSize);
//                biasesHidden = data.biasesHidden;
//                weightsOutput = Reshape(data.weightsHiddenOutputFlat, hiddenSize, outputSize);
//                biasesOutput = data.biasesOutput;
//            }

//            private float[] Flatten(float[,] array2D)
//            {
//                int rows = array2D.GetLength(0);
//                int cols = array2D.GetLength(1);
//                float[] flat = new float[rows * cols];
//                for (int i = 0; i < rows; i++)
//                    for (int j = 0; j < cols; j++)
//                        flat[i * cols + j] = array2D[i, j];
//                return flat;
//            }

//            private float[,] Reshape(float[] arrayFlat, int rows, int cols)
//            {
//                float[,] array2D = new float[rows, cols];
//                if (arrayFlat.Length != rows * cols)
//                {
//                    Debug.LogError($"Reshape error: Flat array size {arrayFlat.Length} does not match target 2D size {rows}x{cols}={rows * cols}");
//                    return new float[rows, cols];
//                }

//                for (int i = 0; i < rows; i++)
//                    for (int j = 0; j < cols; j++)
//                        array2D[i, j] = arrayFlat[i * cols + j];
//                return array2D;
//            }
//        }
//        #endregion

//        #region Reinforcement Learning Components
//        public struct Experience
//        {
//            public float[] state;
//            public bool[] actionsTaken;
//            public float reward;
//            public float[] nextState;
//            public bool done;
//        }

//        public class ReplayBuffer
//        {
//            private List<Experience> buffer;
//            private int capacity;
//            private int currentIdx;

//            public ReplayBuffer(int cap)
//            {
//                capacity = cap;
//                buffer = new List<Experience>(capacity);
//                currentIdx = 0;
//            }

//            public void AddExperience(Experience exp)
//            {
//                if (buffer.Count < capacity)
//                    buffer.Add(exp);
//                else
//                    buffer[currentIdx] = exp;
//                currentIdx = (currentIdx + 1) % capacity;
//            }

//            public List<Experience> Sample(int batchSize)
//            {
//                List<Experience> batch = new List<Experience>(batchSize);
//                int actualBatchSize = Mathf.Min(batchSize, buffer.Count);
//                for (int i = 0; i < actualBatchSize; i++)
//                {
//                    int randomIndex = UnityEngine.Random.Range(0, buffer.Count);
//                    batch.Add(buffer[randomIndex]);
//                }
//                return batch;
//            }

//            public int Count => buffer.Count;
//        }
//        #endregion
//    }
//}