using SumoCore;
using SumoInput;
using SumoManager;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace SumoBot {
    public class AIBot_NN : Bot
    {
        #region Runtime Properties
        public override string ID => "Bot_NN";
        public override SkillType SkillType => SkillType.Boost;
        #endregion

        #region NN Properties
        [Header("Neural Network Hyperparameters")]
        public float learningRate = 0.001f;
        public float discountFactor = 0.99f;    // Gamma for future rewards
        public float epsilon = 1.0f;            // Exploration rate (starts high, decays)
        public float epsilonDecay = 0.999f;     // Epsilon decay factor per update step
        public float minEpsilon = 0.01f;        // Minimum exploration rate
        public float trainingInterval = 0.1f;  // How frequently to attempt a training step 

        [Header("Replay Buffer")]
        public int replayBufferSize = 10000;
        public int batchSize = 64;              // Number of experiences to learn from per training step
        public int minExpLearn = 500;         // Start learning after collecting this many experiences
        #endregion

        #region Model Properties
        [Header("Model Saving/Loading")]
        public bool loadPretrainedModel = false;
        public string modelFileName = "NN_Model.json";
        private string modelFilePath;           // Full path to the model file
        #endregion

        #region Private Properties
        private const int INPUT_SIZE = 16;      // Adjust based on GatherInputs method
        private const int OUTPUT_SIZE = 5;      // Outputs: 0=Accelerate, 1=TurnLeft, 2=TurnRight, 3=Dash, 4=Skill
        private const int HIDDEN_SIZE = 128;    // Number of neurons in hidden layer (can be tuned)
        
        private NeuralNetwork qNetwork;
        private ReplayBuffer replayBuffer;

        private SumoAPI api;
        private PlayerSide mySide;

        private float[] lastState;              // State from previous update for experience tuple
        private int lastAction;                 // Action from previous update
        private float trainingStepTimer;
        private float tempReward = 0f;
        #endregion

        #region Bot Methods
        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;

            replayBuffer = new ReplayBuffer(replayBufferSize);
            modelFilePath = Path.Combine(Application.persistentDataPath, modelFileName);

            // Load model if exist
            if (loadPretrainedModel && File.Exists(modelFilePath))
            {
                LoadModel();
                Debug.Log("Loaded pre-trained model from: " + modelFilePath);
                epsilon = minEpsilon; // Start with minimal exploration if exists
            }
            else
            {
                qNetwork = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
                Debug.Log("Initialized new model. Starting training from scratch.");
                epsilon = 1.0f; // Start with full exploration if not exists
            }

            lastState = null; 
            trainingStepTimer = 0f;

            Debug.Log($"NN Bot Initialized. Side: {mySide}");
        }

        public override void OnBattleStateChanged(BattleState state)
        {
            BattleState currState = state;
            Debug.Log($"Battle State Changed: {state}");
            if (currState == BattleState.Battle_End)
            {
                SaveModel(); 
                lastState = null;
                Debug.Log("Battle ended. Model saved and episode context reset.");
            }

            // Optionally save model when paused

            // For other state changes, you might also reset `lastState` if you want each round
            // to be a distinct episode, or only reset at Battle_End for continuous training across rounds.
        }

        public override void OnBotCollision(EventParameter param)
        {
            // Reward for collision 
            float speedColThreshold = 0.1f; 
            RobotStateAPI myRobot = api.MyRobot;
            RobotStateAPI enemyRobot = api.EnemyRobot;

            if (myRobot.LinearVelocity.magnitude - speedColThreshold > enemyRobot.LinearVelocity.magnitude)
                tempReward -= 0.5f;
            else if (myRobot.LinearVelocity.magnitude < enemyRobot.LinearVelocity.magnitude - speedColThreshold)
                tempReward += 0.5f;
            else
                tempReward = 0f; //-= 0.1f;

            Debug.Log($"Collision detected! Temp Reward: {tempReward} (My Speed: {myRobot.LinearVelocity.magnitude}, Enemy Speed: {enemyRobot.LinearVelocity.magnitude})");
        }

        public override void OnBotUpdate()
        {
            ClearCommands();
            RobotStateAPI myRobot = api.MyRobot;
            RobotStateAPI enemyRobot = api.EnemyRobot;

            float[] currentState = GatherInputs(myRobot, enemyRobot);

            if (lastState != null) 
            {
                float currentReward = CalculateReward(myRobot, enemyRobot, lastAction);
                bool done = IsEpisodeDone();

                // Add experience (S_t-1, A_t-1, R_t, S_t, Done) to replay buffer
                replayBuffer.AddExperience(new Experience
                {
                    state = lastState,
                    action = lastAction,
                    reward = currentReward,
                    nextState = currentState,
                    done = done
                });

                if (done)
                    lastState = null; 
            }

            // Choose Action (A_t) using Epsilon-Greedy
            int chosenAction = ChooseAction(currentState);
            ExecuteAction(chosenAction, myRobot);

            lastState = currentState;
            lastAction = chosenAction;

            // Learn from Replay Buffer (Train Network)
            trainingStepTimer += Time.deltaTime;
            if (trainingStepTimer >= trainingInterval && replayBuffer.Count > minExpLearn)
            {
                List<Experience> batch = replayBuffer.Sample(batchSize);
                if (batch.Count > 0)
                    TrainNetwork(batch);

                trainingStepTimer = 0f; 
            }

            // Decay epsilon for exploration
            epsilon = Mathf.Max(minEpsilon, epsilon * epsilonDecay);

            Submit(); 
        }
        #endregion

        #region NN and RL methods
        private float[] GatherInputs(RobotStateAPI myRobot, RobotStateAPI enemyRobot)
        {
            float[] inputs = new float[INPUT_SIZE];

            // Normalize values (e.g., 0 to 1, or -1 to 1)
            // Own Robot State: [0] Position.x, [1] Position.y, [2] Linear Velocity, [3] Angular Velocity, [4] Facing Angle, [5] Distance to Enemy, [6] Relative Facing Angle, [7] Dash Cooldown, [8] Skill Cooldown
            inputs[0] = myRobot.Position.x / api.BattleInfo.ArenaRadius; 
            inputs[1] = myRobot.Position.y / api.BattleInfo.ArenaRadius; 
            inputs[2] = myRobot.LinearVelocity.magnitude / myRobot.MoveSpeed; 
            inputs[3] = myRobot.AngularVelocity / (myRobot.RotateSpeed * myRobot.TurnRate * 100f); // [edit later]
            inputs[4] = api.Angle() / 180f; 
            inputs[5] = Vector3.Distance(myRobot.Position, enemyRobot.Position) / (2 * api.BattleInfo.ArenaRadius); 
            inputs[6] = (Vector3.SignedAngle(myRobot.Rotation * Vector3.up, (api.BattleInfo.ArenaPosition - myRobot.Position).normalized, Vector3.forward)) / 180f; 
            inputs[7] = myRobot.IsDashOnCooldown ? 0f : 1f;
            inputs[8] = myRobot.Skill.IsSkillOnCooldown ? 0f : 1f;
            // Enemy State: [9] Position.x, [10] Position.y, [11] Linear Velocity, [12] Angular Velocity, [13] Facing Angle, [14] Relative Facing Angle, [15] Distance to Enemy
            inputs[9] = enemyRobot.Position.x / api.BattleInfo.ArenaRadius;
            inputs[10] = enemyRobot.Position.y / api.BattleInfo.ArenaRadius;
            inputs[11] = enemyRobot.LinearVelocity.magnitude / enemyRobot.MoveSpeed;
            inputs[12] = enemyRobot.AngularVelocity / (enemyRobot.RotateSpeed * enemyRobot.TurnRate * 100f);
            inputs[13] = enemyRobot.Rotation / 360f;
            inputs[14] = Vector3.Dot((myRobot.Rotation * Vector3.up).normalized, (enemyRobot.Rotation * Vector3.up).normalized);
            inputs[15] = Vector3.Dot(myRobot.LinearVelocity.normalized, (enemyRobot.Position - myRobot.Position).normalized);

            for (int i = 0; i < INPUT_SIZE; i++)
                inputs[i] = Mathf.Clamp(inputs[i], -1f, 1f);

            return inputs;
        }

        private int ChooseAction(float[] state)
        {
            // Epsilon-greedy strategy
            if (UnityEngine.Random.value < epsilon)
            {
                // Explore
                return UnityEngine.Random.Range(0, OUTPUT_SIZE);
            }
            else
            {
                // Exploit
                float[] qValues = qNetwork.FeedForward(state);
                int bestAction = 0;
                for (int i = 1; i < OUTPUT_SIZE; i++)
                {
                    if (qValues[i] > qValues[bestAction])
                        bestAction = i;

                }
                return bestAction;
            }
        }

        private void ExecuteAction(int action, RobotStateAPI myRobot)
        {
            switch (action)
            {
                case 0: 
                    Enqueue(new AccelerateAction(InputType.Script));
                    break;
                case 1: // Turn Left (fixed duration for simplicity)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, 0.15f));
                    break;
                case 2: // Turn Right (fixed duration for simplicity)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, 0.15f));
                    break;
                case 3: 
                    if (!myRobot.IsDashOnCooldown)
                        Enqueue(new DashAction(InputType.Script));
                    break;
                case 4: 
                    if (!myRobot.Skill.IsSkillOnCooldown)
                        Enqueue(new SkillAction(InputType.Script));
                    break;
            }
        }

        private float CalculateReward(RobotStateAPI myRobot, RobotStateAPI enemyRobot, int actionTaken)
        {
            // Reward & penalty settings 
            float reward = 0f;

            // 1. Stay moving 
            if (myRobot.LinearVelocity.magnitude > 0.1f)
                reward += 0.01f;

            // 2. Pushing towards the edge
            float enemyDistFromCenter = Vector3.Distance(enemyRobot.Position, api.BattleInfo.ArenaPosition);
            float myDistFromCenter = Vector3.Distance(myRobot.Position, api.BattleInfo.ArenaPosition);
            float arenaEdgeThreshold = api.BattleInfo.ArenaRadius * 0.9f;
            
            if (enemyDistFromCenter > arenaEdgeThreshold) 
            {
                reward += 1f; 

                if (Vector3.Distance(myRobot.Position, enemyRobot.Position) < 1.0f && Vector3.Dot(myRobot.LinearVelocity.normalized, (enemyRobot.Position - myRobot.Position).normalized) > 0.8f)
                    reward += 5f; 
            }

            // 3. Edge penalty 
            if (myDistFromCenter > arenaEdgeThreshold)
            {
                reward -= 2f;
                if (myDistFromCenter > api.BattleInfo.ArenaRadius * 0.95f)
                    reward -= 10f;
            }

            // 4. Close to enemy
            if (Vector3.Distance(myRobot.Position, enemyRobot.Position) < 2.5f)
                reward += 0.05f;

            // 5. Far from enemy penalty
            if (Vector3.Distance(myRobot.Position, enemyRobot.Position) > 5.0f)
                reward -= 0.05f;

            // 6. Unecessary actions penalty
            if ((actionTaken == 3 && myRobot.IsDashOnCooldown) || (actionTaken == 4 && myRobot.Skill.IsSkillOnCooldown))
                reward -= 0.1f;

            // 7. Collision initiator 
            if (tempReward != 0f) {
                reward += tempReward;
                tempReward = 0f;
            }

            return reward;
        }

        private bool IsEpisodeDone()
        {
            // An episode is considered "done" when the battle ends, or a round ends (if rounds reset state).
            // This is crucial for the Q-learning algorithm to properly calculate discounted future rewards.
            return api.BattleInfo.CurrentState == BattleState.Battle_End;
        }

        // Q-Network Training loop 
        private void TrainNetwork(List<Experience> batch)
        {
            foreach (Experience exp in batch)
            {
                float[] currentQValues = qNetwork.FeedForward(exp.state); // Q-values for current state (S)
                float targetQValue = exp.reward; // Immediate reward (R)

                if (!exp.done)
                {
                    // Calculate target Q-value using Bellman equation: Q(s,a) = r + gamma * max(Q(s',a'))
                    float[] nextQValues = qNetwork.FeedForward(exp.nextState); // Q-values for next state (S')
                    float maxNextQ = nextQValues[0];
                    for (int i = 1; i < nextQValues.Length; i++)
                    {
                        if (nextQValues[i] > maxNextQ)
                            maxNextQ = nextQValues[i];
                    }
                    targetQValue += discountFactor * maxNextQ;
                }

                // Create target output array for training
                // Only update the Q-value for the action that was actually taken (A)
                float[] targetOutputs = (float[])currentQValues.Clone(); // Start with current Q-values
                targetOutputs[exp.action] = targetQValue; // Update target for the taken action

                qNetwork.Train(exp.state, targetOutputs, learningRate);
            }
        }

        private void SaveModel()
        {
            try
            {
                qNetwork.PrepareForSerialization();
                string json = JsonUtility.ToJson(qNetwork);
                File.WriteAllText(modelFilePath, json);
                Debug.Log("Model saved successfully to: " + modelFilePath);
            }
            catch (Exception e)
            {
                Debug.LogError("Failed to save model: " + e.Message);
            }
        }

        private void LoadModel()
        {
            try
            {
                string json = File.ReadAllText(modelFilePath);
                NeuralNetwork loadedNetwork = JsonUtility.FromJson<NeuralNetwork>(json);
                if (loadedNetwork != null)
                {
                    qNetwork = loadedNetwork;
                    qNetwork.RestoreFromSerialization();
                    Debug.Log("Model loaded successfully from: " + modelFilePath);
                }
                else
                    throw new Exception("JsonUtility returned null during deserialization.");
            }
            catch (Exception e)
            {
                Debug.LogError("Failed to load model: " + e.Message);
                qNetwork = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            }
        }
        #endregion
    }

    #region Neural Network Core Components
    public static class Activation 
    { 
        public static float ReLU(float x) => MathF.Max(0, x);
        public static float ReluDerivative(float x) => x > 0 ? 1 : 0;
        public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
        public static float SigmoidDerivative(float x) => Sigmoid(x) * (1 - Sigmoid(x)); 
    }

    // Feed-forward Neural Network
    [System.Serializable]
    public class NeuralNetwork
    {
        #region Neural Network Properties
        private int inputSize;
        private int hiddenSize;
        private int outputSize;
        private float[,] weightsInput;      // [inputSize, hiddenSize]
        private float[,] weightsOutput;     // [hiddenSize, outputSize]
        private float[] biasesHidden;       // [hiddenSize]
        private float[] biasesOutput;       // [outputSize]
        #endregion

        #region JsonUtility
        private float[] weightsInputFlat;      
        private float[] weightsOutputFlat;     
        private float[] biasesHiddenSerializable;       
        private float[] biasesOutputSerializable;
        #endregion

        #region Neural Network Components
        public NeuralNetwork(int input, int hidden, int output) 
        {
            inputSize = input;
            hiddenSize = hidden;
            outputSize = output;
            weightsInput = new float[inputSize, hiddenSize];
            weightsOutput = new float[hiddenSize, outputSize];
            biasesHidden = new float[hiddenSize];
            biasesOutput = new float[outputSize];
            
            InitializeWeights();
        }

        private void InitializeWeights() 
        {
            // Simple ReLU
            float limitInput = Mathf.Sqrt(2f / inputSize); //or 1f / MathF.Sqrt(inputSize);
            float limitHidden = Mathf.Sqrt(2f / hiddenSize); //or 1f / MathF.Sqrt(hiddenSize);

            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) { 
                    weightsInput[i, j] = UnityEngine.Random.Range(-limitInput, limitInput);
                }
            }

            for (int i = 0; i < hiddenSize; i++)
            {
                biasesHidden[i] = 0.01f;
            }

            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weightsOutput[i, j] = UnityEngine.Random.Range(-limitHidden, limitHidden);
                }
            }

            for (int i = 0; i < outputSize; i++)
            {
                biasesOutput[i] = 0.01f;
            }
        }

        // Inference 
        public float[] FeedForward(float[] inputs) 
        {
            if (inputs.Length != inputSize) {
                Debug.LogError($"Input array size mismatch! Expected {inputSize}, got {inputs.Length}");
                return new float[outputSize];
            }

            float[] hiddenOutputs = new float[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                float sum = biasesHidden[i];
                for (int j = 0; j < inputSize; j++)
                    sum += inputs[j] * weightsInput[j, i];

                hiddenOutputs[i] = Activation.ReLU(sum);
            }

            float[] finalOutputs = new float[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                float sum = biasesOutput[i];
                for (int j = 0; j < hiddenSize; j++)
                    sum += hiddenOutputs[j] * weightsOutput[j, i];

                finalOutputs[i] = Activation.Sigmoid(sum); 
            }

            return finalOutputs;
        }

        // Backpropagation
        public void Train(float[] inputs, float[] targetOutputs, float learningRate)
        { 
            // Forward Pass 
            float[] hiddenInputsRaw = new float[hiddenSize];
            float[] hiddenOutputs = new float[hiddenSize];
            
            for (int i = 0; i < hiddenSize; i++)
            {
                float sum = biasesHidden[i];
                for (int j = 0; j < inputSize; j++)
                    sum += inputs[j] * weightsInput[j, i];

                hiddenInputsRaw[i] = sum;
                hiddenOutputs[i] = Activation.ReLU(sum);
            }

            float[] outputInputsRaw = new float[outputSize];
            float[] actualOutputs = new float[outputSize];
            for (int i = 0; i < outputSize; i++) { 
                float sum = biasesOutput[i];
                for (int j = 0; j < hiddenSize; j++)
                    sum += hiddenOutputs[j] * weightsOutput[j, i];

                outputInputsRaw[i] = sum;
                actualOutputs[i] = Activation.Sigmoid(sum);
            }

            // Back Pass 
            float[] outputErrors = new float[outputSize];
            for (int i = 0; i < outputSize; i++)
                outputErrors[i] = actualOutputs[i]- targetOutputs[i] * Activation.SigmoidDerivative(outputInputsRaw[i]); // MSE 

            float[] hiddenErrors = new float[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                float error = 0f;
                for (int j = 0; j < outputSize; j++)
                    error += outputErrors[j] * weightsOutput[i, j];

                hiddenErrors[i] = error * Activation.ReluDerivative(hiddenInputsRaw[i]);
            }

            // Update weights and biases
            for (int i = 0; i < hiddenSize; i++) 
            {
                for (int j = 0; j < outputSize; j++)  
                    weightsOutput[i, j] -= learningRate * outputErrors[j] * hiddenOutputs[i];
            }
            for (int i = 0; i < outputSize; i++) { 
                biasesOutput[i] -= learningRate * outputErrors[i];
            }

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                    weightsInput[i, j] -= learningRate * hiddenErrors[j] * inputs[i];
            }
            for (int i = 0; i < hiddenSize; i++)
                biasesHidden[i] -= learningRate * hiddenErrors[i];
        }
        #endregion

        #region JsonUtility Helper (convert 2D arrays to 1D and vice cersa)
        public void PrepareForSerialization()
        {
            weightsInputFlat = Flatten(weightsInput);
            weightsOutputFlat = Flatten(weightsOutput);
            biasesHiddenSerializable = biasesHidden;
            biasesOutputSerializable = biasesOutput;
        }

        private float[] Flatten(float[,] array2D)
        {
            int rows = array2D.GetLength(0);
            int cols = array2D.GetLength(1);
            float[] flat = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    flat[i * cols + j] = array2D[i, j];
                }
            }
            return flat;
        }

        public void RestoreFromSerialization()
        {
            if (weightsInputFlat == null || weightsOutputFlat == null ||
                biasesHiddenSerializable == null || biasesOutputSerializable == null)
            {
                Debug.LogError("NeuralNetwork data incomplete for deserialization.");
                InitializeWeights();
                return;
            }

            weightsInput = Reshape(weightsInputFlat, inputSize, hiddenSize);
            weightsOutput = Reshape(weightsOutputFlat, hiddenSize, outputSize);
            biasesHidden = biasesHiddenSerializable;
            biasesOutput = biasesOutputSerializable;
        }

        private float[,] Reshape(float[] arrayFlat, int rows, int cols)
        {
            float[,] array2D = new float[rows, cols];
            if (arrayFlat.Length != rows * cols)
            {
                Debug.LogError($"Reshape error: Flat array size {arrayFlat.Length} does not match target 2D size {rows}x{cols}={rows * cols}");
                return new float[rows, cols];
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    array2D[i, j] = arrayFlat[i * cols + j];
                }
            }
            return array2D;
        }
        #endregion

    }
    #endregion

    #region Reinforcement Learning Components
    public struct Experience
    {
        public float[] state;
        public int action;
        public float reward;
        public float[] nextState;
        public bool done;
    }

    public class ReplayBuffer
    {
        private List<Experience> buffer;
        private int capacity;
        private int currentIdx;

        public ReplayBuffer(int cap)
        {
            capacity = cap;
            buffer = new List<Experience>(capacity);
            currentIdx = 0;
        }

        public void AddExperience(Experience exp)
        {
            if (buffer.Count < capacity)
            {
                buffer.Add(exp);
            }
            else
            {
                buffer[currentIdx] = exp;
            }
            currentIdx = (currentIdx + 1) % capacity;
        }

        public List<Experience> Sample(int batchSize)
        {
            List<Experience> batch = new List<Experience>(batchSize);
            // Ensure we don't try to sample more than available experiences
            int actualBatchSize = Mathf.Min(batchSize, buffer.Count);
            for (int i = 0; i < actualBatchSize; i++)
            {
                int randomIndex = UnityEngine.Random.Range(0, buffer.Count);
                batch.Add(buffer[randomIndex]);
            }
            return batch;
        }

        public int Count => buffer.Count;
    }
    #endregion

}

