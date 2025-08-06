using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

public class AIBot_DQN : Bot
{
    #region Bot Template Properties
    public override string ID => "Bot_DQN";
    public override SkillType SkillType => SkillType.Boost;
    private SumoAPI api;
    #endregion

    #region NN Configs
    public bool loadModel = true;
    public bool saveModel = true;
    public string modelFileName = "DQN_Model.json";
    public string csvLogFileName = "DQN_LearningLog.csv";

    public float epsilon = 0.1f;
    public float discountFactor = 0.99f;
    public float learningRate = 0.005f;
    public int replayBufferSize = 10000;
    public int batchSize = 64;
    public float maxEpisodeTime = 10f;
    public int maxTotalEpisodes = 100; 
    public int totalEpisodes = 0;

    public float angleThreshold = 10f;

    private DeepQNetwork DQN;
    private List<DeepQNetwork.Experience> replayBuffer = new();
    private float[] lastStates;     // Inputs: Position X, Position Y, Angle, Distance Normalized
    private int lastAction;

    [SerializeField]
    private float timer = 0f; 
    #endregion

    public override void OnBotInit(SumoAPI botAPI)
    {
        api = botAPI;
        //string path = Path.Combine(Application.persistentDataPath, modelFileName);
        string path = "Assets/Resources/Models/ML/RL/" + modelFileName;
        if (loadModel && File.Exists(path))
        {
            DQN = DeepQNetwork.Load(path);
            Debug.Log($"Loaded DQN from {path}");
        }
        else
        {
            DQN = new DeepQNetwork(4, 16, 3); // 4 Inputs: Position X, Position Y, Angle, Distance Normalized
            Debug.Log("Created new DQN");
        }

        totalEpisodes = 0;
    }

    public override void OnBotUpdate()
    {
        ThinkAndAct();

        Submit();

        timer += 0.1f;
        if (timer >= maxEpisodeTime)
        {
            ResetEpisode();
            Debug.Log($"Angle: {api.Angle()}. AngleDeg: {api.AngleDeg()}, Dist: {api.Distance().magnitude} {api.Distance()}, DistN: {api.DistanceNormalized()}");
            Debug.Log($"MyPos: {api.MyRobot.Position}, MyRot: {api.MyRobot.Rotation}, EnemyPos: {api.EnemyRobot.Position}, EnemyRotation: {api.EnemyRobot.Rotation}");
        }
        if (totalEpisodes >= maxTotalEpisodes) {
#if UNITY_EDITOR
            // If running in the Unity Editor, stop play mode
            EditorApplication.isPlaying = false;
#endif
        }
    }
    public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
    {
        if (state == BattleState.Battle_End)
        {
            ClearCommands();
            if (saveModel)
            {
                //string path = Path.Combine(Application.persistentDataPath, modelFileName);
                string path = "Assets/Resources/Models/ML/RL/" + modelFileName;
                DQN.Save(path);
                Debug.Log($"Saved RL to {path}");
            }
        }
    }

    public override void OnBotCollision(BounceEvent bounceEvent)
    {
        ClearCommands();
    }

    private void ThinkAndAct() 
    {
        float posX = api.MyRobot.Position.x / api.BattleInfo.ArenaRadius;
        float posY = api.MyRobot.Position.y / api.BattleInfo.ArenaRadius;
        float angle = api.Angle();
        float distanceNormalized = api.DistanceNormalized();
        float angleInDur = Mathf.Abs(angle) / api.MyRobot.RotateSpeed * api.MyRobot.TurnRate;

        float[] state = new float[] { posX, posY, angle, distanceNormalized };
        
        // Choose action using epsilon-greedy policy
        int action = DQN.ChooseAction(state, epsilon);
        float reward = CalculateReward(angle, distanceNormalized);

        // [Edit later]
        // Consider to move posX, posY, angle, distanceNormalized into CalculateReward, that means 
        // also consider to use myRobot and enemyRobot states as state[]

        // Store experience
        if (lastStates != null)
        {
            bool done = api.Distance(api.MyRobot.Position, api.BattleInfo.ArenaPosition).magnitude > api.BattleInfo.ArenaRadius + 0.5f 
                || api.Distance(api.EnemyRobot.Position, api.BattleInfo.ArenaPosition).magnitude > api.BattleInfo.ArenaRadius + 0.5f
                || api.BattleInfo.TimeLeft <= 0f || api.BattleInfo.CurrentState == BattleState.Battle_End; // Is the game ended, target reached? 

            replayBuffer.Add(new DeepQNetwork.Experience(lastStates, lastAction, reward, state, done));
            if (replayBuffer.Count > replayBufferSize)
                replayBuffer.RemoveAt(0);
        }

        // Perform action
        if (action == 0)    // && Mathf.Abs(angle) < angleThreshold OR //dot > Mathf.Cos(angleThreshold * Mathf.Deg2Rad))
            Enqueue(new AccelerateAction(InputType.Script));
        else if (action == 1) // && angle > 0)
            Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(0.1f, Mathf.Clamp01(angleInDur))));
        else if (action == 2) // && angle < 0)
            Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(0.1f, Mathf.Clamp01(angleInDur))));

        // Train with a mini batch
        if (replayBuffer.Count >= batchSize) {

            var exp = replayBuffer[replayBuffer.Count - 1]; // Use the latest experience

            float[] outputs = DQN.Forward(exp.state);
            float[] targets = (float[])outputs.Clone(); // Copy current Q-values as base

            float[] nextQ = DQN.Forward(exp.nextState);
            float target = exp.reward;
            if (!exp.done)
                target += discountFactor * Mathf.Max(nextQ);

            targets[exp.action] = target;

            float loss = CalculateLoss(outputs, targets);

            LogDQNLearning(exp.state, outputs, targets, exp.reward, loss, totalEpisodes+1);

            DQN.Train(replayBuffer, batchSize, learningRate, discountFactor);
        }
            
        lastStates = state;
        lastAction = action;
    }
    // Create a function to log DQN learning
    private void LogDQNLearning(float[] inputs, float[] outputs, float[] targets, float reward, float loss, int episode)
    {
        //string path = Path.Combine(Application.persistentDataPath, csvLogFileName);
        string path = "Assets/Resources/Models/ML/RL/" + csvLogFileName;
        bool writeHeader = !File.Exists(path);
        using (StreamWriter sw = new StreamWriter(path, true))
        {
            if (writeHeader)
            {
                sw.WriteLine("Episode,Reward,Timer,Input_PosX,Input_PosY,Input_Angle,Input_DistNorm,Output_Accelerate,Output_TurnLeft,Output_TurnRight,Target_Accelerate,Target_TurnLeft,Target_TurnRight,Loss");
            }
            sw.WriteLine($"{episode},{reward:F2},{timer:F2},{inputs[0]:F4},{inputs[1]:F4},{inputs[2]:F4},{inputs[3]:F4},{outputs[0]:F4},{outputs[1]:F4},{outputs[2]:F4},{targets[0]:F4},{targets[1]:F4},{targets[2]:F4},{loss:F6}");
        }
    }

    private float CalculateLoss(float[] outputs, float[] targets)
    {
        float loss = 0f;
        for (int i = 0; i < outputs.Length; i++)
        {
            float diff = outputs[i] - targets[i];
            loss += diff * diff; // Mean Squared Error
        }
        return loss / outputs.Length;
    }

    float CalculateReward(float angle, float distanceNormalized)
    {
        float reward = 0f;

        // Distance to enemy
        reward += (1 - distanceNormalized) * 2f; // OR (lastdistance - distance) * 1f        
        
        // Angle to enemy 
        reward += (1 - Mathf.Abs(angle) / 180) * 0.5f;

        // MyRobot stays in arena
        if (api.Distance(api.MyRobot.Position, api.BattleInfo.ArenaPosition).magnitude < api.BattleInfo.ArenaRadius + 0.5f)
            reward += 0.1f;
        else if (api.Distance(api.MyRobot.Position, api.BattleInfo.ArenaPosition).magnitude > api.BattleInfo.ArenaRadius + 0.5f)
            reward -= 0.5f;
        // Enemy out of arena
        if (api.Distance(api.EnemyRobot.Position, api.BattleInfo.ArenaPosition).magnitude > api.BattleInfo.ArenaRadius + 0.5f)
            reward += 0.1f;

        return reward;
    }

    void ResetEpisode()
    {
        timer = 0f;
        totalEpisodes++;
        lastStates = null;
    }

    void OnApplicationQuit() 
    {
        if (saveModel)
        {
            //string path = Path.Combine(Application.persistentDataPath, modelFileName);
            string path = "Assets/Resources/Models/ML/RL/" + modelFileName;
            DQN.Save(path);
            Debug.Log($"Saved DQN to {path}");
        }
    }

}

public class DeepQNetwork
{
    #region Experience structure
    public struct Experience
    {
        public float[] state;
        public int action;
        public float reward;
        public float[] nextState;
        public bool done;

        public Experience(float[] state, int action, float reward, float[] nextState, bool done)
        {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
    }
    #endregion

    #region NetworkData structure 
    [Serializable]
    private class NetworkData
    {
        public int inputSize, hiddenSize, outputSize;
        public float[] w1, w2, b1, b2;
    }
    #endregion

    #region Network properties 
    private float[,] weights1; // Input to hidden
    private float[] bias1;
    private float[,] weights2; // Hidden to output
    private float[] bias2;
    private int inputSize, hiddenSize, outputSize;
    private System.Random rand = new();
    #endregion

    public DeepQNetwork(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        weights1 = new float[inputSize, hiddenSize];
        bias1 = new float[hiddenSize];
        weights2 = new float[hiddenSize, outputSize];
        bias2 = new float[outputSize];
        //Randomize(weights1);
        //Randomize(weights2);
        Randomize(weights1, inputSize, hiddenSize);
        Randomize(weights2, hiddenSize, outputSize);
    }

    private void Randomize(float[,] weights, int input, int output)
    {
        for (int i = 0; i < weights.GetLength(0); i++)
            for (int j = 0; j < weights.GetLength(1); j++) {
                //weights[i, j] = (float)(rand.NextDouble() * 0.2 - 0.1);

                // Using Xavier initialization
                int fanIn = input;
                int fanOut = output;
                float limit = Mathf.Sqrt(6f / (fanIn + fanOut));
                weights[i, j] = (float)(rand.NextDouble() * (limit - (-limit)) + (-limit));
            }
                
    }

    public float[] Forward(float[] input)
    {
        float[] hidden = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            hidden[i] = bias1[i];
            for (int j = 0; j < inputSize; j++)
                hidden[i] += input[j] * weights1[j, i];
            hidden[i] = MathF.Tanh(hidden[i]);
        }

        float[] output = new float[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            output[i] = bias2[i];
            for (int j = 0; j < hiddenSize; j++)
                output[i] += hidden[j] * weights2[j, i];
        }
        return output;
    }

    public int ChooseAction(float[] state, float epsilon)
    {
        if (rand.NextDouble() < epsilon)
            return rand.Next(0, outputSize);

        float[] qValues = Forward(state);
        int maxAction = 0;
        float maxValue = qValues[0];
        for (int i = 1; i < qValues.Length; i++)
        {
            if (qValues[i] > maxValue)
            {
                maxValue = qValues[i];
                maxAction = i;
            }
        }
        return maxAction;
    }

    public void Train(List<DeepQNetwork.Experience> replayBuffer, int batchSize, float learningRate, float discountFactor)
    {
        for (int i = 0; i < batchSize; i++)
        {
            int idx = rand.Next(replayBuffer.Count);
            var exp = replayBuffer[idx];

            float[] qValues = Forward(exp.state);
            float[] nextQValues = Forward(exp.nextState);
            float target = exp.reward;
            if (!exp.done)
                target += discountFactor * Mathf.Max(nextQValues);

            float[] dOutput = new float[outputSize];
            for (int j = 0; j < outputSize; j++)
                dOutput[j] = j == exp.action ? (qValues[j] - target) : 0f;

            float[] hidden = new float[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                hidden[j] = bias1[j];
                for (int k = 0; k < inputSize; k++)
                    hidden[j] += exp.state[k] * weights1[k, j];
                hidden[j] = MathF.Tanh(hidden[j]);
            }

            float[] dHidden = new float[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                float error = 0f;
                for (int k = 0; k < outputSize; k++)
                    error += dOutput[k] * weights2[j, k];
                dHidden[j] = error * (1 - hidden[j] * hidden[j]);
            }

            for (int j = 0; j < hiddenSize; j++)
                for (int k = 0; k < outputSize; k++)
                    weights2[j, k] -= learningRate * dOutput[k] * hidden[j];

            for (int j = 0; j < outputSize; j++)
                bias2[j] -= learningRate * dOutput[j];

            for (int j = 0; j < inputSize; j++)
                for (int k = 0; k < hiddenSize; k++)
                    weights1[j, k] -= learningRate * dHidden[k] * exp.state[j];

            for (int j = 0; j < hiddenSize; j++)
                bias1[j] -= learningRate * dHidden[j];
        }
    }

    public void Save(string path)
    {
        NetworkData d = new()
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

    public static DeepQNetwork Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(path);

        NetworkData d = JsonUtility.FromJson<NetworkData>(File.ReadAllText(path));
        DeepQNetwork dqn = new DeepQNetwork(d.inputSize, d.hiddenSize, d.outputSize);
        dqn.Unflatten(d.w1, dqn.weights1);
        dqn.Unflatten(d.w2, dqn.weights2);
        dqn.bias1 = (float[])d.b1.Clone();
        dqn.bias2 = (float[])d.b2.Clone();
        return dqn;
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
