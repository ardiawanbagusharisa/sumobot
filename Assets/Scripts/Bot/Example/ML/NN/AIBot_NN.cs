using UnityEngine;
using SumoBot;
using SumoCore;
using SumoManager;
using SumoInput;
using System.IO;
using System;

public class AIBot_NN : Bot
{
    #region Bot Template Properties
    public override string ID => "Bot_NN";
    public override SkillType SkillType => SkillType.Boost;
    private SumoAPI api;
    #endregion

    #region NN Configs
    public bool loadModel = true;
    public bool saveModel = false;
    public string modelFileName = "NN_Model";
    public string csvLogFileName = "NN_LearningLog.csv";
    public float learningRate = 0.01f;
    public float maxEpisodeTime = 10f;
    public float angleThreshold = 10f;
    public int totalEpisodes = 0;

    public int input = 4; // Position X, Position Y, Angle, Distance Normalized
    public int hidden = 16; // Hidden layer size
    public int output = 5; // Accelerate, TurnLeft, TurnRight, Dash, Skill

    private NeuralNetwork NN;
    [SerializeField]
    private float timer = 0f;

    #endregion

    #region Bot Methods
    public override void OnBotInit(SumoAPI botAPI)
    {
        api = botAPI;

        //string path = Path.Combine(Application.persistentDataPath, modelFileName);
        string path = "ML/Models/NN/" + modelFileName;
        if (loadModel)
        {
            NN = NeuralNetwork.Load(path);
            Debug.Log($"Loaded NN from {path}");
            return;
        }
        else
        {
            // 4 Inputs: Position X, Position Y, Angle, Distance Normalized
            // 5 Outputs: Accelerate, TurnLeft, TurnRight, Dash, Skill 
            NN = new NeuralNetwork(input, hidden, output);
            Debug.Log("Created new NN");
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
            Debug.Log($"Angle: {api.Angle()}, Dist: {api.Distance().magnitude} {api.Distance()}, DistN: {api.DistanceNormalized()}");
            Debug.Log($"MyPos: {api.MyRobot.Position}, MyRot: {api.MyRobot.Rotation}, EnemyPos: {api.EnemyRobot.Position}, EnemyRotation: {api.EnemyRobot.Rotation}");
        }
    }

    public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
    {
        Debug.Log($"{state}. Winner: {winner}");

        if (state == BattleState.Battle_End)
        {
            ClearCommands();
            if (saveModel)
            {
                //string path = Path.Combine(Application.persistentDataPath, modelFileName);
                string path = "Assets/Resources/ML/Models/NN/" + modelFileName + ".json";
                NN.Save(path);
                Debug.Log($"Saved NN to {path}");
            }
        }
    }

    public override void OnBotCollision(BounceEvent bounceEvent)
    {
        ClearCommands();
    }
    #endregion

    #region NN Calls
    private void ThinkAndAct()
    {
        float posX = api.MyRobot.Position.x / api.BattleInfo.ArenaRadius;
        float posY = api.MyRobot.Position.y / api.BattleInfo.ArenaRadius;
        float angle = api.Angle();
        float distanceNormalized = api.DistanceNormalized();
        float isDashCD = api.MyRobot.IsDashOnCooldown ? 1f : 0f;
        float isSkillCD = api.MyRobot.Skill.IsSkillOnCooldown ? 1f : 0f;

        // Get inputs and outputs for NN
        // [Edit later] Consider to use IsDashCD and IsSkillCD as inputs too
        float[] inputs = new float[] { posX, posY, angle, distanceNormalized, isDashCD, isSkillCD };

        float[] outputs = NN.Forward(inputs);

        float accelerate = outputs[0];      // 0 to 1
        float turnLeft = outputs[1];        // -1 to 1
        float turnRight = outputs[2];       // -1 to 1
        float dash = outputs[3];          // 0 or 1
        float skill = outputs[4];         // 0 or 1

        float angleInDur = Mathf.Abs(angle) / api.MyRobot.RotateSpeed;

        if (angle > 0 && turnLeft > 0.05f)
        {
            Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(0.1f, Mathf.Clamp01(angleInDur))));
        }
        else if (angle < 0 && turnRight > 0.05f)
        {
            Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(0.1f, Mathf.Clamp01(angleInDur))));
        }

        if (Mathf.Abs(angle) < angleThreshold && accelerate > 0.05f)
        {
            Enqueue(new AccelerateAction(InputType.Script, Mathf.Max(0.1f, Mathf.Clamp01(accelerate))));
        }

        if (!api.MyRobot.IsDashOnCooldown && dash > 0.05f)  // && angle < dashSkillAngle
        {
            Enqueue(new DashAction(InputType.Script));
        }
        if (!api.MyRobot.Skill.IsSkillOnCooldown && skill > 0.05f) // && angle < dashSkillAngle
        {
            if (SkillType == SkillType.Boost || (SkillType == SkillType.Stone && api.DistanceNormalized(api.MyRobot.Position, api.BattleInfo.ArenaPosition) > 0.8f))
            {
                Enqueue(new SkillAction(InputType.Script));
            }
        }

        // Train 
        float[] targetOutputs = new float[5]; //{ accelerate, turnLeft, turnRight, dash, skill };
        targetOutputs[0] = Mathf.Clamp01(accelerate);
        targetOutputs[1] = angle > 0 ? Mathf.Abs(angle) / 180f : 0f;
        targetOutputs[2] = angle < 0 ? Mathf.Abs(angle) / 180f : 0f;
        targetOutputs[3] = !api.MyRobot.IsDashOnCooldown && Mathf.Abs(angle) < angleThreshold ? 1f : 0f;
        targetOutputs[4] = !api.MyRobot.Skill.IsSkillOnCooldown && Mathf.Abs(angle) < angleThreshold ? 1f : 0f;

        NN.Train(inputs, targetOutputs, learningRate);

        if (saveModel)
            LogNNLearning(inputs, outputs, targetOutputs, CalculateLoss(outputs, targetOutputs));
    }

    void ResetEpisode()
    {
        timer = 0f;
        totalEpisodes++;
    }

    void OnApplicationQuit()
    {
        if (saveModel)
        {
            string modelPath = "ML/Models/NN/" + modelFileName + ".json";
            string path = Path.Combine(Application.streamingAssetsPath, modelPath);
            NN.Save(path);
            Debug.Log($"Saved NN to {path}");
        }
    }

    private void LogNNLearning(float[] inputs, float[] outputs, float[] targets, float loss)
    {
        string csvPath = "ML/Models/NN/" + csvLogFileName;
        string path = Path.Combine(Application.streamingAssetsPath, csvPath);
        bool writeHeader = !File.Exists(path);
        using (StreamWriter sw = new StreamWriter(path, true))
        {
            if (writeHeader)
            {
                sw.WriteLine("Episode,Timer,Input_PosX,Input_PosY,Input_Angle,Input_DistNorm,Output_Accelerate,Output_TurnLeft,Output_TurnRight,Output_IsDashCD,Output_IsSkillCD,Target_Accelerate,Target_TurnLeft,Target_TurnRight,Target_IsDashCD,Target_IsSkillCD,Loss");
            }
            sw.WriteLine($"{totalEpisodes},{timer:F2},{inputs[0]:F4},{inputs[1]:F4},{inputs[2]:F4},{inputs[3]:F4},{outputs[0]:F4},{outputs[1]:F4},{outputs[2]:F4},{outputs[3]:F4},{outputs[4]:F4},{targets[0]:F4},{targets[1]:F4},{targets[2]:F4},{targets[3]:F4},{targets[4]:F4},{loss:F6}");
        }
    }

    private float CalculateLoss(float[] outputs, float[] targets)
    {
        // MSE loss (for logging purposes)
        float loss = 0f;
        int len = Math.Min(outputs.Length, targets.Length);
        for (int i = 0; i < len; i++)
        {
            float diff = outputs[i] - targets[i];
            loss += diff * diff;
        }
        return loss / len;
    }
    #endregion
}

public class NeuralNetwork
{
    #region NN Properties
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
    #endregion

    #region NN Methods
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
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
                weights[i, j] = (float)(rand.NextDouble() * 2 - 1);
    }

    private float[] Activate(float[] input)
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
            output[i] = MathF.Tanh(output[i]);
        }

        return output;
    }

    public float[] Forward(float[] input) => Activate(input);

    public void Train(float[] input, float[] target, float learningRate = 0.01f)
    {
        // Forward pass
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
            output[i] = MathF.Tanh(output[i]);
        }

        // Backpropagation
        float[] dOutput = new float[outputSize];
        for (int i = 0; i < outputSize; i++)
            dOutput[i] = (output[i] - target[i]) * (1 - output[i] * output[i]); // derivative of tanh

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
        {
            for (int j = 0; j < outputSize; j++)
                weights2[i, j] -= learningRate * dOutput[j] * hidden[i];
        }

        for (int i = 0; i < outputSize; i++)
            bias2[i] -= learningRate * dOutput[i];

        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
                weights1[i, j] -= learningRate * dHidden[j] * input[i];
        }

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

    public static NeuralNetwork Load(string path)
    {
        TextAsset model = Resources.Load<TextAsset>(path);
        if (model == null)
            throw new FileNotFoundException(path);

        NetworkData d = JsonUtility.FromJson<NetworkData>(model.text);
        NeuralNetwork nn = new NeuralNetwork(d.inputSize, d.hiddenSize, d.outputSize);

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
    #endregion
}
