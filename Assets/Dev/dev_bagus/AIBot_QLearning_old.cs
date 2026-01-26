//using UnityEngine;
//using SumoBot;
//using SumoInput;
//using SumoCore;
//using SumoManager;
//using System.Collections.Generic;
//using System.IO;
//using System;
//using System.Linq;

//public class AIBot_Q_Learning : Bot
//{
//    public override string ID => "QLearningBot";
//    public override SkillType SkillType => SkillType.Boost;

//    private SumoAPI api;
//    private SimpleNN nn;
//    private List<Transition> memory = new();
//    private const int inputSize = 18;
//    private const int hiddenSize = 24;
//    private const int outputSize = 5;

//    private float epsilon = 0.4f; // exploration
//    private float gamma = 0.99f;  // future reward discount
//    private float learningRate = 0.01f;

//    private float[] lastState;
//    private List<int> lastActions = new();

//    private BattleState currState;
//    private int gameCounter = 0;
//    private float episodeReward = 0f;
//    private float trainInterval = 1f; 
//    private float lastTrainTime = 0f;

//    public override void OnBotInit(SumoAPI botAPI)
//    {
//        api = botAPI;
//        nn = new SimpleNN(inputSize, hiddenSize, outputSize);
//        LoadModel("Assets/Logs/model.json");
//    }

//    public override void OnBotUpdate()
//    {
//        float[] state = ExtractState();
//        float[] qValues = nn.Forward(state);
//        float threshold = 0.8f;

//        lastActions.Clear(); // Clear from previous tick

//        // Choose action with high Q-value or explore randomly
//        qValues.Select((q, i) => new { Value = q, Index = i })
//               .OrderByDescending(x => x.Value)
//               .Take(2) // Take top 3 actions
//               .ToList()
//               .ForEach(x =>
//               {
//                   if (x.Value > threshold)
//                   {
//                       EnqueueAction(x.Index);
//                       lastActions.Add(x.Index);
//                       Logger.Info($"Action {x.Index} with Q-value {x.Value:F2} selected (threshold: {threshold:F2})");
//                   }
//               });

//        //for (int i = 0; i < qValues.Length; i++)
//        //{
//        //    if (qValues[i] > threshold)
//        //    {
//        //        EnqueueAction(i);
//        //        lastActions.Add(i);
//        //        // Logger.Info($"Action {i} with Q-value {qValues[i]:F2} selected (threshold: {threshold:F2})");
//        //    }

//        //    Logger.Info($"Action {i} with Q-value {qValues[i]:F2}");
//        //}

//        Submit();

//        if (lastState != null && lastActions.Count > 0)
//        {
//            float reward = ComputeShapedReward();
//            episodeReward += reward;

//            foreach (int action in lastActions)
//                memory.Add(new Transition(lastState, action, reward, state));
//        }

//        lastState = state;

//        if (Time.time - lastTrainTime >= trainInterval)
//        {
//            TrainFromMemory();
//            lastTrainTime = Time.time;
//            Logger.Info($"Training game {gameCounter} time {lastTrainTime}, episode reward: {episodeReward:F2}, epsilon: {epsilon:F2}.");
//        }
//    }


//    public override void OnBotCollision(BounceEvent bounceEvent) { }

//    public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
//    {
//        currState = state;

//        if (currState == BattleState.Battle_End)
//        {
//            float finalReward = (winner.ToString() == api.MyRobot.Side.ToString()) ? 1f : -1f;
//            episodeReward += finalReward;

//            if (lastState != null && lastActions.Count > 0)
//            {
//                foreach (int action in lastActions)
//                    memory.Add(new Transition(lastState, action, finalReward, null));
//            }

//            TrainFromMemory();

//            if (lastState != null)
//            {
//                float[] qValues = nn.Forward(lastState);
//                LogQValuesAndReward(
//                    winner.ToString(),
//                    episodeReward,
//                    new List<int>(lastActions), // log all actions taken last tick
//                    qValues,
//                    ((int)Time.unscaledTime).ToString()
//                );
//            }

//            SaveModel("Assets/Logs/model.json");

//            memory.Clear();
//            lastState = null;
//            lastActions.Clear();
//            epsilon = Mathf.Max(0.05f, epsilon * 0.995f);
//            gameCounter++;
//            episodeReward = 0f;
//        }
//    }


//    float[] ExtractState()
//    {
//        var my = api.MyRobot;
//        var enemy = api.EnemyRobot;
//        Vector2 center = api.BattleInfo.ArenaPosition;
//        float arenaRadius = api.BattleInfo.ArenaRadius;
//        Vector2 distToEnemy = enemy.Position - my.Position;
//        Vector2 distToCenter = center - my.Position;
//        float angleToEnemy = api.Angle(my.Position, my.Rotation, enemy.Position, false);
//        float angleToCenter = api.Angle(my.Position, my.Rotation, center);

//        return new float[] {
//            my.Position.x, my.Position.y,
//            my.Rotation,
//            distToEnemy.magnitude / arenaRadius,
//            distToCenter.magnitude / arenaRadius,
//            angleToEnemy / 180f,
//            angleToCenter / 180f,
//            my.LinearVelocity.x, my.LinearVelocity.y,
//            my.AngularVelocity,
//            my.IsDashOnCooldown ? 1f : 0f,
//            my.Skill.IsSkillOnCooldown ? 1f : 0f,
//            enemy.Position.x, enemy.Position.y,
//            enemy.Rotation,
//            enemy.LinearVelocity.x, enemy.LinearVelocity.y,
//            enemy.Skill.IsSkillOnCooldown ? 1f : 0f
//        };
//    }

//    int ChooseAction(float[] state)
//    {
//        if (UnityEngine.Random.value < epsilon)
//            return UnityEngine.Random.Range(0, outputSize);

//        float[] qValues = nn.Forward(state);
//        return qValues.ToList().IndexOf(qValues.Max());
//    }

//    void EnqueueAction(int index)
//    {
//        switch (index)
//        {
//            case 0: Enqueue(new AccelerateAction(InputType.Script)); break;
//            case 1: Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, 0.1f)); break;
//            case 2: Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, 0.1f)); break;
//            case 3: if (!api.MyRobot.IsDashOnCooldown) Enqueue(new DashAction(InputType.Script)); break;
//            case 4: if (!api.MyRobot.Skill.IsSkillOnCooldown) Enqueue(new SkillAction(InputType.Script)); break;
//        }
//    }

//    float IntermediateReward()
//    {
//        float dist = (api.MyRobot.Position - api.BattleInfo.ArenaPosition).magnitude;
//        float normDist = dist / api.BattleInfo.ArenaRadius;
//        return 0.1f * (1f - normDist); // reward staying near center
//    }

//    float ComputeShapedReward()
//    {
//        var my = api.MyRobot;
//        var enemy = api.EnemyRobot;
//        Vector2 center = api.BattleInfo.ArenaPosition;
//        float arenaRadius = api.BattleInfo.ArenaRadius;

//        float distToCenter = (my.Position - center).magnitude / arenaRadius;
//        float distToEnemy = (enemy.Position - my.Position).magnitude / arenaRadius;
//        float enemyDistToCenter = (enemy.Position - center).magnitude / arenaRadius;
//        float angleToEnemy = api.Angle(my.Position, my.Rotation, enemy.Position, false);

//        float reward = 0f;

//        // 1. Stay near center
//        reward += (1f - distToCenter) * 0.1f;

//        // 2. Get close to enemy
//        reward += (1f - distToEnemy) * 0.05f;

//        // 3. Enemy near edge
//        reward += Mathf.Clamp01(enemyDistToCenter - 0.9f) * 0.2f;

//        // 4. Self near edge penalty
//        if (distToCenter > 0.9f)
//            reward -= 0.1f;

//        // 5. Dash hit bonus (if enemy is in front and in range)
//        if (!my.IsDashOnCooldown && angleToEnemy < 25f && (enemy.Position - my.Position).magnitude < 2f)
//            reward += 0.2f;

//        // 6. Skill bonus (if available and enemy is in range)
//        if (!my.Skill.IsSkillOnCooldown && (enemy.Position - my.Position).magnitude < 2f)
//            reward += 0.2f;

//        // 7. Enemy is slower
//        float myVel = my.LinearVelocity.magnitude;
//        float enemyVel = enemy.LinearVelocity.magnitude;
//        if (myVel > enemyVel)
//            reward += 0.05f;

//        // 8. Facing enemy and able to chase = encourage acceleration
//        Vector2 toEnemyN = (enemy.Position - my.Position).normalized;
//        Vector2 forward = new Vector2(Mathf.Cos(my.Rotation * Mathf.Deg2Rad), Mathf.Sin(my.Rotation * Mathf.Deg2Rad));
//        float facingDot = Vector2.Dot(toEnemyN, forward); // close to 1 = directly facing

//        // 8. Encourage chasing
//        if (facingDot > 0.8f && (enemy.Position - my.Position).magnitude < 3f)
//            reward += 0.2f;

//        // 9. Penalize forward near edge
//        if (distToCenter > 0.85f && facingDot > 0.7f)
//            reward -= 0.2f;

//        return reward;
//    }

//    void TrainFromMemory()
//    {
//        foreach (var trans in memory)
//        {
//            float[] target = nn.Forward(trans.State);
//            float qTarget = trans.Reward;

//            //if (trans.NextState != null)
//            //{
//            //    float[] nextQ = nn.Forward(trans.NextState);
//            //    qTarget += gamma * nextQ.Max();
//            //}

//            if (trans.NextState != null)
//            {
//                float[] nextQ = nn.Forward(trans.NextState);
//                float nextQMax = nextQ.Max();

//                // If the action was accelerate (0) but poorly aligned, lower the Q target
//                bool wasAccel = trans.Action == 0;
//                float[] state = trans.State;
//                float angleToEnemy = state[5] * 180f; // assuming index 5 = angle to enemy
//                float misalignmentPenalty = Mathf.Clamp01(Mathf.Abs(angleToEnemy) / 180f);

//                if (wasAccel && misalignmentPenalty > 0.3f)
//                    nextQMax *= (1f - misalignmentPenalty); // degrade Q for poor direction

//                qTarget += gamma * nextQMax;
//            }


//            target[trans.Action] = qTarget;

//            // Boost Q[0] in early training
//            if (gameCounter < 25) 
//                target[0] += 0.2f; 

//            nn.Train(trans.State, target, learningRate);
//        }
//    }

//    void SaveModel(string path)
//    {
//        File.WriteAllText(path, nn.ToJson());
//    }

//    void LoadModel(string path)
//    {
//        if (File.Exists(path))
//            nn.FromJson(File.ReadAllText(path));
//    }

//    [Serializable]
//    public class Transition
//    {
//        public float[] State;
//        public int Action;
//        public float Reward;
//        public float[] NextState;

//        public Transition(float[] state, int action, float reward, float[] nextState)
//        {
//            State = state;
//            Action = action;
//            Reward = reward;
//            NextState = nextState;
//        }
//    }

//    void LogQValuesAndReward(string winner, float reward, List<int> actions, float[] qValues, string timestamp)
//    {
//        string dir = "Assets/Logs";
//        string path = $"{dir}/qvalues_log.csv";

//        if (!Directory.Exists(dir))
//            Directory.CreateDirectory(dir);

//        bool exists = File.Exists(path);
//        using StreamWriter writer = new(path, true);

//        if (!exists)
//            writer.WriteLine("TimeStamp,Episode,Winner,Reward,Actions,MaxQ,Q0,Q1,Q2,Q3,Q4");

//        float maxQ = qValues.Max();
//        string actionsStr = string.Join("|", actions); // e.g. "0|3|4"

//        string line = $"{timestamp},{gameCounter},{winner},{reward:F2},{actionsStr},{maxQ:F4}," +
//                      string.Join(",", qValues.Select(q => q.ToString("F4")));

//        writer.WriteLine(line);
//    }

//}

//// Custom renamed data model for JSON log
//[Serializable] public class SumoGameLog { public int Index; public string Timestamp; public string Winner; public List<SumoRound> Rounds; }
//[Serializable] public class SumoRound { public int Index; public string Timestamp; public string Winner; public float Duration; public List<SumoPlayerEvent> PlayerEvents; }
//[Serializable] public class SumoPlayerEvent { public float StartedAt; public float UpdatedAt; public string Actor; public string Target; public float Duration; public string Category; public int State; public SumoActionData Data; }
//[Serializable] public class SumoActionData { public string Name; public float Duration; public string Reason; public SumoRobotState Robot; }
//[Serializable] public class SumoRobotState { public float AngularVelocity; public SumoVector2 LinearVelocity; public SumoVector2 Position; public float Rotation; }
//[Serializable] public class SumoVector2 { public float X; public float Y; }

//public class TrainingSample { public float[] Inputs; public float[] Targets; public TrainingSample(float[] i, float[] t) { Inputs = i; Targets = t; } }

//[Serializable]
//public class SimpleNN
//{
//    public int inputSize, hiddenSize, outputSize;
//    public float[,] weightsInputHidden, weightsHiddenOutput;
//    public float[] hiddenBiases, outputBiases;

//    public SimpleNN(int inputSize, int hiddenSize, int outputSize)
//    {
//        this.inputSize = inputSize;
//        this.hiddenSize = hiddenSize;
//        this.outputSize = outputSize;

//        weightsInputHidden = new float[inputSize, hiddenSize];
//        weightsHiddenOutput = new float[hiddenSize, outputSize];
//        hiddenBiases = new float[hiddenSize];
//        outputBiases = new float[outputSize];

//        RandomizeWeights();
//    }

//    private void RandomizeWeights()
//    {
//        System.Random rnd = new();
//        for (int i = 0; i < inputSize; i++)
//            for (int j = 0; j < hiddenSize; j++)
//                weightsInputHidden[i, j] = (float)(rnd.NextDouble() - 0.5);

//        for (int i = 0; i < hiddenSize; i++)
//            for (int j = 0; j < outputSize; j++)
//                weightsHiddenOutput[i, j] = (float)(rnd.NextDouble() - 0.5);
//    }

//    public float[] Forward(float[] input)
//    {
//        float[] hidden = new float[hiddenSize];
//        for (int i = 0; i < hiddenSize; i++)
//        {
//            hidden[i] = hiddenBiases[i];
//            for (int j = 0; j < inputSize; j++)
//                hidden[i] += input[j] * weightsInputHidden[j, i];
//            hidden[i] = Sigmoid(hidden[i]);
//        }

//        float[] output = new float[outputSize];
//        for (int i = 0; i < outputSize; i++)
//        {
//            output[i] = outputBiases[i];
//            for (int j = 0; j < hiddenSize; j++)
//                output[i] += hidden[j] * weightsHiddenOutput[j, i];
//            output[i] = Sigmoid(output[i]);
//        }
//        return output;
//    }

//    public void Train(float[] input, float[] target, float lr)
//    {
//        float[] hidden = new float[hiddenSize];
//        float[] output = new float[outputSize];

//        for (int i = 0; i < hiddenSize; i++)
//        {
//            hidden[i] = hiddenBiases[i];
//            for (int j = 0; j < inputSize; j++)
//                hidden[i] += input[j] * weightsInputHidden[j, i];
//            hidden[i] = Sigmoid(hidden[i]);
//        }

//        for (int i = 0; i < outputSize; i++)
//        {
//            output[i] = outputBiases[i];
//            for (int j = 0; j < hiddenSize; j++)
//                output[i] += hidden[j] * weightsHiddenOutput[j, i];
//            output[i] = Sigmoid(output[i]);
//        }

//        float[] outputErrors = new float[outputSize];
//        float[] hiddenErrors = new float[hiddenSize];

//        for (int i = 0; i < outputSize; i++)
//            outputErrors[i] = (target[i] - output[i]) * output[i] * (1 - output[i]);

//        for (int i = 0; i < hiddenSize; i++)
//        {
//            for (int j = 0; j < outputSize; j++)
//                hiddenErrors[i] += outputErrors[j] * weightsHiddenOutput[i, j];
//            hiddenErrors[i] *= hidden[i] * (1 - hidden[i]);
//        }

//        for (int i = 0; i < hiddenSize; i++)
//            for (int j = 0; j < outputSize; j++)
//                weightsHiddenOutput[i, j] += lr * outputErrors[j] * hidden[i];

//        for (int i = 0; i < inputSize; i++)
//            for (int j = 0; j < hiddenSize; j++)
//                weightsInputHidden[i, j] += lr * hiddenErrors[j] * input[i];

//        for (int i = 0; i < outputSize; i++)
//            outputBiases[i] += lr * outputErrors[i];
//        for (int i = 0; i < hiddenSize; i++)
//            hiddenBiases[i] += lr * hiddenErrors[i];
//    }

//    private float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

//    [Serializable]
//    private class NNModel
//    {
//        public int inputSize, hiddenSize, outputSize;
//        public float[] weightsInputHidden;
//        public float[] weightsHiddenOutput;
//        public float[] hiddenBiases;
//        public float[] outputBiases;
//    }

//    public string ToJson()
//    {
//        NNModel model = new()
//        {
//            inputSize = inputSize,
//            hiddenSize = hiddenSize,
//            outputSize = outputSize,
//            hiddenBiases = hiddenBiases,
//            outputBiases = outputBiases,
//            weightsInputHidden = new float[inputSize * hiddenSize],
//            weightsHiddenOutput = new float[hiddenSize * outputSize]
//        };

//        for (int i = 0; i < inputSize; i++)
//            for (int j = 0; j < hiddenSize; j++)
//                model.weightsInputHidden[i * hiddenSize + j] = weightsInputHidden[i, j];

//        for (int i = 0; i < hiddenSize; i++)
//            for (int j = 0; j < outputSize; j++)
//                model.weightsHiddenOutput[i * outputSize + j] = weightsHiddenOutput[i, j];

//        return JsonUtility.ToJson(model);
//    }

//    public void FromJson(string json)
//    {
//        NNModel model = JsonUtility.FromJson<NNModel>(json);
//        if (model.inputSize != inputSize || model.hiddenSize != hiddenSize || model.outputSize != outputSize)
//        {
//            Debug.LogWarning("Model dimensions do not match.");
//            return;
//        }

//        hiddenBiases = model.hiddenBiases;
//        outputBiases = model.outputBiases;
//        weightsInputHidden = new float[inputSize, hiddenSize];
//        weightsHiddenOutput = new float[hiddenSize, outputSize];

//        for (int i = 0; i < inputSize; i++)
//            for (int j = 0; j < hiddenSize; j++)
//                weightsInputHidden[i, j] = model.weightsInputHidden[i * hiddenSize + j];

//        for (int i = 0; i < hiddenSize; i++)
//            for (int j = 0; j < outputSize; j++)
//                weightsHiddenOutput[i, j] = model.weightsHiddenOutput[i * outputSize + j];
//    }
//}
