using SumoBot;
using SumoCore;
using SumoManager;
using SumoInput;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;

public class AIBot_PPO_Nunu : Bot
{
    public override string ID => "PPO_Nunu";
    public override SkillType DefaultSkillType => SkillType.Boost;

    [Header("Model")]
    [Tooltip("If true, loads Nunu_PPO_Model.json from Resources. Set in code defaults; bots are created at runtime so this rarely appears in Inspector.")]
    public bool loadModel = true;
    public bool saveModel = false;
    public string modelFileName = "Nunu_PPO_Model";
    public string csvLogFileName = "Nunu_PPO_LearningLog_phase1_resume.csv";

    [Header("PPO Hyperparameters")]
    public int nSteps = 512;
    public float gamma = 0.99f;
    public float gaeLambda = 0.95f;
    public float clipEpsilon = 0.2f;
    public float learningRate = 0.0003f;
    public float valueLossCoefficient = 0.5f;
    public float entropyCoefficient = 0.008f;
    public int ppoEpochs = 4;
    public int batchSize = 64;
    public int maxTotalEpisodes = 500;

    [Header("Action")]
    [Tooltip("Only used when edge override steers toward center.")]
    public float angleThreshold = 15f;

    [Header("Edge Safety Override")]
    [Tooltip("Off during training so PPO learns edge recovery from reward. Enable for inference if needed.")]
    public bool enableEdgeOverride = false;
    public float edgeThreshold = 0.92f;

    public int inputSize = 11;
    public int hiddenSize = 64;
    public int outputSize = 5;

    private static class NunuPpoTrainingConfig
    {
        public const int MaxEpisodesOverride = 0;
    }

    private SumoAPI api;
    private NunuProximalPolicyOptimization ppo;
    private readonly List<NunuProximalPolicyOptimization.Experience> rolloutBuffer = new();

    private int totalEpisodes;
    private int stepInEpisode;
    private int episodeOverrideCount;
    private int totalOverrideCount;
    private float episodeRewardSum;
    private float prevDistNorm = -1f;
    private string lastEpisodeOutcome = "Step";
    private bool battleTerminalReached;

    private string GetModelPath()
    {
        string folder = Path.Combine(Application.dataPath, "Resources/Bot/Example/Micro-Competition");

        // Create folder automatically if not exist
        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);

        return Path.Combine(folder, modelFileName + ".json");
    }

    public override void OnBotInit(SumoAPI botAPI)
    {
        api = botAPI;

#if UNITY_EDITOR
        if (NunuPpoTrainingConfig.MaxEpisodesOverride > 0)
        {
            maxTotalEpisodes = NunuPpoTrainingConfig.MaxEpisodesOverride;
            Logger.Info($"[Bot_nunu] maxTotalEpisodes overridden to {maxTotalEpisodes}");
        }
#endif

        if (ppo != null)
        {
            Logger.Info($"[Bot_nunu] OnBotInit re-entry â€” keeping PPO weights (totalEpisodes={totalEpisodes})");
            return;
        }

        string path = GetModelPath(); //"ML/Models/RL/" + modelFileName;

        if (loadModel)
        {
            try
            {
                ppo = NunuProximalPolicyOptimization.Load(path);
                Logger.Info($"[Bot_nunu] Loaded PPO from {path}");
            }
            catch (Exception ex)
            {
                Logger.Info($"[Bot_nunu] Load failed ({ex.Message}). Creating new PPO network.");
                ppo = new NunuProximalPolicyOptimization(inputSize, hiddenSize, outputSize);
            }
        }
        else
        {
            ppo = new NunuProximalPolicyOptimization(inputSize, hiddenSize, outputSize);
            Logger.Info("[Bot_nunu] Created new PPO network");
        }

        ResetEpisodeCounters();
        totalEpisodes = 0;
        totalOverrideCount = 0;
    }

    public override void OnBotUpdate()
    {
        ClearCommands();

        if (api != null
            && api.BattleInfo.CurrentState == BattleState.Battle_Ongoing
            && !battleTerminalReached)
        {
            ThinkAndAct();
        }

        Submit();
    }

    public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
    {
        if (state == BattleState.Battle_Countdown)
        {
            ResetEpisodeCounters();
            battleTerminalReached = false;
        }

        if (state != BattleState.Battle_End)
            return;

        ClearCommands();

#if UNITY_EDITOR
        // Headless sim runs Iteration N then stops; don't kill Play mid-batch.
        if (!Application.isBatchMode && totalEpisodes >= maxTotalEpisodes)
        {
            Logger.Info($"[Bot_nunu] Reached maxTotalEpisodes={maxTotalEpisodes}. Stopping Play mode.");
            UnityEditor.EditorApplication.isPlaying = false;
        }
#endif
    }

    public override void OnBotCollision(BounceEvent bounceEvent)
    {
        ClearCommands();
    }

    private void ThinkAndAct()
    {
        float[] obs = BuildObservation();
        (float[] probs, float value) = ppo.Forward(obs);

        int rawAction = SampleAction(probs);
        int finalAction = TryEdgeOverride(rawAction, out bool wasOverridden);
        if (wasOverridden)
        {
            episodeOverrideCount++;
            totalOverrideCount++;
        }

        ExecuteAction(finalAction, wasOverridden);

        bool done = CheckDone(out bool selfOut, out bool enemyOut);
        float reward = CalculateStepReward(obs, done, selfOut, enemyOut);
        episodeRewardSum += reward;

        if (done)
        {
            if (selfOut)
                lastEpisodeOutcome = "Loss";
            else if (enemyOut)
                lastEpisodeOutcome = "Win";
            else
                lastEpisodeOutcome = "Draw";

#if UNITY_EDITOR
            if (saveModel)
                LogTrainingUpdate(true, 0f, probs, 0f, 0f);
#endif
        }

        if (ShouldTrain() && !wasOverridden)
        {
            float oldProb = probs[Mathf.Clamp(rawAction, 0, probs.Length - 1)];
            float[] nextObs = done ? obs : BuildObservation();
            rolloutBuffer.Add(new NunuProximalPolicyOptimization.Experience(
                (float[])obs.Clone(),
                rawAction,
                reward,
                (float[])nextObs.Clone(),
                oldProb,
                value,
                done));

            if (rolloutBuffer.Count >= nSteps)
                RunPpoUpdate(obs, done, probs);
        }

        stepInEpisode++;

        if (done && !battleTerminalReached)
        {
            battleTerminalReached = true;
            ResetEpisodeCounters();
            totalEpisodes++;
        }
    }

    private float[] BuildObservation()
    {
        Vector2 centerToMe = (api.MyRobot.Position - (Vector2)api.BattleInfo.ArenaPosition).normalized;
        float zRot = api.MyRobot.Rotation % 360f;
        if (zRot < 0) zRot += 360f;
        Vector2 facingDir = ((Vector2)(Quaternion.Euler(0, 0, zRot) * Vector2.up)).normalized;
        float facingOutside = Vector2.Dot(facingDir, centerToMe);

        float nearArena = api.Distance(api.MyRobot.Position, api.BattleInfo.ArenaPosition).magnitude
            / api.BattleInfo.ArenaRadius;

        Vector2 myVelNorm = NormalizeLinearVelocity(api.MyRobot);
        Vector2 enemyVelNorm = NormalizeLinearVelocity(api.EnemyRobot);
        float myAngNorm = NormalizeAngularVelocity(api.MyRobot);
        float enemyAngNorm = NormalizeAngularVelocity(api.EnemyRobot);

        return new float[]
        {
            api.Angle(normalized: true),
            api.DistanceNormalized(),
            nearArena,
            facingOutside,
            api.MyRobot.IsDashOnCooldown ? 1f : 0f,
            myVelNorm.x,
            myVelNorm.y,
            enemyVelNorm.x,
            enemyVelNorm.y,
            myAngNorm,
            enemyAngNorm
        };
    }

    private static Vector2 NormalizeLinearVelocity(SumoBotAPI robot)
    {
        float maxSpeed = Mathf.Max(robot.MoveSpeed, robot.DashSpeed, 0.01f);
        return robot.LinearVelocity / maxSpeed;
    }

    private static float NormalizeAngularVelocity(SumoBotAPI robot)
    {
        return robot.AngularVelocity / Mathf.Max(robot.RotateSpeed, 0.01f);
    }

    private int TryEdgeOverride(int rawAction, out bool wasOverridden)
    {
        wasOverridden = false;
        if (!enableEdgeOverride)
            return rawAction;

        float[] obs = BuildObservation();
        float nearArena = obs[2];
        float facingOutside = obs[3];

        if (nearArena <= edgeThreshold || facingOutside <= 0f)
            return rawAction;

        float angleToCenter = api.Angle(targetPos: api.BattleInfo.ArenaPosition);
        wasOverridden = true;
        return angleToCenter > 0f ? 1 : 2;
    }

    private void ExecuteAction(int action, bool turnTowardCenter)
    {
        float angle = turnTowardCenter
            ? api.Angle(targetPos: api.BattleInfo.ArenaPosition)
            : api.Angle();
        float angleInDur = Mathf.Abs(angle) / Mathf.Max(api.MyRobot.RotateSpeed, 0.01f);

        switch (action)
        {
            case 0:
                Enqueue(new AccelerateAction(InputType.Script));
                break;
            case 1:
                if (angle > 0f)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(0.1f, Mathf.Clamp01(angleInDur))));
                break;
            case 2:
                if (angle < 0f)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(0.1f, Mathf.Clamp01(angleInDur))));
                break;
            case 3:
                if (!api.MyRobot.IsDashOnCooldown)
                    Enqueue(new DashAction(InputType.Script));
                break;
            case 4:
                if (!api.MyRobot.Skill.IsSkillOnCooldown)
                    Enqueue(new SkillAction(InputType.Script));
                break;
        }
    }

    private float CalculateStepReward(float[] obs, bool done, bool selfOut, bool enemyOut)
    {
        float angleScore = obs[0];
        float distNorm = obs[1];
        float nearArena = obs[2];
        float facingOutside = obs[3];

        float reward = 0f;

        // Phase 1 survival v4: aggressive edge avoidance, reward staying alive
        const float edgeStart = 0.65f;
        if (nearArena > edgeStart)
        {
            reward -= (nearArena - edgeStart) * 14f;
            if (facingOutside > 0f)
                reward -= facingOutside * (nearArena - 0.6f) * 1.2f;
        }
        else if (nearArena < 0.5f)
            reward += 0.12f;
        else
            reward += (1f - nearArena) * 0.15f;

        // Minimal chase â€” survival first during phase 1
        if (prevDistNorm >= 0f)
            reward += (prevDistNorm - distNorm) * 0.1f;
        prevDistNorm = distNorm;

        reward += angleScore * 0.02f;

        if (done)
        {
            if (selfOut)
                reward -= 15f;
            else if (enemyOut)
                reward += 10f;
            else
                reward += 2f;
        }

        return reward;
    }

    private bool CheckDone(out bool selfOut, out bool enemyOut)
    {
        float outThreshold = api.BattleInfo.ArenaRadius + 0.4f;
        float myDist = api.Distance(api.MyRobot.Position, api.BattleInfo.ArenaPosition).magnitude;
        float enemyDist = api.Distance(api.EnemyRobot.Position, api.BattleInfo.ArenaPosition).magnitude;

        selfOut = myDist > outThreshold;
        enemyOut = enemyDist > outThreshold;

        return selfOut || enemyOut || api.BattleInfo.TimeLeft <= 0f;
    }

    private void RunPpoUpdate(float[] currentObs, bool done, float[] probs)
    {
        float bootstrap = done ? 0f : ppo.Forward(currentObs).value;

        // Short rollouts (< batchSize) overfit easily; cap epochs during phase 1 survival runs
        int effectiveEpochs = rolloutBuffer.Count < batchSize
            ? Mathf.Min(2, ppoEpochs)
            : ppoEpochs;

        var (policyLoss, valueLoss, _) = ppo.TrainInternal(
            rolloutBuffer,
            effectiveEpochs,
            batchSize,
            clipEpsilon,
            learningRate,
            valueLossCoefficient,
            entropyCoefficient,
            gamma,
            gaeLambda,
            bootstrap,
            done);

#if UNITY_EDITOR
        if (saveModel)
        {
            string path = GetModelPath(); 
            ppo.Save(path);
        }
#endif
//"Assets/Resources/ML/Models/RL/" + modelFileName + ".json";
        rolloutBuffer.Clear();
    }

    private bool ShouldTrain() => saveModel || !loadModel;

    private int SampleAction(float[] probs)
    {
        float[] working = (float[])probs.Clone();
        float sum = 0f;
        for (int i = 0; i < working.Length; i++)
        {
            if (float.IsNaN(working[i]) || working[i] < 0f)
                working[i] = 0.0001f;
            sum += working[i];
        }

        if (sum <= 0f)
        {
            for (int i = 0; i < working.Length; i++)
                working[i] = 1f / working.Length;
        }
        else
        {
            for (int i = 0; i < working.Length; i++)
                working[i] = working[i] / sum;
        }

        float r = UnityEngine.Random.value;
        float cumulative = 0f;
        for (int i = 0; i < working.Length; i++)
        {
            cumulative += working[i];
            if (r <= cumulative)
                return i;
        }

        return working.Length - 1;
    }

    private void ResetEpisodeCounters()
    {
        stepInEpisode = 0;
        episodeOverrideCount = 0;
        episodeRewardSum = 0f;
        prevDistNorm = -1f;
        lastEpisodeOutcome = "Step";
    }

    private void LogTrainingUpdate(bool done, float bootstrap, float[] probs, float valueLoss, float policyLoss)
    {
        //string path = "Assets/Resources/ML/Models/RL/" + csvLogFileName;
        string path = Path.Combine(
            Path.GetDirectoryName(GetModelPath()),
            csvLogFileName
        );
        bool writeHeader = !File.Exists(path);
        float entropy = CalculateEntropy(probs);
        using var sw = new StreamWriter(path, true);
        if (writeHeader)
        {
            sw.WriteLine("Episode,EpisodeLength,AvgReward,OverrideCount,TotalOverrideCount,PolicyEntropy,ValueLoss,PolicyLoss,RolloutDone,Bootstrap,Outcome");
        }

        float avgReward = stepInEpisode > 0 ? episodeRewardSum / stepInEpisode : episodeRewardSum;
        string outcome = done ? lastEpisodeOutcome : "Step";
        sw.WriteLine($"{totalEpisodes},{stepInEpisode},{avgReward:F4},{episodeOverrideCount},{totalOverrideCount},{entropy:F4},{valueLoss:F4},{policyLoss:F4},{done},{bootstrap:F4},{outcome}");
    }

    private static float CalculateEntropy(float[] probs)
    {
        float entropy = 0f;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] > 0f)
                entropy -= probs[i] * Mathf.Log(probs[i]);
        return entropy;
    }

    private void OnApplicationQuit()
    {
#if UNITY_EDITOR
        if (saveModel)
        {
            string path = "Assets/Resources/ML/Models/RL/" + modelFileName + ".json";
            ppo.Save(path);
            Logger.Info($"[Bot_nunu] Saved model to {path}");
        }
#endif
    }
}

public class NunuProximalPolicyOptimization
{
    public struct Experience
    {
        public float[] state;
        public int action;
        public float reward;
        public float[] nextState;
        public float oldProb;
        public float value;
        public bool done;

        public Experience(float[] state, int action, float reward, float[] nextState, float oldProb, float value, bool done)
        {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.oldProb = oldProb;
            this.value = value;
            this.done = done;
        }
    }

    [Serializable]
    private class NetworkData
    {
        public int inputSize, hiddenSize, outputSize;
        public float[] w1, w2, b1, b2, wValue;
        public float bValue;
    }

    private float[,] weights1;
    private float[] bias1;
    private float[,] weights2;
    private float[] bias2;
    private float[] weightsValue;
    private float biasValue;
    private int inputSize, hiddenSize, outputSize;
    private readonly System.Random rand = new System.Random();

    public NunuProximalPolicyOptimization(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        weights1 = new float[inputSize, hiddenSize];
        bias1 = new float[hiddenSize];
        weights2 = new float[hiddenSize, outputSize];
        bias2 = new float[outputSize];
        weightsValue = new float[hiddenSize];
        biasValue = 0f;
        Randomize(weights1);
        Randomize(weights2);
        Randomize(bias1);
        Randomize(bias2);
        Randomize(weightsValue);
    }

    private void Randomize(float[,] weights)
    {
        for (int i = 0; i < weights.GetLength(0); i++)
            for (int j = 0; j < weights.GetLength(1); j++)
                weights[i, j] = (float)(rand.NextDouble() * 0.1 - 0.05);
    }

    private void Randomize(float[] values)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rand.NextDouble() * 0.1 - 0.05);
    }

    public (float[] probs, float value) Forward(float[] input)
    {
        if (input.Length != inputSize)
        {
            Logger.Error($"[Bot_nunu] Input size mismatch: expected {inputSize}, got {input.Length}");
            return (new float[outputSize], 0f);
        }

        float[] hidden = Hidden(input);
        float[] logits = new float[outputSize];
        float maxLogit = float.MinValue;
        for (int i = 0; i < outputSize; i++)
        {
            logits[i] = bias2[i];
            for (int j = 0; j < hiddenSize; j++)
                logits[i] += hidden[j] * weights2[j, i];
            maxLogit = Mathf.Max(maxLogit, logits[i]);
        }

        float sum = 0f;
        float[] probs = new float[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            probs[i] = Mathf.Exp(Mathf.Clamp(logits[i] - maxLogit, -20f, 20f));
            sum += probs[i];
        }
        for (int i = 0; i < outputSize; i++)
            probs[i] = sum > 1e-8f ? probs[i] / sum : 1f / outputSize;

        float value = biasValue;
        for (int i = 0; i < hiddenSize; i++)
            value += hidden[i] * weightsValue[i];

        return (probs, value);
    }

    public (float policyLoss, float valueLoss, float totalLoss) TrainInternal(List<Experience> buffer, int epochs, int batchSize, float clipEpsilon, float learningRate, float valueLossCoefficient, float entropyCoefficient, float gamma, float gaeLambda, float bootstrap, bool rolloutDone)
    {
        if (buffer == null || buffer.Count == 0)
            return (0f, 0f, 0f);

        float[] returns = CalculateReturns(buffer, gamma, bootstrap, rolloutDone);
        float avgPolicyLoss = 0f, avgValueLoss = 0f, avgTotalLoss = 0f;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalPolicyLoss = 0f, totalValueLoss = 0f, totalLoss = 0f;
            int count = 0;

            for (int start = 0; start < buffer.Count; start += Mathf.Max(1, batchSize))
            {
                int end = Math.Min(start + Mathf.Max(1, batchSize), buffer.Count);
                for (int i = start; i < end; i++)
                {
                    Experience exp = buffer[i];
                    var forward = Forward(exp.state);
                    float[] probs = forward.probs;
                    float value = forward.value;
                    float advantage = returns[i] - exp.value;
                    float ratio = probs[exp.action] / Mathf.Max(exp.oldProb, 1e-8f);
                    float clippedRatio = Mathf.Clamp(ratio, 1f - clipEpsilon, 1f + clipEpsilon);
                    float policyLoss = -Mathf.Min(ratio * advantage, clippedRatio * advantage);
                    float valueLoss = (value - returns[i]) * (value - returns[i]);
                    float entropy = CalculateEntropy(probs);
                    float loss = policyLoss + valueLossCoefficient * valueLoss - entropyCoefficient * entropy;

                    ApplyGradient(exp.state, exp.action, probs, value, returns[i], advantage, ratio, clippedRatio, learningRate, valueLossCoefficient);

                    totalPolicyLoss += policyLoss;
                    totalValueLoss += valueLoss;
                    totalLoss += loss;
                    count++;
                }
            }

            if (count > 0)
            {
                avgPolicyLoss = totalPolicyLoss / count;
                avgValueLoss = totalValueLoss / count;
                avgTotalLoss = totalLoss / count;
            }
        }

        return (avgPolicyLoss, avgValueLoss, avgTotalLoss);
    }

    private void ApplyGradient(float[] input, int action, float[] probs, float value, float returnValue, float advantage, float ratio, float clippedRatio, float learningRate, float valueLossCoefficient)
    {
        float[] hidden = Hidden(input);
        float policyScale = Math.Abs(ratio - clippedRatio) > 1e-6f ? 0f : -advantage * ratio;
        float[] dLogits = new float[outputSize];
        for (int i = 0; i < outputSize; i++)
            dLogits[i] = policyScale * ((i == action ? 1f : 0f) - probs[i]);

        float dValue = 2f * (value - returnValue) * valueLossCoefficient;
        float[] dHidden = new float[hiddenSize];

        for (int h = 0; h < hiddenSize; h++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                dHidden[h] += dLogits[o] * weights2[h, o];
                weights2[h, o] -= learningRate * Mathf.Clamp(dLogits[o] * hidden[h], -0.1f, 0.1f);
            }
            dHidden[h] += dValue * weightsValue[h];
            weightsValue[h] -= learningRate * Mathf.Clamp(dValue * hidden[h], -0.1f, 0.1f);
        }

        for (int o = 0; o < outputSize; o++)
            bias2[o] -= learningRate * Mathf.Clamp(dLogits[o], -0.1f, 0.1f);
        biasValue -= learningRate * Mathf.Clamp(dValue, -0.1f, 0.1f);

        for (int h = 0; h < hiddenSize; h++)
        {
            float dz = dHidden[h] * (1f - hidden[h] * hidden[h]);
            for (int i = 0; i < inputSize; i++)
                weights1[i, h] -= learningRate * Mathf.Clamp(dz * input[i], -0.1f, 0.1f);
            bias1[h] -= learningRate * Mathf.Clamp(dz, -0.1f, 0.1f);
        }
    }

    private float[] Hidden(float[] input)
    {
        float[] hidden = new float[hiddenSize];
        for (int h = 0; h < hiddenSize; h++)
        {
            hidden[h] = bias1[h];
            for (int i = 0; i < inputSize; i++)
                hidden[h] += input[i] * weights1[i, h];
            hidden[h] = (float)Math.Tanh(hidden[h]);
        }
        return hidden;
    }

    private static float CalculateEntropy(float[] probs)
    {
        float entropy = 0f;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] > 0f)
                entropy -= probs[i] * Mathf.Log(probs[i]);
        return entropy;
    }

    private static float[] CalculateReturns(List<Experience> buffer, float gamma, float bootstrap, bool rolloutDone)
    {
        float[] returns = new float[buffer.Count];
        float discounted = rolloutDone ? 0f : bootstrap;
        for (int i = buffer.Count - 1; i >= 0; i--)
        {
            discounted = buffer[i].reward + gamma * discounted * (buffer[i].done ? 0f : 1f);
            returns[i] = discounted;
        }
        return returns;
    }

    public void Save(string path)
    {
        NetworkData data = new NetworkData
        {
            inputSize = inputSize,
            hiddenSize = hiddenSize,
            outputSize = outputSize,
            w1 = Flatten(weights1),
            w2 = Flatten(weights2),
            b1 = (float[])bias1.Clone(),
            b2 = (float[])bias2.Clone(),
            wValue = (float[])weightsValue.Clone(),
            bValue = biasValue
        };
        File.WriteAllText(path, JsonUtility.ToJson(data, true));
    }

    public static NunuProximalPolicyOptimization Load(string path)
    {
        TextAsset model = Resources.Load<TextAsset>(path);
        if (model == null)
            throw new FileNotFoundException(path);

        NetworkData data = JsonUtility.FromJson<NetworkData>(model.text);
        NunuProximalPolicyOptimization ppo = new NunuProximalPolicyOptimization(data.inputSize, data.hiddenSize, data.outputSize);
        ppo.Unflatten(data.w1, ppo.weights1);
        ppo.Unflatten(data.w2, ppo.weights2);
        ppo.bias1 = (float[])data.b1.Clone();
        ppo.bias2 = (float[])data.b2.Clone();
        ppo.weightsValue = (float[])data.wValue.Clone();
        ppo.biasValue = data.bValue;
        return ppo;
    }

    private static float[] Flatten(float[,] matrix)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        float[] flat = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                flat[r * cols + c] = matrix[r, c];
        return flat;
    }

    private void Unflatten(float[] flat, float[,] matrix)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                matrix[r, c] = flat[r * cols + c];
    }
}


