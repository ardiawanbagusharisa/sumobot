using SumoBot;
using SumoCore;
using SumoManager;
using SumoInput;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;

public class AIBot_PPO : Bot
{
    #region Bot Template Properties
    public override string ID => "Bot_PPO";
    public override SkillType DefaultSkillType => SkillType.Boost;
    private SumoAPI api;
    #endregion

    #region Configs Properties
    public bool loadModel = true;
    public bool saveModel = false;
    public string modelFileName = "PPO_Model";
    public string csvLogFileName = "PPO_LearningLog.csv";
    #endregion

    #region NN & PPO Properties
    public float clipEpsilon = 0.2f;
    public float learningRate = 0.01f;
    public float valueLossCoefficient = 0.3f;
    public float entropyCoefficient = 0.02f;
    public int ppoEpochs = 15;
    public int batchSize = 64;
    public float maxEpisodeTime = 10f;
    public int maxTotalEpisodes = 100;
    public int totalEpisodes = 0;
    public float angleThreshold = 10f;

    private ProximalPolicyOptimization PPO;
    private List<ProximalPolicyOptimization.Experience> episodeBuffer = new List<ProximalPolicyOptimization.Experience>();
    private float[] lastState;
    private int lastAction;
    private int stepInEpisode = 0;
    private float timer = 0f;

    public int input = 4;   // 4 Inputs: Position X, Position Y, Angle, Distance Normalized 
    public int hidden = 16; // Hidden layer size
    public int output = 5;  // 5 Outputs: // Accelerate, TurnLeft, TurnRight, Dash, SkillBoost
    #endregion

    #region Bot Template Methods
    public override void OnBotInit(SumoAPI botAPI)
    {
        api = botAPI;
        string path = "ML/Models/RL/" + modelFileName;
        if (loadModel)
        {
            PPO = ProximalPolicyOptimization.Load(path);
            Logger.Info($"Loaded PPO from {path}");
        }
        else
        {
            PPO = new ProximalPolicyOptimization(input, hidden, output); // 4 Inputs: Position X, Position Y, Angle, Distance Normalized
            Logger.Info("Created new PPO");
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
            Logger.Info($"Angle: {api.Angle():F2}, Dist: {api.Distance().magnitude:F2}, DistN: {api.DistanceNormalized():F2}");
            Logger.Info($"MyPos: {api.MyRobot.Position}, MyRot: {api.MyRobot.Rotation:F2}, EnemyPos: {api.EnemyRobot.Position}, EnemyRot: {api.EnemyRobot.Rotation:F2}");
        }
        if (totalEpisodes >= maxTotalEpisodes)
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }

    public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
    {
        if (state == BattleState.Battle_End)
        {
            ClearCommands();
#if UNITY_EDITOR
            if (saveModel)
            {
                string path = "Assets/Resources/ML/Models/RL/" + modelFileName + ".json";
                PPO.Save(path);
                Logger.Info($"Saved PPO to {path}");
            }
#endif
        }
    }

    public override void OnBotCollision(BounceEvent bounceEvent)
    {
        ClearCommands();
    }
    #endregion

    #region Bot Logic Methods
    private void ThinkAndAct()
    {
        float posX = Mathf.Clamp01(api.MyRobot.Position.x / api.BattleInfo.ArenaRadius);
        float posY = Mathf.Clamp01(api.MyRobot.Position.y / api.BattleInfo.ArenaRadius);
        float angle = api.Angle();
        float distanceNormalized = Mathf.Clamp01(api.DistanceNormalized());
        float angleInDur = Mathf.Abs(angle) / api.MyRobot.RotateSpeed;

        float[] state = new float[] { posX, posY, angle, distanceNormalized };
        (float[] probs, float value) = PPO.Forward(state);
        int action = SampleAction(probs);
        float reward = CalculateReward(angle, distanceNormalized);
        float entropy = CalculateEntropy(probs);

        bool done = api.Distance(api.MyRobot.Position, api.BattleInfo.ArenaPosition).magnitude > api.BattleInfo.ArenaRadius + 0.5f
                || api.Distance(api.EnemyRobot.Position, api.BattleInfo.ArenaPosition).magnitude > api.BattleInfo.ArenaRadius + 0.5f
                || api.BattleInfo.TimeLeft <= 0f || api.BattleInfo.CurrentState == BattleState.Battle_End;

        float advantage = 0f, clippedRatio = 0f, returnVal = reward;
        if (lastState != null)
        {
            float nextValue = done ? 0f : PPO.Forward(state).value;
            returnVal = reward + 0.99f * nextValue;
            advantage = returnVal - value;
            clippedRatio = Mathf.Clamp(probs[lastAction] / (episodeBuffer.Count > 0 ? episodeBuffer[episodeBuffer.Count - 1].oldProb + 1e-10f : 1f), 1f - clipEpsilon, 1f + clipEpsilon);
            episodeBuffer.Add(new ProximalPolicyOptimization.Experience(lastState, lastAction, reward, state, probs[lastAction], value, done));
        }

        // Perform Action
        if (action == 0 && Mathf.Abs(angle) <= angleThreshold)
            Enqueue(new AccelerateAction(InputType.Script));
        else if (action == 1 && angle > angleThreshold)
            Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(0.1f, Mathf.Clamp01(angleInDur))));
        else if (action == 2 && angle < -angleThreshold)
            Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(0.1f, Mathf.Clamp01(angleInDur))));
        else if (action == 3 && Mathf.Abs(angle) <= angleThreshold && !api.MyRobot.IsDashOnCooldown)
            Enqueue(new DashAction(InputType.Script));
        else if (action == 4 && Mathf.Abs(angle) <= angleThreshold && !api.MyRobot.Skill.IsSkillOnCooldown)
            Enqueue(new SkillAction(InputType.Script));

        // Calculate loss for logging 
        float policyLoss = 0f, valueLoss = 0f, totalLoss = 0f;
        if (episodeBuffer.Count > 0)
        {
            var lastExp = episodeBuffer[episodeBuffer.Count - 1];
            (policyLoss, valueLoss, totalLoss) = PPO.CalculateLoss(lastExp, probs, value, returnVal, clipEpsilon, valueLossCoefficient, entropyCoefficient);
        }

#if UNITY_EDITOR
        if (saveModel)
            LogPPOLearning(totalEpisodes, stepInEpisode, posX, posY, angle, distanceNormalized, probs, action, reward, entropy, policyLoss, valueLoss, totalLoss, advantage, clippedRatio, returnVal);
#endif

        lastState = state;
        lastAction = action;

        if (done || timer >= maxEpisodeTime)
        {
            if (episodeBuffer.Count > 0)
            {
                PPO.Train(episodeBuffer, ppoEpochs, batchSize, clipEpsilon, learningRate, valueLossCoefficient, entropyCoefficient);
            }
            episodeBuffer.Clear();
            ResetEpisode();
        }
        stepInEpisode++;
    }

    private float CalculateReward(float angle, float distNormalized)
    {
        float reward = 0f;
        BattleInfoAPI battleInfo = api.BattleInfo;
        SumoBotAPI myRobot = api.MyRobot;
        SumoBotAPI enemyRobot = api.EnemyRobot;
        float enemyDistance = api.Distance(enemyRobot.Position, battleInfo.ArenaPosition).magnitude;
        float myDistance = api.Distance(myRobot.Position, battleInfo.ArenaPosition).magnitude;
        float outThreshold = battleInfo.ArenaRadius + 0.4f;

        reward += (1 - Mathf.Abs(angle) / 180) * 2f;
        reward += (1 - distNormalized) * 2f;
        reward -= Mathf.Abs(angle) / 180 * 0.1f;

        float endReward = 0f;
        if (enemyDistance > outThreshold && enemyDistance > myDistance)
            endReward += 5f * 0.2f;

        reward += endReward;

        // Reward for using dash and skill
        if (myRobot.Skill.IsActive && Mathf.Abs(angle) < angleThreshold)
            reward += 0.5f; // Reward for using skill when facing enemy
        if (myRobot.IsDashOnCooldown && Mathf.Abs(angle) < angleThreshold)
            reward += 0.5f;

        Logger.Info($"Reward: {reward:F2} | EndReward: {endReward:F2} | Angle: {(1 - Mathf.Abs(angle) / 180f):F2} | Dist: {(1 - distNormalized):F2}");

        return reward;
    }

    private float CalculateEntropy(float[] probs)
    {
        float entropy = 0f;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] > 0)
                entropy -= probs[i] * Mathf.Log(probs[i]);
        return entropy;
    }

    private int SampleAction(float[] probs)
    {
        float[] probsBeforeNoise = (float[])probs.Clone(); // Store original probs for logging
        float sum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            if (float.IsNaN(probs[i]) || probs[i] < 0)
            {
                Debug.LogWarning($"Invalid prob[{i}]={probs[i]}, setting to 0.0001");
                probs[i] = 0.0001f;
            }
            sum += probs[i];
        }
        if (sum <= 0)
        {
            Debug.LogWarning("Invalid action probs sum <= 0, resetting to uniform");
            for (int i = 0; i < probs.Length; i++)
                probs[i] = 1f / probs.Length;
            sum = 1f;
        }
        else
        {
            for (int i = 0; i < probs.Length; i++)
                probs[i] = probs[i] / sum * 0.95f + 0.02f / probs.Length; // Reduced noise to 2%
        }

        float r = UnityEngine.Random.value;
        float cumulative = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative)
                return i;
        }
        return probs.Length - 1;
    }

    private void LogPPOLearning(int episode, int step, float posX, float posY, float angle, float distN, float[] actionProbs, int action, float reward, float entropy, float policyLoss, float valueLoss, float totalLoss, float advantage, float clippedRatio, float returnVal)
    {
        string path = "Assets/Resources/ML/Models/RL/" + csvLogFileName;
        bool writeHeader = !File.Exists(path);
        using (StreamWriter sw = new StreamWriter(path, true))
        {
            if (writeHeader)
            {
                sw.WriteLine("Episode,Step,PosX,PosY,Angle,DistN,AccProb,TurnLeftProb,TurnRightProb,DashProb,SkillProb,Action,Reward,Entropy,OriginalEntropy,PolicyLoss,ValueLoss,TotalLoss,Advantage,ClippedRatio,Return");
            }
            float originalEntropy = CalculateEntropy(actionProbs); // Use actionProbs before noise for original
            sw.WriteLine($"{episode},{step},{posX:F2},{posY:F2},{angle:F2},{distN:F2}," +
                         $"{actionProbs[0]:F4},{actionProbs[1]:F4},{actionProbs[2]:F4},{actionProbs[3]:F4},{actionProbs[4]:F4}," +
                         $"{action},{reward:F4},{entropy:F4},{originalEntropy:F4},{policyLoss:F4},{valueLoss:F4},{totalLoss:F4},{advantage:F4},{clippedRatio:F4},{returnVal:F4}");
        }
    }

    private void ResetEpisode()
    {
        timer = 0f;
        totalEpisodes++;
        lastState = null;
        stepInEpisode = 0;
    }

    private void OnApplicationQuit()
    {
#if UNITY_EDITOR
        if (saveModel)
        {
            string path = "Assets/Resources/ML/Models/RL/" + modelFileName + ".json";
            PPO.Save(path);
            Logger.Info($"Saved PPO to {path}");
        }
#endif
    }
    #endregion
}

public class ProximalPolicyOptimization
{
    #region Experience structure
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
    #endregion

    #region NetworkData structure
    [Serializable]
    private class NetworkData
    {
        public int inputSize, hiddenSize, outputSize;
        public float[] w1, w2, b1, b2, wValue;
        public float bValue;
    }
    #endregion

    #region Network Properties
    private float[,] weights1; // Input to hidden
    private float[] bias1;
    private float[,] weights2; // Hidden to policy output
    private float[] bias2;
    private float[] weightsValue; // Hidden to value output
    private float biasValue;
    private int inputSize, hiddenSize, outputSize;
    private System.Random rand = new System.Random();
    #endregion

    public ProximalPolicyOptimization(int inputSize, int hiddenSize, int outputSize)
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

    private void Randomize(float[] biases)
    {
        for (int i = 0; i < biases.Length; i++)
            biases[i] = (float)(rand.NextDouble() * 0.1 - 0.05);
    }

    public (float[] probs, float value) Forward(float[] input)
    {
        if (input.Length != inputSize)
        {
            Logger.Error($"Input size mismatch: expected {inputSize}, got {input.Length}");
            return (new float[outputSize], 0f);
        }

        float[] hidden = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            hidden[i] = bias1[i];
            for (int j = 0; j < inputSize; j++)
                hidden[i] += input[j] * weights1[j, i];
            hidden[i] = MathF.Tanh(hidden[i]);
        }

        float[] output = new float[outputSize];
        float maxLogit = float.MinValue;
        for (int i = 0; i < outputSize; i++)
        {
            output[i] = bias2[i];
            for (int j = 0; j < hiddenSize; j++)
                output[i] += hidden[j] * weights2[j, i];
            if (output[i] > maxLogit)
                maxLogit = output[i];
        }

        float sum = 0f;
        for (int i = 0; i < outputSize; i++)
        {
            output[i] = MathF.Exp(Mathf.Clamp(output[i] - maxLogit, -10f, 10f));
            sum += output[i];
        }
        for (int i = 0; i < outputSize; i++)
            output[i] = sum > 0 ? output[i] / sum : 1f / outputSize;
        for (int i = 0; i < outputSize; i++)
            output[i] = Mathf.Clamp(output[i], 0.05f, 0.95f);
        sum = 0f;
        for (int i = 0; i < outputSize; i++)
            sum += output[i];
        for (int i = 0; i < outputSize; i++)
            output[i] /= sum;

        float value = biasValue;
        for (int i = 0; i < hiddenSize; i++)
            value += hidden[i] * weightsValue[i];

        return (output, value);
    }

    public (float policyLoss, float valueLoss, float totalLoss) CalculateLoss(Experience exp, float[] newProbs, float newValue, float returnVal, float clipEpsilon, float valueLossCoefficient, float entropyCoefficient)
    {
        float ratio = newProbs[exp.action] / (exp.oldProb + 1e-10f);
        float advantage = returnVal - exp.value;
        float policyLoss = -Mathf.Min(ratio * advantage, Mathf.Clamp(ratio, 1f - clipEpsilon, 1f + clipEpsilon) * advantage);
        float valueLoss = (newValue - returnVal) * (newValue - returnVal);
        float entropy = CalculateEntropy(newProbs);
        float totalLoss = policyLoss + valueLossCoefficient * valueLoss - entropyCoefficient * entropy;

        if (float.IsNaN(totalLoss) || float.IsInfinity(totalLoss))
        {
            Debug.LogWarning($"Invalid loss: policyLoss={policyLoss}, valueLoss={valueLoss}, entropy={entropy}, ratio={ratio}, advantage={advantage}");
        }

        return (policyLoss, valueLoss, totalLoss);
    }

    public void Train(List<Experience> buffer, int epochs, int batchSize, float clipEpsilon, float learningRate, float valueLossCoefficient, float entropyCoefficient)
    {
        float[] returns = CalculateReturns(buffer);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalPolicyLoss = 0f, totalValueLoss = 0f, totalLoss = 0f;
            int batchCount = 0;

            for (int i = 0; i < buffer.Count; i += batchSize)
            {
                for (int j = i; j < Math.Min(i + batchSize, buffer.Count); j++)
                {
                    var exp = buffer[j];
                    (float[] probs, float value) = Forward(exp.state);
                    float ratio = probs[exp.action] / (exp.oldProb + 1e-10f);
                    float advantage = returns[j] - exp.value;
                    float policyLoss = -Mathf.Min(ratio * advantage, Mathf.Clamp(ratio, 1f - clipEpsilon, 1f + clipEpsilon) * advantage);
                    float valueLoss = (value - returns[j]) * (value - returns[j]);
                    float entropy = CalculateEntropy(probs);
                    float loss = policyLoss + valueLossCoefficient * valueLoss - entropyCoefficient * entropy;

                    totalPolicyLoss += policyLoss;
                    totalValueLoss += valueLoss;
                    totalLoss += loss;
                    batchCount++;

                    float[] dOutput = new float[outputSize];
                    dOutput[exp.action] = loss;

                    float[] hidden = new float[hiddenSize];
                    for (int k = 0; k < hiddenSize; k++)
                    {
                        hidden[k] = bias1[k];
                        for (int m = 0; m < inputSize; m++)
                            hidden[k] += exp.state[m] * weights1[m, k];
                        hidden[k] = MathF.Tanh(hidden[k]);
                    }

                    float dValue = 2f * (value - returns[j]) * valueLossCoefficient;

                    float[] dHidden = new float[hiddenSize];
                    for (int k = 0; k < hiddenSize; k++)
                    {
                        float error = dValue * weightsValue[k];
                        for (int m = 0; m < outputSize; m++)
                            error += dOutput[m] * weights2[k, m];
                        dHidden[k] = error * (1 - hidden[k] * hidden[k]);
                    }

                    for (int k = 0; k < hiddenSize; k++)
                        for (int m = 0; m < outputSize; m++)
                            weights2[k, m] -= learningRate * Mathf.Clamp(dOutput[m] * hidden[k], -0.1f, 0.1f);

                    for (int k = 0; k < outputSize; k++)
                        bias2[k] -= learningRate * Mathf.Clamp(dOutput[k], -0.1f, 0.1f);

                    for (int k = 0; k < hiddenSize; k++)
                        weightsValue[k] -= learningRate * Mathf.Clamp(dValue * hidden[k], -0.1f, 0.1f);

                    biasValue -= learningRate * Mathf.Clamp(dValue, -0.1f, 0.1f);

                    for (int k = 0; k < inputSize; k++)
                        for (int m = 0; m < hiddenSize; m++)
                            weights1[k, m] -= learningRate * Mathf.Clamp(dHidden[m] * exp.state[k], -0.1f, 0.1f);

                    for (int k = 0; k < hiddenSize; k++)
                        bias1[k] -= learningRate * Mathf.Clamp(dHidden[k], -0.1f, 0.1f);

                    // Check for large weights
                    for (int k = 0; k < hiddenSize; k++)
                        for (int m = 0; m < inputSize; m++)
                            if (Mathf.Abs(weights1[m, k]) > 10f)
                                Debug.LogWarning($"Large weight in weights1[{m},{k}]={weights1[m, k]}");
                    for (int k = 0; k < hiddenSize; k++)
                        for (int m = 0; m < outputSize; m++)
                            if (Mathf.Abs(weights2[k, m]) > 10f)
                                Debug.LogWarning($"Large weight in weights2[{k},{m}]={weights2[k, m]}");
                    for (int k = 0; k < weightsValue.Length; k++)
                        if (Mathf.Abs(weightsValue[k]) > 10f)
                            Debug.LogWarning($"Large weight in weightsValue[{k}]={weightsValue[k]}");
                }
            }

            if (batchCount > 0)
            {
                Logger.Info($"Epoch {epoch}: Avg Policy Loss={totalPolicyLoss / batchCount:F4}, Avg Value Loss={totalValueLoss / batchCount:F4}, Avg Total Loss={totalLoss / batchCount:F4}");
            }
        }

        float w1Mag = CalculateWeightMagnitude(weights1);
        float w2Mag = CalculateWeightMagnitude(weights2);
        float wValueMag = CalculateWeightMagnitude(weightsValue);
        Logger.Info($"Weight Magnitudes: W1={w1Mag:F3}, W2={w2Mag:F3}, WValue={wValueMag:F3}");
    }

    private float CalculateEntropy(float[] probs)
    {
        float entropy = 0f;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] > 0)
                entropy -= probs[i] * Mathf.Log(probs[i]);
        float sum = 0f;
        for (int i = 0; i < probs.Length; i++) sum += probs[i];
        Logger.Info($"Probs sum: {sum:F6}, Entropy: {entropy:F4}"); // Debug prob sum
        return entropy;
    }

    private float CalculateWeightMagnitude(float[] weights)
    {
        float sum = 0f;
        for (int i = 0; i < weights.Length; i++)
            sum += weights[i] * weights[i];
        return Mathf.Sqrt(sum);
    }

    private float CalculateWeightMagnitude(float[,] weights)
    {
        float sum = 0f;
        for (int i = 0; i < weights.GetLength(0); i++)
            for (int j = 0; j < weights.GetLength(1); j++)
                sum += weights[i, j] * weights[i, j];
        return Mathf.Sqrt(sum);
    }

    private float[] CalculateReturns(List<Experience> buffer)
    {
        float[] returns = new float[buffer.Count];
        float discountedReturn = 0f;
        for (int i = buffer.Count - 1; i >= 0; i--)
        {
            discountedReturn = buffer[i].reward + 0.99f * discountedReturn;
            returns[i] = discountedReturn;
        }
        return returns;
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
            b2 = (float[])bias2.Clone(),
            wValue = (float[])weightsValue.Clone(),
            bValue = biasValue
        };
        string json = JsonUtility.ToJson(d, true);
        File.WriteAllText(path, json);
    }

    public static ProximalPolicyOptimization Load(string path)
    {
        TextAsset model = Resources.Load<TextAsset>(path);
        if (model == null)
            throw new FileNotFoundException(path);

        NetworkData d = JsonUtility.FromJson<NetworkData>(model.text);
        ProximalPolicyOptimization ppo = new ProximalPolicyOptimization(d.inputSize, d.hiddenSize, d.outputSize);
        ppo.Unflatten(d.w1, ppo.weights1);
        ppo.Unflatten(d.w2, ppo.weights2);
        ppo.bias1 = (float[])d.b1.Clone();
        ppo.bias2 = (float[])d.b2.Clone();
        ppo.weightsValue = (float[])d.wValue.Clone();
        ppo.biasValue = d.bValue;
        return ppo;
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