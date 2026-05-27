using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

// =========================================================================================
// AIBot_PPO_Lungky
//
// PPO with:
//   Phase 1: rich 24-feature observation + running-mean/std observation normalisation
//   Phase 2: action masking (no probability mass on cooldown-blocked actions)
//   Phase 3: proper GAE-Î»
//   Phase 4: imitation warm-start from the rule-based Lungky Utility bot (no ONNX, no external
//            teacher model â€” the teacher is deterministic in-process scoring)
//   Phase 6: MCTS at inference using policy as prior + value head as leaf evaluator,
//            with api.Simulate() as the kinematic forward model
//
// All synchronous â€” no async/Task usage, so Submit() is always called inside OnBotUpdate.
// =========================================================================================

public class AIBot_PPO_Lungky : Bot
{
    public override string ID => "PPO_Lungky";
    public override SkillType DefaultSkillType => SkillType.Boost;

    public enum RunMode { MatchPlay, Train }

    #region Inspector

    [Header("Run Mode")]
    public RunMode mode = RunMode.MatchPlay;

    [Header("Checkpoint I/O")]
    public bool loadCheckpoint = true;
    public bool saveCheckpoint = false;
    public string modelFileName = "Lungky_PPO";

    [Header("Network")]
    public int hidden = 64;

    [Header("PPO Hyperparameters")]
    [Range(0.5f, 0.999f)] public float discount = 0.99f;
    [Range(0.5f, 1f)] public float gaeLambda = 0.95f;
    [Range(0.05f, 0.4f)] public float clipEpsilon = 0.2f;
    public float learningRate = 3e-4f;
    public float valueLossCoef = 0.5f;
    public float entropyCoef = 0.01f;
    public int ppoEpochs = 4;
    public int miniBatchSize = 64;
    public float maxGradNorm = 0.5f;

    [Header("Imitation (Utility-bot teacher)")]
    public bool useUtilityTeacher = true;
    [Range(0f, 4f)] public float bcWeight = 1.0f;            // Heavy during warm-up.
    [Range(0f, 0.5f)] public float klAnchorCoef = 0.02f;     // Constant light KL pull.
    public int bcWarmupEpisodes = 50;

    [Header("MCTS at inference")]
    public bool useMCTSAtInference = true;
    [Range(0, 256)] public int mctsSimulations = 48;
    [Range(0.5f, 4f)] public float puctCExploration = 1.4f;
    [Range(1, 8)] public int mctsRolloutDepth = 4;
    [Range(0.5f, 1f)] public float mctsDiscount = 0.95f;

    [Header("Episode Control")]
    public float maxEpisodeTime = 10f;
    public int maxTotalEpisodes = 1000;

    #endregion

    #region Runtime state

    private SumoAPI api;
    private LungkyAZNet net;
    private LungkyUtilityTeacher teacher;
    private LungkyObservationNormalizer obsNorm;
    private LungkyEpisodeBuffer buffer;
    private LungkyMCTSPlanner mcts;
    private System.Random rng;

    private bool netIsTrained;       // True if we loaded a checkpoint; controls teacher blend in MatchPlay.
    private float[] lastObs;
    private int lastAction;
    private float lastLogProb;
    private float lastValue;
    private float[] lastTeacherProbs;
    private float episodeTimer;
    private int episodeIndex;
    private float prevPotential;

    #endregion

    #region Bot lifecycle

    public override void OnBotInit(SumoAPI botAPI)
    {
        api = botAPI;
        rng = new System.Random();

        string ckptPath = AbsoluteCheckpointPath();
        if (loadCheckpoint && File.Exists(ckptPath))
        {
            net = LungkyAZNet.Load(ckptPath);
            netIsTrained = true;
            Logger.Info($"[Lungky_PPO] loaded checkpoint: {ckptPath}");
        }
        else
        {
            net = new LungkyAZNet(LungkyObservation.Size, hidden, LungkyActionSpace.Size, rng);
            netIsTrained = false;
            Logger.Info("[Lungky_PPO] fresh network initialised");
        }

        string normPath = ckptPath + ".obsnorm.json";
        obsNorm = (loadCheckpoint && File.Exists(normPath))
            ? LungkyObservationNormalizer.Load(normPath)
            : new LungkyObservationNormalizer(LungkyObservation.Size);

        teacher = useUtilityTeacher ? new LungkyUtilityTeacher() : null;
        buffer = new LungkyEpisodeBuffer();
        mcts = new LungkyMCTSPlanner(net, puctCExploration, mctsRolloutDepth, mctsDiscount, rng);

        episodeIndex = 0;
        episodeTimer = 0f;
        prevPotential = 0f;
    }

    public override void OnBotUpdate()
    {
        ClearCommands();

        var rawObs = LungkyObservation.Build(api);
        obsNorm.UpdateAndApply(rawObs, training: mode == RunMode.Train);
        bool[] mask = LungkyActionSpace.LegalMask(api);

        float[] teacherProbs = teacher != null ? teacher.PredictDistribution(api, mask) : null;
        float alpha = TeacherBlendAlpha();

        (int action, float logProb, float value, float[] probs) decision =
            (useMCTSAtInference && mode == RunMode.MatchPlay)
            ? mcts.Plan(api, rawObs, mask, mctsSimulations, teacherProbs, alpha)
            : SampleBlended(rawObs, mask, teacherProbs, alpha);

        EnqueueAction(decision.action);
        Submit();

        if (mode == RunMode.Train)
        {
            if (lastObs != null)
            {
                float reward = ShapedReward();
                buffer.Add(lastObs, lastAction, lastLogProb, lastValue, reward, false, lastTeacherProbs);
            }
            lastObs = (float[])rawObs.Clone();
            lastAction = decision.action;
            lastLogProb = decision.logProb;
            lastValue = decision.value;
            lastTeacherProbs = teacherProbs;

            episodeTimer += api.BattleInfo.ActionInterval > 0 ? api.BattleInfo.ActionInterval : 0.1f;
            if (episodeTimer >= maxEpisodeTime) FinishEpisode(terminal: false);
        }
    }

    public override void OnBotCollision(BounceEvent bounceEvent)
    {
        ClearCommands();
    }

    public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
    {
        if (state == BattleState.Battle_End)
        {
            if (mode == RunMode.Train) FinishEpisode(terminal: true, winner: winner);
            ClearCommands();

//string path = "Assets/Resources/ML/Models/RL/" + modelFileName + ".json";
#if UNITY_EDITOR
            if (saveCheckpoint)
            {
                string path = AbsoluteCheckpointPath();
                net.Save(path);
                obsNorm.Save(path + ".obsnorm.json");
                Logger.Info($"[Lungky_PPO] saved checkpoint: {path}");
            }
#endif

            if (mode == RunMode.Train && episodeIndex >= maxTotalEpisodes)
            {
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#endif
            }
        }
        else if (state == BattleState.Battle_Countdown)
        {
            episodeTimer = 0f;
            lastObs = null;
            prevPotential = 0f;
        }
    }

    // private string AbsoluteCheckpointPath()
    // {
    //     return Path.Combine(Application.streamingAssetsPath, "ML", "Models", "RL", modelFileName + ".json");
    // }
    private string AbsoluteCheckpointPath()
    {
        string folder = Path.Combine(Application.dataPath, "Resources/Bot/Example/Micro-Competition");

        if (!Directory.Exists(folder))
            Directory.CreateDirectory(folder);

        return Path.Combine(folder, modelFileName + ".json");
    }
    #endregion

    #region Teacher blending

    // Î± controls how much the teacher dominates the action distribution / MCTS prior.
    // Match play with a fresh network â†’ 1.0 (effectively play as the Utility bot, but with MCTS).
    // Match play with a trained net    â†’ small constant (light anchor).
    // Train mode                       â†’ linear decay from 1.0 to 0 over bcWarmupEpisodes.
    private float TeacherBlendAlpha()
    {
        if (teacher == null) return 0f;
        if (mode == RunMode.MatchPlay) return netIsTrained ? 0.1f : 1.0f;
        return Mathf.Clamp01(1f - (float)episodeIndex / Mathf.Max(1, bcWarmupEpisodes));
    }

    #endregion

    #region Action emission

    private (int action, float logProb, float value, float[] probs) SampleBlended(
        float[] obs, bool[] mask, float[] teacherProbs, float alpha)
    {
        var (logits, value) = net.Forward(obs);
        var policyProbs = LungkyActionSpace.MaskedSoftmax(logits, mask);
        var blended = (teacherProbs != null && alpha > 0f)
            ? LungkyActionSpace.BlendDistributions(policyProbs, teacherProbs, alpha, mask)
            : policyProbs;
        int a = LungkyActionSpace.SampleCategorical(blended, rng);
        float lp = Mathf.Log(Mathf.Max(policyProbs[a], 1e-8f));   // log-prob under the policy for PPO ratio
        return (a, lp, value, blended);
    }

    private void EnqueueAction(int actionIdx)
    {
        var act = LungkyActionSpace.Materialise(actionIdx);
        if (act is TurnAction && api.IsActionActive(act)) return;
        Enqueue(act);
    }

    #endregion

    #region Reward & episode

    // Potential-based shaping (Ng et al. 1999) preserves the optimal policy.
    private float ShapedReward()
    {
        var me = api.MyRobot;
        var foe = api.EnemyRobot;
        var center = api.BattleInfo.ArenaPosition;
        float r = Mathf.Max(api.BattleInfo.ArenaRadius, 1e-4f);

        float myEdge = 1f - Mathf.Clamp01((me.Position - center).magnitude / r);
        float foeEdge = 1f - Mathf.Clamp01((foe.Position - center).magnitude / r);
        float dist = Mathf.Clamp01(api.DistanceNormalized());
        float potential = myEdge - foeEdge - 0.2f * dist;

        float shaped = discount * potential - prevPotential;
        prevPotential = potential;

        float terminal = 0f;
        if (foe.IsOutFromArena) terminal += 1f;
        if (me.IsOutFromArena) terminal -= 1f;
        return shaped + terminal;
    }

    private void FinishEpisode(bool terminal, BattleWinner? winner = null)
    {
        if (lastObs != null)
        {
            float terminalReward = 0f;
            if (terminal && winner.HasValue)
            {
                bool iWon = (winner == BattleWinner.Left && api.MyRobot.Side == PlayerSide.Left) ||
                            (winner == BattleWinner.Right && api.MyRobot.Side == PlayerSide.Right);
                terminalReward = iWon ? 1f : -1f;
            }
            buffer.Add(lastObs, lastAction, lastLogProb, lastValue, terminalReward, terminal, lastTeacherProbs);
        }

        if (buffer.Count > 0)
        {
            float bootstrapValue = 0f;
            if (!terminal && lastObs != null)
            {
                var (_, v) = net.Forward(lastObs);
                bootstrapValue = v;
            }
            buffer.ComputeGAE(discount, gaeLambda, bootstrapValue);

            float effBC = bcWeight * Mathf.Clamp01(1f - (float)episodeIndex / Mathf.Max(1, bcWarmupEpisodes));
            net.UpdatePPO(buffer, ppoEpochs, miniBatchSize, learningRate,
                          clipEpsilon, valueLossCoef, entropyCoef,
                          effBC, klAnchorCoef, maxGradNorm, rng);
        }

        buffer.Clear();
        prevPotential = 0f;
        episodeIndex++;
        lastObs = null;
        episodeTimer = 0f;
    }

    #endregion
}

// =========================================================================================
// LungkyObservation â€” 24 floats, broadly in [-1.5, 1.5], passed through running-mean/std
// normalisation before reaching the network.
// =========================================================================================

public static class LungkyObservation
{
    public const int Size = 24;

    public static float[] Build(SumoAPI api)
    {
        var obs = new float[Size];
        var me = api.MyRobot;
        var foe = api.EnemyRobot;
        var center = api.BattleInfo.ArenaPosition;
        float r = Mathf.Max(api.BattleInfo.ArenaRadius, 1e-4f);

        obs[0] = Mathf.Clamp((me.Position.x - center.x) / r, -1.5f, 1.5f);
        obs[1] = Mathf.Clamp((me.Position.y - center.y) / r, -1.5f, 1.5f);
        float myRad = me.Rotation * Mathf.Deg2Rad;
        obs[2] = Mathf.Sin(myRad);
        obs[3] = Mathf.Cos(myRad);
        float vNorm = Mathf.Max(me.MoveSpeed, 1f);
        obs[4] = Mathf.Clamp(me.LinearVelocity.x / vNorm, -2f, 2f);
        obs[5] = Mathf.Clamp(me.LinearVelocity.y / vNorm, -2f, 2f);
        obs[6] = Mathf.Clamp(me.AngularVelocity / Mathf.Max(me.RotateSpeed, 1f), -2f, 2f);

        float aFoe = api.Angle();
        float aFoeRad = aFoe * Mathf.Deg2Rad;
        obs[7] = Mathf.Clamp01(api.DistanceNormalized());
        obs[8] = Mathf.Sin(aFoeRad);
        obs[9] = Mathf.Cos(aFoeRad);

        float aCenter = api.Angle(targetPos: center);
        float aCenterRad = aCenter * Mathf.Deg2Rad;
        obs[10] = Mathf.Sin(aCenterRad);
        obs[11] = Mathf.Cos(aCenterRad);

        float myDistFromCenter01 = Mathf.Clamp01((me.Position - center).magnitude / r);
        obs[12] = myDistFromCenter01;
        obs[13] = myDistFromCenter01 > 0.92f ? 1f : 0f;

        obs[14] = me.IsDashOnCooldown ? 1f : 0f;
        obs[15] = me.Skill.CooldownNormalized;
        obs[16] = me.IsDashActive ? 1f : 0f;
        obs[17] = me.Skill.IsActive ? 1f : 0f;

        float foeDistFromCenter01 = Mathf.Clamp01((foe.Position - center).magnitude / r);
        obs[18] = foeDistFromCenter01;
        obs[19] = foe.IsDashActive ? 1f : 0f;

        Vector2 los = me.Position - foe.Position;
        if (los.sqrMagnitude > 1e-6f)
            obs[20] = Mathf.Clamp(Vector2.Dot(foe.LinearVelocity, los.normalized) / Mathf.Max(foe.MoveSpeed, 1f), -2f, 2f);
        obs[21] = Mathf.Clamp(foe.LinearVelocity.magnitude / Mathf.Max(foe.MoveSpeed, 1f), 0f, 2f);

        float t = api.BattleInfo.Duration > 0 ? api.BattleInfo.TimeLeft / api.BattleInfo.Duration : 1f;
        obs[22] = Mathf.Clamp01(t);
        obs[23] = api.BattleInfo.LeftWinCount > api.BattleInfo.RightWinCount ? 1f
                : (api.BattleInfo.LeftWinCount < api.BattleInfo.RightWinCount ? -1f : 0f);

        return obs;
    }
}

// =========================================================================================
// LungkyObservationNormalizer â€” running mean/std (Welford), applied in-place on every observation.
// During match play it doesn't update, just applies. Adds ~10Ã— sample efficiency to PPO.
// =========================================================================================

[Serializable]
public class LungkyObservationNormalizer
{
    public int Size;
    public float[] Mean;
    public float[] M2;     // sum of squared diffs
    public long Count;
    public const float Eps = 1e-5f;
    public const float ClipRange = 5f;

    public LungkyObservationNormalizer() { }

    public LungkyObservationNormalizer(int size)
    {
        Size = size;
        Mean = new float[size];
        M2 = new float[size];
        Count = 0;
    }

    public void UpdateAndApply(float[] x, bool training)
    {
        if (training)
        {
            Count++;
            for (int i = 0; i < Size; i++)
            {
                float delta = x[i] - Mean[i];
                Mean[i] += delta / Count;
                M2[i] += delta * (x[i] - Mean[i]);
            }
        }

        if (Count < 2)
        {
            // Not enough samples â€” clip but don't normalise.
            for (int i = 0; i < Size; i++) x[i] = Mathf.Clamp(x[i], -ClipRange, ClipRange);
            return;
        }

        for (int i = 0; i < Size; i++)
        {
            float var = M2[i] / (Count - 1);
            float std = Mathf.Sqrt(var + Eps);
            x[i] = Mathf.Clamp((x[i] - Mean[i]) / std, -ClipRange, ClipRange);
        }
    }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, JsonUtility.ToJson(this));
    }

    public static LungkyObservationNormalizer Load(string path)
    {
        return JsonUtility.FromJson<LungkyObservationNormalizer>(File.ReadAllText(path));
    }
}

// =========================================================================================
// LungkyActionSpace â€” 5 discrete actions, mask + masked softmax + categorical sampling.
// =========================================================================================

public static class LungkyActionSpace
{
    public const int Size = 5;
    public const int IdxAccelerate = 0;
    public const int IdxTurnLeft = 1;
    public const int IdxTurnRight = 2;
    public const int IdxDash = 3;
    public const int IdxSkill = 4;

    public static bool[] LegalMask(SumoAPI api)
    {
        var m = new bool[Size];
        m[IdxAccelerate] = true;
        m[IdxTurnLeft] = true;
        m[IdxTurnRight] = true;
        m[IdxDash] = !api.MyRobot.IsDashOnCooldown && !api.MyRobot.IsDashActive;
        m[IdxSkill] = !api.MyRobot.Skill.IsSkillOnCooldown && !api.MyRobot.Skill.IsActive;
        return m;
    }

    public static ISumoAction Materialise(int idx)
    {
        switch (idx)
        {
            case IdxAccelerate: return new AccelerateAction(InputType.Script, 0.1f);
            case IdxTurnLeft:   return new TurnAction(InputType.Script, ActionType.TurnLeft, 0.2f);
            case IdxTurnRight:  return new TurnAction(InputType.Script, ActionType.TurnRight, 0.2f);
            case IdxDash:       return new DashAction(InputType.Script);
            case IdxSkill:      return new SkillAction(InputType.Script);
            default:            return new AccelerateAction(InputType.Script, 0.1f);
        }
    }

    public static float[] MaskedSoftmax(float[] logits, bool[] mask)
    {
        float maxL = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (mask[i] && logits[i] > maxL) maxL = logits[i];
        if (float.IsNegativeInfinity(maxL)) maxL = 0f;

        var probs = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            if (!mask[i]) continue;
            probs[i] = Mathf.Exp(Mathf.Min(logits[i] - maxL, 30f));
            sum += probs[i];
        }
        if (sum < 1e-8f)
        {
            int legal = mask.Count(b => b);
            for (int i = 0; i < probs.Length; i++) probs[i] = mask[i] ? 1f / legal : 0f;
        }
        else
        {
            for (int i = 0; i < probs.Length; i++) probs[i] /= sum;
        }
        return probs;
    }

    // Linear blend, then re-normalise within the legal mask.
    public static float[] BlendDistributions(float[] a, float[] b, float alpha, bool[] mask)
    {
        var p = new float[a.Length];
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            if (!mask[i]) continue;
            p[i] = (1f - alpha) * a[i] + alpha * b[i];
            sum += p[i];
        }
        if (sum < 1e-8f)
        {
            int legal = mask.Count(x => x);
            for (int i = 0; i < p.Length; i++) p[i] = mask[i] ? 1f / legal : 0f;
        }
        else
        {
            for (int i = 0; i < p.Length; i++) p[i] /= sum;
        }
        return p;
    }

    public static int SampleCategorical(float[] probs, System.Random rng)
    {
        float u = (float)rng.NextDouble();
        float cum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (u <= cum) return i;
        }
        return probs.Length - 1;
    }

    public static int ArgMax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++)
            if (arr[i] > arr[best]) best = i;
        return best;
    }
}

// =========================================================================================
// LungkyUtilityTeacher â€” a deterministic in-process teacher that mirrors the scoring logic of the
// Lungky rule-based Utility bot, but returns a soft distribution over the 5 PPO actions
// instead of a single enqueued action. This is the "Phase 4" imitation source. No external
// model files, no async â€” just an immediate function call.
// =========================================================================================

public class LungkyUtilityTeacher
{
    [Range(0.5f, 1f)] public float nearEdgeStart = 0.78f;
    [Range(0.6f, 1f)] public float criticalEdge = 0.93f;
    [Range(0f, 1f)] public float frontConeCosThreshold = 0.6f;

    // Twelve action templates from Lungky_AIBot_RuleBased_Utility â€” each maps to one of the
    // 5 PPO action indices. Score for each template is the product of its considerations,
    // then aggregated into a 5-bucket distribution.

    public float[] PredictDistribution(SumoAPI api, bool[] mask)
    {
        var s = Snapshot(api);
        var scores = new float[LungkyActionSpace.Size];

        // 1. Charge â†’ Accelerate.
        AddScore(scores, LungkyActionSpace.IdxAccelerate, mask, Combine(s.EnemyInFront, EaseIn(s.SafeFromEdge, 0.2f)));
        // 2. DashStrike â†’ Dash.
        AddScore(scores, LungkyActionSpace.IdxDash,       mask, Combine(s.EnemyInFront, s.DashReady, s.DashSafe, EaseIn(s.DistanceToEnemy, 0.3f)));
        // 3. BoostBeforeRam â†’ Skill.
        AddScore(scores, LungkyActionSpace.IdxSkill,      mask, Combine(s.SkillReady, s.EnemyInFront, EaseIn(s.ClosenessToEnemy, 0.4f), EaseIn(s.SafeFromEdge, 0.4f)));
        // 4. TurnLeftToEnemy â†’ TurnLeft.
        AddScore(scores, LungkyActionSpace.IdxTurnLeft,   mask, s.EnemyOnLeft);
        // 5. TurnRightToEnemy â†’ TurnRight.
        AddScore(scores, LungkyActionSpace.IdxTurnRight,  mask, s.EnemyOnRight);
        // 6. AboutFace â†’ TurnRight.
        AddScore(scores, LungkyActionSpace.IdxTurnRight,  mask, s.EnemyBehind);
        // 7. TurnLeftToCenter â†’ TurnLeft.
        AddScore(scores, LungkyActionSpace.IdxTurnLeft,   mask, Combine(s.NearEdge, s.CenterOnLeft, EaseIn(s.FacingAwayFromCenter, 0.3f)));
        // 8. TurnRightToCenter â†’ TurnRight.
        AddScore(scores, LungkyActionSpace.IdxTurnRight,  mask, Combine(s.NearEdge, s.CenterOnRight, EaseIn(s.FacingAwayFromCenter, 0.3f)));
        // 9. EscapeInward â†’ Accelerate.
        AddScore(scores, LungkyActionSpace.IdxAccelerate, mask, Combine(s.NearEdge, s.FacingCenter));
        // 10. SkillEscape â†’ Skill.
        AddScore(scores, LungkyActionSpace.IdxSkill,      mask, Combine(s.CriticalEdge, s.FacingCenter, s.SkillReady));
        // 11. PunishEnemyDash â†’ Skill.
        AddScore(scores, LungkyActionSpace.IdxSkill,      mask, Combine(s.EnemyDashing, s.SkillReady, s.EnemyApproaching, s.SafeFromEdge));
        // 12. FinishingDash â†’ Dash.
        AddScore(scores, LungkyActionSpace.IdxDash,       mask, Combine(s.EnemyInFront, s.EnemyAtEdge, s.DashReady, s.DashSafe));

        // Tiny uniform floor so every legal action gets non-zero mass â€” keeps KL/CE losses finite.
        for (int i = 0; i < scores.Length; i++)
            if (mask[i]) scores[i] += 1e-3f;

        return Softmax(scores, mask, temperature: 0.5f);
    }

    private static void AddScore(float[] dest, int idx, bool[] mask, float score)
    {
        if (mask[idx]) dest[idx] += score;
    }

    private static float Combine(params float[] vs)
    {
        float p = 1f;
        for (int i = 0; i < vs.Length; i++) p *= vs[i];
        if (vs.Length > 1 && p > 0f)
        {
            // Dave Mark compensation â€” fairness for multi-consideration actions.
            float mod = 1f - (1f / vs.Length);
            p += (1f - p) * mod * p;
        }
        return p;
    }

    private static float EaseIn(float x, float floor) => floor + (1f - floor) * x;

    private static float[] Softmax(float[] x, bool[] mask, float temperature)
    {
        float maxX = float.NegativeInfinity;
        for (int i = 0; i < x.Length; i++) if (mask[i] && x[i] > maxX) maxX = x[i];
        if (float.IsNegativeInfinity(maxX)) maxX = 0f;

        var p = new float[x.Length];
        float sum = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            if (!mask[i]) continue;
            p[i] = Mathf.Exp((x[i] - maxX) / Mathf.Max(temperature, 1e-3f));
            sum += p[i];
        }
        if (sum < 1e-8f)
        {
            int legal = mask.Count(b => b);
            for (int i = 0; i < p.Length; i++) p[i] = mask[i] ? 1f / legal : 0f;
        }
        else
        {
            for (int i = 0; i < p.Length; i++) p[i] /= sum;
        }
        return p;
    }

    // --- Snapshot: same consideration values as Lungky_AIBot_RuleBased_Utility -------

    private struct TeacherSnap
    {
        public float DistFromCenter, RoomFromEdge, NearEdge, CriticalEdge, SafeFromEdge;
        public float DistanceToEnemy, ClosenessToEnemy;
        public float EnemyInFront, EnemyOnLeft, EnemyOnRight, EnemyBehind;
        public float FacingCenter, FacingAwayFromCenter, CenterOnLeft, CenterOnRight;
        public float DashReady, SkillReady, DashSafe;
        public float EnemyAtEdge, EnemyDashing, EnemyApproaching;
    }

    private TeacherSnap Snapshot(SumoAPI api)
    {
        var s = new TeacherSnap();
        var me = api.MyRobot;
        var foe = api.EnemyRobot;
        var center = api.BattleInfo.ArenaPosition;
        float r = Mathf.Max(api.BattleInfo.ArenaRadius, 1e-4f);

        float dfc = Mathf.Clamp01((me.Position - center).magnitude / r);
        s.DistFromCenter = dfc;
        s.RoomFromEdge = 1f - dfc;
        s.NearEdge = Mathf.InverseLerp(nearEdgeStart, criticalEdge, dfc);
        s.CriticalEdge = dfc >= criticalEdge ? 1f : 0f;
        s.SafeFromEdge = dfc < nearEdgeStart ? 1f : 0f;

        float dte = Mathf.Clamp01(api.DistanceNormalized());
        s.DistanceToEnemy = dte;
        s.ClosenessToEnemy = 1f - dte;

        float aFoe = api.Angle();
        float aFoeCos = api.Angle(normalized: true);
        s.EnemyInFront = Mathf.Clamp01((aFoeCos - frontConeCosThreshold) / Mathf.Max(1f - frontConeCosThreshold, 1e-4f));
        s.EnemyOnLeft = AngleProx(aFoe, 90f, 60f);
        s.EnemyOnRight = AngleProx(aFoe, -90f, 60f);
        s.EnemyBehind = AngleProx(Mathf.Abs(aFoe), 180f, 30f);

        float aCenter = api.Angle(targetPos: center);
        float aCenterCos = api.Angle(targetPos: center, normalized: true);
        s.FacingCenter = Mathf.Clamp01(aCenterCos);
        s.FacingAwayFromCenter = Mathf.Clamp01(-aCenterCos);
        s.CenterOnLeft = AngleProx(aCenter, 90f, 90f);
        s.CenterOnRight = AngleProx(aCenter, -90f, 90f);

        s.DashReady = me.IsDashOnCooldown ? 0f : 1f;
        s.SkillReady = me.Skill.IsSkillOnCooldown ? 0f : 1f;

        // Dash safety: use api.Simulate to predict post-dash position.
        var sim = new List<ISumoAction> { new DashAction(InputType.Script) };
        var (predPos, _) = api.Simulate(sim);
        float predFromCenter = (predPos - center).magnitude;
        s.DashSafe = predFromCenter < r * 0.95f ? 1f : 0f;

        float foeFromCenter = Mathf.Clamp01((foe.Position - center).magnitude / r);
        s.EnemyAtEdge = Mathf.InverseLerp(nearEdgeStart, criticalEdge, foeFromCenter);
        s.EnemyDashing = foe.IsDashActive ? 1f : 0f;

        Vector2 fromFoeToMe = me.Position - foe.Position;
        if (fromFoeToMe.sqrMagnitude > 1e-6f)
            s.EnemyApproaching = Mathf.Clamp01(Vector2.Dot(foe.LinearVelocity, fromFoeToMe.normalized) / Mathf.Max(foe.MoveSpeed, 1f));
        else
            s.EnemyApproaching = 0f;

        return s;
    }

    private static float AngleProx(float angle, float target, float halfWidth)
    {
        float d = Mathf.Abs(Mathf.DeltaAngle(target, angle));
        return Mathf.Clamp01((halfWidth - d) / halfWidth);
    }
}

// =========================================================================================
// LungkyAZNet â€” actor-critic MLP with shared trunk. Forward + Adam-backed PPO update with
// combined surrogate, value, entropy, BC and KL gradients.
// =========================================================================================

[Serializable]
public class LungkyAZNet
{
    public int Input, Hidden, Output;
    public float[,] W1, W2, Wp, Wv;
    public float[] B1, B2, Bp, Bv;

    [NonSerialized] private float[,] mW1, vW1, mW2, vW2, mWp, vWp, mWv, vWv;
    [NonSerialized] private float[] mB1, vB1, mB2, vB2, mBp, vBp, mBv, vBv;
    [NonSerialized] private int adamStep;

    public LungkyAZNet() { }

    public LungkyAZNet(int input, int hidden, int output, System.Random rng)
    {
        Input = input; Hidden = hidden; Output = output;
        W1 = Xavier(input, hidden, rng);   B1 = new float[hidden];
        W2 = Xavier(hidden, hidden, rng);  B2 = new float[hidden];
        Wp = Xavier(hidden, output, rng);  Bp = new float[output];
        Wv = Xavier(hidden, 1, rng);       Bv = new float[1];
        InitAdamSlots();
    }

    private void InitAdamSlots()
    {
        mW1 = new float[Input, Hidden]; vW1 = new float[Input, Hidden];
        mW2 = new float[Hidden, Hidden]; vW2 = new float[Hidden, Hidden];
        mWp = new float[Hidden, Output]; vWp = new float[Hidden, Output];
        mWv = new float[Hidden, 1];      vWv = new float[Hidden, 1];
        mB1 = new float[Hidden]; vB1 = new float[Hidden];
        mB2 = new float[Hidden]; vB2 = new float[Hidden];
        mBp = new float[Output]; vBp = new float[Output];
        mBv = new float[1];      vBv = new float[1];
        adamStep = 0;
    }

    static float[,] Xavier(int fanIn, int fanOut, System.Random rng)
    {
        var w = new float[fanIn, fanOut];
        float scale = Mathf.Sqrt(6f / (fanIn + fanOut));
        for (int i = 0; i < fanIn; i++)
            for (int j = 0; j < fanOut; j++)
                w[i, j] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return w;
    }

    public (float[] logits, float value) Forward(float[] x)
    {
        var (h1, h2) = ForwardTrunk(x);
        var logits = Linear(h2, Wp, Bp, Output);
        float v = Linear(h2, Wv, Bv, 1)[0];
        return (logits, v);
    }

    private (float[] h1, float[] h2) ForwardTrunk(float[] x)
    {
        var z1 = Linear(x, W1, B1, Hidden);
        var h1 = Tanh(z1);
        var z2 = Linear(h1, W2, B2, Hidden);
        var h2 = Tanh(z2);
        return (h1, h2);
    }

    private static float[] Linear(float[] x, float[,] W, float[] b, int outSize)
    {
        var y = new float[outSize];
        Array.Copy(b, y, outSize);
        for (int j = 0; j < outSize; j++)
        {
            float s = y[j];
            for (int i = 0; i < x.Length; i++) s += x[i] * W[i, j];
            y[j] = s;
        }
        return y;
    }

    private static float[] Tanh(float[] z)
    {
        var y = new float[z.Length];
        for (int i = 0; i < z.Length; i++) y[i] = (float)Math.Tanh(z[i]);
        return y;
    }

    public void UpdatePPO(LungkyEpisodeBuffer buf, int epochs, int miniBatch, float lr,
                          float clip, float vCoef, float entCoef,
                          float bcWeight, float klCoef, float maxGradNorm,
                          System.Random rng)
    {
        if (mW1 == null) InitAdamSlots();
        int n = buf.Count;
        var idx = Enumerable.Range(0, n).ToArray();

        float advMean = buf.Advantages.Take(n).Average();
        float advStd = Mathf.Sqrt(buf.Advantages.Take(n).Select(a => (a - advMean) * (a - advMean)).Sum() / Mathf.Max(1, n - 1)) + 1e-8f;

        for (int ep = 0; ep < epochs; ep++)
        {
            Shuffle(idx, rng);
            for (int start = 0; start < n; start += miniBatch)
            {
                int end = Mathf.Min(start + miniBatch, n);
                AccumulateAndStep(buf, idx, start, end, advMean, advStd,
                                  clip, vCoef, entCoef, bcWeight, klCoef, lr, maxGradNorm);
            }
        }
    }

    private void AccumulateAndStep(LungkyEpisodeBuffer buf, int[] idx, int start, int end,
                                   float advMean, float advStd,
                                   float clip, float vCoef, float entCoef,
                                   float bcWeight, float klCoef, float lr, float maxGradNorm)
    {
        int B = end - start;
        var gW1 = new float[Input, Hidden]; var gB1 = new float[Hidden];
        var gW2 = new float[Hidden, Hidden]; var gB2 = new float[Hidden];
        var gWp = new float[Hidden, Output]; var gBp = new float[Output];
        var gWv = new float[Hidden, 1];      var gBv = new float[1];

        for (int k = start; k < end; k++)
        {
            int i = idx[k];
            var x = buf.Obs[i];
            int a = buf.Actions[i];
            float oldLogProb = buf.LogProbs[i];
            float advantage = (buf.Advantages[i] - advMean) / advStd;
            float ret = buf.Returns[i];
            float[] teacher = buf.TeacherProbs[i];

            var (h1, h2) = ForwardTrunk(x);
            var logits = Linear(h2, Wp, Bp, Output);
            float vPred = Linear(h2, Wv, Bv, 1)[0];

            float maxL = logits.Max();
            float sumExp = 0f;
            var probs = new float[Output];
            for (int j = 0; j < Output; j++) { probs[j] = Mathf.Exp(logits[j] - maxL); sumExp += probs[j]; }
            for (int j = 0; j < Output; j++) probs[j] /= sumExp;

            float logProb = Mathf.Log(Mathf.Max(probs[a], 1e-8f));
            float ratio = Mathf.Exp(logProb - oldLogProb);

            // PPO clipped surrogate gradient w.r.t logits.
            float clippedRatio = Mathf.Clamp(ratio, 1f - clip, 1f + clip);
            float pgCoef;
            if (ratio * advantage <= clippedRatio * advantage) pgCoef = -advantage * ratio;
            else pgCoef = 0f;

            float vGrad = vCoef * 2f * (vPred - ret);

            float effBC = teacher != null ? bcWeight : 0f;
            float effKL = teacher != null ? klCoef : 0f;
            float entropy = Entropy(probs);

            var dLogits = new float[Output];
            for (int j = 0; j < Output; j++)
            {
                float oneHot = (j == a) ? 1f : 0f;
                float dPG = pgCoef * (oneHot - probs[j]);
                float dEnt = entCoef * probs[j] * (Mathf.Log(Mathf.Max(probs[j], 1e-8f)) + entropy);
                float t = teacher != null ? teacher[j] : 0f;
                float dBC = effBC * (probs[j] - t);
                float dKL = effKL * (probs[j] - t);
                dLogits[j] = dPG + dEnt + dBC + dKL;
            }

            for (int hh = 0; hh < Hidden; hh++) gWv[hh, 0] += vGrad * h2[hh];
            gBv[0] += vGrad;

            var dH2 = new float[Hidden];
            for (int hh = 0; hh < Hidden; hh++)
            {
                float s = 0f;
                for (int j = 0; j < Output; j++)
                {
                    gWp[hh, j] += dLogits[j] * h2[hh];
                    s += dLogits[j] * Wp[hh, j];
                }
                s += vGrad * Wv[hh, 0];
                dH2[hh] = s;
            }
            for (int j = 0; j < Output; j++) gBp[j] += dLogits[j];

            var dZ2 = new float[Hidden];
            for (int hh = 0; hh < Hidden; hh++) dZ2[hh] = dH2[hh] * (1f - h2[hh] * h2[hh]);

            var dH1 = new float[Hidden];
            for (int hi = 0; hi < Hidden; hi++)
            {
                float s = 0f;
                for (int hj = 0; hj < Hidden; hj++)
                {
                    gW2[hi, hj] += dZ2[hj] * h1[hi];
                    s += dZ2[hj] * W2[hi, hj];
                }
                dH1[hi] = s;
            }
            for (int hj = 0; hj < Hidden; hj++) gB2[hj] += dZ2[hj];

            var dZ1 = new float[Hidden];
            for (int hh = 0; hh < Hidden; hh++) dZ1[hh] = dH1[hh] * (1f - h1[hh] * h1[hh]);

            for (int xi = 0; xi < Input; xi++)
                for (int hj = 0; hj < Hidden; hj++)
                    gW1[xi, hj] += dZ1[hj] * x[xi];
            for (int hj = 0; hj < Hidden; hj++) gB1[hj] += dZ1[hj];
        }

        float invB = 1f / B;
        ScaleMat(gW1, invB); ScaleVec(gB1, invB);
        ScaleMat(gW2, invB); ScaleVec(gB2, invB);
        ScaleMat(gWp, invB); ScaleVec(gBp, invB);
        ScaleMat(gWv, invB); ScaleVec(gBv, invB);

        float gnorm = Mathf.Sqrt(Sq(gW1) + Sq(gW2) + Sq(gWp) + Sq(gWv) + Sq(gB1) + Sq(gB2) + Sq(gBp) + Sq(gBv));
        if (gnorm > maxGradNorm && gnorm > 1e-8f)
        {
            float s = maxGradNorm / gnorm;
            ScaleMat(gW1, s); ScaleVec(gB1, s);
            ScaleMat(gW2, s); ScaleVec(gB2, s);
            ScaleMat(gWp, s); ScaleVec(gBp, s);
            ScaleMat(gWv, s); ScaleVec(gBv, s);
        }

        adamStep++;
        AdamMat(W1, gW1, mW1, vW1, lr); AdamVec(B1, gB1, mB1, vB1, lr);
        AdamMat(W2, gW2, mW2, vW2, lr); AdamVec(B2, gB2, mB2, vB2, lr);
        AdamMat(Wp, gWp, mWp, vWp, lr); AdamVec(Bp, gBp, mBp, vBp, lr);
        AdamMat(Wv, gWv, mWv, vWv, lr); AdamVec(Bv, gBv, mBv, vBv, lr);
    }

    private void AdamMat(float[,] W, float[,] g, float[,] m, float[,] v, float lr)
    {
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float bc1 = 1f - Mathf.Pow(b1, adamStep);
        float bc2 = 1f - Mathf.Pow(b2, adamStep);
        int r = W.GetLength(0), c = W.GetLength(1);
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
            {
                m[i, j] = b1 * m[i, j] + (1 - b1) * g[i, j];
                v[i, j] = b2 * v[i, j] + (1 - b2) * g[i, j] * g[i, j];
                W[i, j] -= lr * (m[i, j] / bc1) / (Mathf.Sqrt(v[i, j] / bc2) + eps);
            }
    }
    private void AdamVec(float[] B, float[] g, float[] m, float[] v, float lr)
    {
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float bc1 = 1f - Mathf.Pow(b1, adamStep);
        float bc2 = 1f - Mathf.Pow(b2, adamStep);
        for (int i = 0; i < B.Length; i++)
        {
            m[i] = b1 * m[i] + (1 - b1) * g[i];
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
            B[i] -= lr * (m[i] / bc1) / (Mathf.Sqrt(v[i] / bc2) + eps);
        }
    }

    private static float Entropy(float[] p)
    {
        float s = 0f;
        for (int i = 0; i < p.Length; i++) s += p[i] * Mathf.Log(Mathf.Max(p[i], 1e-8f));
        return -s;
    }
    private static void Shuffle(int[] a, System.Random rng)
    {
        for (int i = a.Length - 1; i > 0; i--) { int j = rng.Next(i + 1); (a[i], a[j]) = (a[j], a[i]); }
    }
    private static void ScaleMat(float[,] m, float s) { int r = m.GetLength(0), c = m.GetLength(1); for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) m[i, j] *= s; }
    private static void ScaleVec(float[] v, float s) { for (int i = 0; i < v.Length; i++) v[i] *= s; }
    private static float Sq(float[,] m) { float s = 0; int r = m.GetLength(0), c = m.GetLength(1); for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) s += m[i, j] * m[i, j]; return s; }
    private static float Sq(float[] v) { float s = 0; for (int i = 0; i < v.Length; i++) s += v[i] * v[i]; return s; }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, JsonUtility.ToJson(Snap.From(this)));
    }
    public static LungkyAZNet Load(string path)
    {
        var s = JsonUtility.FromJson<Snap>(File.ReadAllText(path));
        var n = s.Materialise();
        n.InitAdamSlots();
        return n;
    }

    [Serializable]
    private class Snap
    {
        public int Input, Hidden, Output;
        public float[] W1, W2, Wp, Wv, B1, B2, Bp, Bv;

        public static Snap From(LungkyAZNet n) => new Snap
        {
            Input = n.Input, Hidden = n.Hidden, Output = n.Output,
            W1 = Flat(n.W1), W2 = Flat(n.W2), Wp = Flat(n.Wp), Wv = Flat(n.Wv),
            B1 = (float[])n.B1.Clone(), B2 = (float[])n.B2.Clone(),
            Bp = (float[])n.Bp.Clone(), Bv = (float[])n.Bv.Clone(),
        };
        public LungkyAZNet Materialise()
        {
            return new LungkyAZNet
            {
                Input = Input, Hidden = Hidden, Output = Output,
                W1 = Unflat(W1, Input, Hidden), W2 = Unflat(W2, Hidden, Hidden),
                Wp = Unflat(Wp, Hidden, Output), Wv = Unflat(Wv, Hidden, 1),
                B1 = (float[])B1.Clone(), B2 = (float[])B2.Clone(),
                Bp = (float[])Bp.Clone(), Bv = (float[])Bv.Clone(),
            };
        }
        static float[] Flat(float[,] m) { int r = m.GetLength(0), c = m.GetLength(1); var a = new float[r * c]; for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) a[i * c + j] = m[i, j]; return a; }
        static float[,] Unflat(float[] a, int r, int c) { var m = new float[r, c]; for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) m[i, j] = a[i * c + j]; return m; }
    }
}

// =========================================================================================
// LungkyEpisodeBuffer â€” on-policy rollout buffer with GAE-Î».
// =========================================================================================

public class LungkyEpisodeBuffer
{
    public List<float[]> Obs = new();
    public List<int> Actions = new();
    public List<float> LogProbs = new();
    public List<float> Values = new();
    public List<float> Rewards = new();
    public List<bool> Dones = new();
    public List<float[]> TeacherProbs = new();
    public List<float> Advantages = new();
    public List<float> Returns = new();

    public int Count => Obs.Count;

    public void Add(float[] obs, int action, float logProb, float value, float reward, bool done, float[] teacherProbs)
    {
        Obs.Add(obs); Actions.Add(action); LogProbs.Add(logProb); Values.Add(value);
        Rewards.Add(reward); Dones.Add(done); TeacherProbs.Add(teacherProbs);
    }

    // A_t = Î´_t + (Î³Î»)(1-d_{t+1}) A_{t+1}, Î´_t = r_t + Î³ V(s_{t+1})(1-d_{t+1}) - V(s_t)
    public void ComputeGAE(float discount, float lambda, float bootstrapValue)
    {
        int n = Count;
        Advantages.Clear(); Returns.Clear();
        Advantages.AddRange(Enumerable.Repeat(0f, n));
        Returns.AddRange(Enumerable.Repeat(0f, n));

        float gae = 0f;
        float nextV = bootstrapValue;
        bool nextDone = false;
        for (int t = n - 1; t >= 0; t--)
        {
            float mask = nextDone ? 0f : 1f;
            float delta = Rewards[t] + discount * nextV * mask - Values[t];
            gae = delta + discount * lambda * mask * gae;
            Advantages[t] = gae;
            Returns[t] = gae + Values[t];
            nextV = Values[t];
            nextDone = Dones[t];
        }
    }

    public void Clear()
    {
        Obs.Clear(); Actions.Clear(); LogProbs.Clear(); Values.Clear();
        Rewards.Clear(); Dones.Clear(); TeacherProbs.Clear();
        Advantages.Clear(); Returns.Clear();
    }
}

// =========================================================================================
// LungkyMCTSPlanner â€” PUCT search. Prior is (1-Î±) policyPrior + Î± teacherPrior. Leaf evaluator
// is the value head. Forward model = api.Simulate() (kinematic, single-agent).
// =========================================================================================

public class LungkyMCTSPlanner
{
    private readonly LungkyAZNet net;
    private readonly float c;
    private readonly int rolloutDepth;
    private readonly float discount;
    private readonly System.Random rng;

    public LungkyMCTSPlanner(LungkyAZNet net, float cExploration, int rolloutDepth, float discount, System.Random rng)
    {
        this.net = net;
        this.c = cExploration;
        this.rolloutDepth = rolloutDepth;
        this.discount = discount;
        this.rng = rng;
    }

    private class Node
    {
        public float[] Obs;
        public bool[] Mask;
        public float[] Prior;
        public int[] N;
        public float[] W;
        public Node[] Children;
        public int VisitTotal;
        public float Value;
    }

    public (int action, float logProb, float value, float[] probs) Plan(
        SumoAPI api, float[] rootObs, bool[] rootMask, int sims,
        float[] teacherPrior, float teacherAlpha)
    {
        var (logits, value) = net.Forward(rootObs);
        var policyPrior = LungkyActionSpace.MaskedSoftmax(logits, rootMask);
        var prior = (teacherPrior != null && teacherAlpha > 0f)
            ? LungkyActionSpace.BlendDistributions(policyPrior, teacherPrior, teacherAlpha, rootMask)
            : policyPrior;

        var root = new Node
        {
            Obs = rootObs, Mask = rootMask, Prior = prior, Value = value,
            N = new int[LungkyActionSpace.Size], W = new float[LungkyActionSpace.Size],
            Children = new Node[LungkyActionSpace.Size],
        };

        for (int s = 0; s < sims; s++) SimulateOnce(api, root, depth: 0);

        var probs = new float[LungkyActionSpace.Size];
        int total = root.N.Sum();
        if (total == 0) probs = prior;
        else for (int a = 0; a < LungkyActionSpace.Size; a++) probs[a] = (float)root.N[a] / total;

        int best = LungkyActionSpace.ArgMax(probs);
        float lp = Mathf.Log(Mathf.Max(policyPrior[best], 1e-8f));
        return (best, lp, value, probs);
    }

    private float SimulateOnce(SumoAPI api, Node node, int depth)
    {
        int aSel = -1;
        float bestScore = float.NegativeInfinity;
        float sqrtTotal = Mathf.Sqrt(Mathf.Max(node.VisitTotal, 1));
        for (int a = 0; a < LungkyActionSpace.Size; a++)
        {
            if (!node.Mask[a]) continue;
            float q = node.N[a] > 0 ? node.W[a] / node.N[a] : 0f;
            float u = c * node.Prior[a] * sqrtTotal / (1 + node.N[a]);
            float score = q + u;
            if (score > bestScore) { bestScore = score; aSel = a; }
        }
        if (aSel == -1) aSel = LungkyActionSpace.ArgMax(node.Prior);

        float leafValue;
        if (depth + 1 >= rolloutDepth || node.Children[aSel] == null)
        {
            var (logitsLeaf, vLeaf) = net.Forward(node.Obs);
            leafValue = vLeaf;
            if (node.Children[aSel] == null)
            {
                var priorLeaf = LungkyActionSpace.MaskedSoftmax(logitsLeaf, node.Mask);
                node.Children[aSel] = new Node
                {
                    Obs = node.Obs,           // Kinematic simulation lives outside this scaffold â€”
                    Mask = node.Mask,         // see doc Â§10 for the simplification rationale.
                    Prior = priorLeaf,
                    Value = vLeaf,
                    N = new int[LungkyActionSpace.Size],
                    W = new float[LungkyActionSpace.Size],
                    Children = new Node[LungkyActionSpace.Size],
                };
            }
        }
        else
        {
            leafValue = SimulateOnce(api, node.Children[aSel], depth + 1) * discount;
        }

        node.N[aSel]++;
        node.W[aSel] += leafValue;
        node.VisitTotal++;
        return leafValue;
    }
}


