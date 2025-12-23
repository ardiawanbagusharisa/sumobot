using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using Unity.InferenceEngine;
using UnityEngine;

class AIBot_ML_Classification : Bot
{
    public override string ID => "Bot_MLP";
    public override SkillType DefaultSkillType => SkillType.Boost;
    public override bool UseAsync => true;

    public Model runtimeModel;
    public Worker engine;
    private SumoAPI api;
    private bool isInitializing = false;
    private bool isGenerating = false;
    private readonly List<string> labels = new()
    {
        "FWD", "TL", "TR"
    };

    public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
    {
        if (state == BattleState.Battle_End)
        {
            engine.Dispose();
            ClearCommands();
        }
        else if (state == BattleState.Battle_Countdown)
        {
            CreateEngine();
        }
    }

    public override void OnBotCollision(BounceEvent bounceEvent)
    {
    }

    public override void OnBotInit(SumoAPI botAPI)
    {
        api = botAPI;
        CreateEngine();
    }
    public override void OnBotUpdate()
    {
        _ = Run(GenerateState());
        Submit();
    }

    async Task Run(float[] inputs)
    {
        if (isGenerating) return;
        isGenerating = true;

        Logger.Info($"[ML][Classification] RUN with inputs: {string.Join(", ", inputs.Select((x) => x.ToString()).ToList())}");

        Tensor<float> inputTensor = new(new TensorShape(1, 5), inputs);


        engine.Schedule(inputTensor);

        Tensor<float> outputTensorSkill = await (engine.PeekOutput("skill") as Tensor<float>).ReadbackAndCloneAsync();
        Tensor<float> outputTensorDash = await (engine.PeekOutput("dash") as Tensor<float>).ReadbackAndCloneAsync();
        Tensor<float> outputTensorMovement = await (engine.PeekOutput("movement") as Tensor<float>).ReadbackAndCloneAsync();
        Tensor<float> outputTensorDuration = await (engine.PeekOutput("duration") as Tensor<float>).ReadbackAndCloneAsync();

        var outputTensorSkillRes = outputTensorSkill.DownloadToArray()[0];
        var outputTensorDashRes = outputTensorDash.DownloadToArray()[0];
        var outputTensorActionRes = outputTensorMovement.DownloadToArray();
        var outputTensorDurationRes = outputTensorDuration.DownloadToArray()[0];

        inputTensor.Dispose();
        outputTensorSkill.Dispose();
        outputTensorDash.Dispose();
        outputTensorMovement.Dispose();
        outputTensorDuration.Dispose();

        int predictedIndex = ArgMax(outputTensorActionRes);
        string predictedLabel = labels[predictedIndex];

        Logger.Info($"[ML][Classification] Output Detail\nSkillProb: {outputTensorSkillRes:F2}\nDashProb: {outputTensorDashRes:F2}\nMovement: {predictedLabel}\nDuration: {outputTensorDurationRes:F2}");

        if (outputTensorSkillRes > 0.5f)
            Enqueue(new SkillAction(InputType.Script, DefaultSkillType.ToActionType()));

        if (outputTensorDashRes > 0.5f)
            Enqueue(new DashAction(InputType.Script));

        Enqueue(GetAction(predictedLabel, outputTensorDurationRes));

        isGenerating = false;
    }

    public float[] GenerateState()
    {
        var signedAngle = api.Angle();
        var signedAngleScore = api.Angle(normalized: true);
        var distanceToEnemy = 1 - api.DistanceNormalized();
        var nearArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition).magnitude / api.BattleInfo.ArenaRadius;

        var centerToMe = api.Distance(targetPos: api.MyRobot.Position, oriPos: api.BattleInfo.ArenaPosition).normalized;

        var zRot = api.MyRobot.Rotation % 360f;
        if (zRot < 0) zRot += 360f;
        Vector2 facingDir = Quaternion.Euler(0, 0, zRot) * Vector2.up;

        var facingToOutside = Vector2.Dot(facingDir, centerToMe);
        return new[] { signedAngle, signedAngleScore, distanceToEnemy, nearArena, facingToOutside };
    }

    private ISumoAction GetAction(string predictedAction, float duration)
    {
        switch (predictedAction)
        {
            case "FWD":
                return new AccelerateAction(InputType.Script, Mathf.Max(0.1f, duration));
            case "TL":
                return new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(0.1f, duration));
            case "TR":
                return new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(0.1f, duration));
        }
        return new AccelerateAction(InputType.Script, 0.1f);
    }


    private void CreateEngine()
    {
        if (isInitializing)
            return;

        isInitializing = true;

        engine?.Dispose();

        if (runtimeModel == null)
        {
            ModelAsset modelAsset = Resources.Load($"ML/Models/Classification/ml_enhanced_actions") as ModelAsset;
            runtimeModel = ModelLoader.Load(modelAsset);
        }

        engine = new Worker(runtimeModel, BackendType.GPUPixel);
        isInitializing = false;
        Logger.Info($"Engine worker of MLP created!");
    }

    public override void OnBotDestroy()
    {
        engine?.Dispose();
    }

    int ArgMax(float[] array)
    {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > maxValue)
            {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}