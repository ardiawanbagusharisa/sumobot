using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using Unity.InferenceEngine;
using UnityEngine;

class AIBot_ML_Classification : Bot
{
    public override string ID => "MLP";

    public override SkillType SkillType => SkillType.Boost;

    public Model runtimeModel;
    public Worker engine;
    private SumoAPI api;
    private bool isInitializing = false;
    private readonly List<string> labels = new()
    {
        "Accelerate", "Dash", "SkillBoost", "TurnLeft", "TurnRight"
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
        var inputs = new float[] {
                api.MyRobot.Position.x,
                api.MyRobot.Position.y,
                Normalize360(api.MyRobot.Rotation),
                // api.MyRobot.LinearVelocity.magnitude,
                // api.MyRobot.AngularVelocity,
                // api.MyRobot.IsDashActive ? 1 : 0,
                // api.MyRobot.Skill.IsActive ? 1 : 0,
                // api.MyRobot.IsOutFromArena ? 1 : 0,
                // Enemy
                api.EnemyRobot.Position.x,
                api.EnemyRobot.Position.y,
                Normalize360(api.EnemyRobot.Rotation),
                // api.EnemyRobot.LinearVelocity.magnitude,
                // api.EnemyRobot.AngularVelocity,
                // api.EnemyRobot.IsDashActive ? 1 : 0,
                // api.EnemyRobot.Skill.IsActive ? 1 : 0,
                // api.EnemyRobot.IsOutFromArena ? 1 : 0,
             };
        Tensor<float> inputTensor = new(new TensorShape(1, 6), inputs);

        engine.Schedule(inputTensor);

        // output actions at 0
        Tensor<float> outputTensorAct = (engine.PeekOutput(0) as Tensor<float>).ReadbackAndClone();

        // output actions at 1
        Tensor<float> outputTensorDur = (engine.PeekOutput(1) as Tensor<float>).ReadbackAndClone();

        var outputTensorActRes = outputTensorAct.DownloadToArray();
        var outputTensorDurRes = outputTensorDur.DownloadToArray()[0];

        inputTensor.Dispose();
        outputTensorAct.Dispose();
        outputTensorDur.Dispose();

        int predictedIndex = ArgMax(outputTensorActRes);
        string predictedLabel = labels[predictedIndex];

        Debug.Log($"$outputTensorArr {string.Join(", ", outputTensorActRes.Select((x) => $"{x}"))}");
        Debug.Log($"$predictedLabel {predictedLabel}");
        Debug.Log($"$predictedDuraation {outputTensorDurRes}");
        var action = GetAction(predictedLabel, outputTensorDurRes);

        Enqueue(action);
        Submit();
    }

    float Normalize360(float angle)
    {
        angle %= 360f;
        if (angle < 0) angle += 360f;
        return angle;
    }

    private ISumoAction GetAction(string predictedAction, float duration)
    {
        switch (predictedAction)
        {
            case "Accelerate":
                return new AccelerateAction(InputType.Script, Mathf.Max(0.1f, duration));
            case "TurnLeft":
                return new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(0.1f, duration));
            case "TurnRight":
                return new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(0.1f, duration));
            case "Dash":
                return new DashAction(InputType.Script);
            case "SkillStone":
                return new SkillAction(InputType.Script, ActionType.SkillStone);
            case "SkillBoost":
                return new SkillAction(InputType.Script, ActionType.SkillBoost);
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
            ModelAsset modelAsset = Resources.Load("Models/ML/ml_6_4_layers") as ModelAsset;
            runtimeModel = ModelLoader.Load(modelAsset);
        }

        engine = new Worker(runtimeModel, BackendType.CPU);
        isInitializing = false;
        Debug.Log($"Engine worker of MLP created!");
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