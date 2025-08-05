using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using Unity.InferenceEngine;
using UnityEngine;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Threading.Tasks;
using System.Collections;
using UnityEngine.Rendering;

class AIBot_SLM_MicroGPT : Bot
{
    public override string ID => "SLM_MicroGPT";

    public override SkillType SkillType => SkillType.Boost;

    public Model runtimeModel;
    public Worker engine;
    private SumoAPI api;
    private bool isInitializing = false;
    private int interval = 0;
    private int runInterval = 15;
    private bool isGenerating = false;
    private readonly List<string> labels = new()
    {
        "Accelerate", "Dash", "SkillBoost", "TurnLeft", "TurnRight"
    };

    private Tokenizer tokenizer;

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
        tokenizer = new();
        tokenizer.LoadTokenizer();
        CreateEngine();
        SetRoutine(Run());
    }

    public override void OnBotUpdate()
    {
        Debug.Log($"Adding interval: {interval}");
        interval += 1;
    }

    IEnumerator Run()
    {
        while (true)
        {
            Debug.Log($"Running in interval: {interval}");
            if (interval % runInterval != 0)
            {
                string prompt = $"GameState: BotPos=[{api.MyRobot.Position.x:F2}, {api.MyRobot.Position.y:F2}], BotRot={Normalize360(api.MyRobot.Rotation)}, EnemyPos=[{api.EnemyRobot.Position.x:F2}, {api.EnemyRobot.Position.y:F2}], EnemyRot={Normalize360(api.EnemyRobot.Rotation)}";

                int[] input = tokenizer.Encode(prompt);
                int blockSize = 128;
                int vocabSize = tokenizer.stoi.Count;

                List<int> outputTokens = new(input);

                var currIters = 0;

                while (currIters < 300)
                {
                    int[] inputSlice = outputTokens
                            .Skip(Mathf.Max(0, outputTokens.Count - blockSize))
                            .ToArray();

                    var tensor = new Tensor<int>(new TensorShape(1, inputSlice.Length), inputSlice);

                    engine.SetInput("input_ids", tensor);
                    engine.Schedule();

                    using var output = (Tensor<float>)engine.PeekOutput(0).ReadbackAndClone();
                    float[] logits = output.DownloadToArray();
                    tensor.Dispose();
                    output.Dispose();

                    int nextToken = ArgMax(logits, inputSlice.Length - 1, vocabSize);
                    outputTokens.Add(nextToken);

                    // Break on newline token
                    if (tokenizer.itos.TryGetValue(nextToken, out char tokenChar) && tokenChar == '\n')
                        break;

                    currIters += 1;
                    yield return null;
                }

                string generated = tokenizer.Decode(outputTokens);
                Debug.Log("🧠 Generated Output:\n" + generated);
                
            }
            yield return null;
        }
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
            ModelAsset modelAsset = Resources.Load("Models/ML/LanguageModels/micro_gpt") as ModelAsset;
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

    int ArgMax(float[] logits, int seqIndex, int vocabSize)
    {
        int start = seqIndex * vocabSize;
        float max = float.MinValue;
        int argmax = 0;

        for (int i = 0; i < vocabSize; i++)
        {
            float val = logits[start + i];
            if (val > max)
            {
                max = val;
                argmax = i;
            }
        }
        return argmax;
    }


}

public class Tokenizer
{
    public Dictionary<char, int> stoi { get; private set; }
    public Dictionary<int, char> itos { get; private set; }

    public void LoadTokenizer()
    {
        TextAsset jsonText = Resources.Load<TextAsset>("Models/ML/LanguageModels/tokenizer");
        if (jsonText == null)
        {
            Debug.LogError("❌ Tokenizer file not found at Resources/Models/ML/tokenizer.json");
            return;
        }

        try
        {
            // Parse entire JSON
            JObject root = JObject.Parse(jsonText.text);
            JObject stoiObj = (JObject)root["stoi"];

            // Convert to Dictionary<char, int>
            stoi = stoiObj
                .Properties()
                .ToDictionary(
                    prop => prop.Name[0], // convert key string -> char
                    prop => (int)prop.Value
                );

            // Reverse
            itos = stoi.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

            Debug.Log($"✅ Tokenizer loaded. Vocab size: {stoi.Count}");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"❌ Failed to parse tokenizer: {ex.Message}");
        }
    }

    public int[] Encode(string input)
    {
        if (stoi == null) return new int[0];

        return input.Select(c => stoi.ContainsKey(c) ? stoi[c] : stoi[' ']).ToArray();
    }

    public string Decode(IEnumerable<int> tokens)
    {
        if (itos == null) return "";

        return string.Concat(tokens.Select(i => itos.ContainsKey(i) ? itos[i] : '?'));
    }
}
