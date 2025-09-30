using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using Unity.InferenceEngine;
using UnityEngine;
using System.Threading.Tasks;

namespace ML.LanguageModels
{
    public enum ModelState
    {
        Initializing,
        Initialzed,
        Standby,
    }
    class AIBot_SLM_ActionGPT : Bot
    {
        public override string ID => "Bot_SLM_ActionGPT";
        public override SkillType DefaultSkillType => SkillType;
        public override bool UseAsync => true;

        public SkillType SkillType = SkillType.Stone;

        public Model runtimeModel;
        public Worker engine;
        private SumoAPI api;
        private ModelState modelLoadState = ModelState.Standby;
        private BPETokenizer tokenizer;
        private BattleState state;
        private bool isGenerating = false;

        private string prompt;

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
            this.state = state;

            if (state == BattleState.Battle_Countdown)
            {
                prompt = null;
                ClearCommands();
                isGenerating = false;
                if (modelLoadState != ModelState.Initialzed)
                {
                    CreateEngine();
                }
            }
            else if (state == BattleState.Battle_End || state == BattleState.Battle_Reset)
            {
                engine?.Dispose();
                engine = null;
                modelLoadState = ModelState.Standby;
            }
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            string tokenizerPath = $"ML/Tokenizer/action_tokenizer";
            TextAsset tokenizerAsset = Resources.Load<TextAsset>(tokenizerPath);
            if (tokenizerAsset == null)
            {
                Logger.Error("Tokenizer JSON not found in Resources!");
                return;
            }
            tokenizer = new(tokenizerAsset);

            CreateEngine();
        }

        public override void OnBotUpdate()
        {
            if (isGenerating) return;
            prompt = GeneratePrompt();

            _ = RunInference();
            Submit();
        }

        public string GeneratePrompt()
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

            // return $"GameState: BotPos=[{api.MyRobot.Position.x:F2},{api.MyRobot.Position.y:F2}], BotRot={Normalize360(api.MyRobot.Rotation):F0}, EnemyPos=[{api.EnemyRobot.Position.x:F2},{api.EnemyRobot.Position.y:F2}], EnemyRot={Normalize360(api.EnemyRobot.Rotation):F0}, EnemyAngle={signedAngle:F2}, EnemyAngleScore={signedAngleScore:F2}, EnemyDistance={distanceToEnemy:F2}, BotArena={nearArena:F2} Result:";
            return $"BotPos=[{api.MyRobot.Position.x:F2},{api.MyRobot.Position.y:F2}] BotRot={Normalize360(api.MyRobot.Rotation):F0} EnemyPos=[{api.EnemyRobot.Position.x:F2},{api.EnemyRobot.Position.y:F2}] EnemyRot={Normalize360(api.EnemyRobot.Rotation):F0} EnemyAngle={signedAngle:F2} EnemyAngleScore={signedAngleScore:F2} EnemyDistance={distanceToEnemy:F2} BotArena={nearArena:F2} FaceArena={facingToOutside:F2} Result:";
        }

        async Task RunInference()
        {
            if (prompt != null)
            {
                Logger.Info($"prompt {prompt}");

                int[] input = tokenizer.Encode(prompt);
                int blockSize = 128;
                int vocabSize = tokenizer.vocab.Count();

                List<int> completeOutput = new(input);
                List<int> outputTokens = new();

                var currIters = 0;
                while (currIters < 15 && state == BattleState.Battle_Ongoing)
                {
                    isGenerating = true;
                    int[] inputSlice = completeOutput
                            .Skip(Mathf.Max(0, completeOutput.Count - blockSize))
                            .ToArray();

                    var tensor = new Tensor<int>(new TensorShape(1, inputSlice.Length), inputSlice);

                    engine?.SetInput("input_ids", tensor);
                    engine?.Schedule();
                    var tensorOutput = (Tensor<float>)engine.PeekOutput(0);
                    using var output = await tensorOutput.ReadbackAndCloneAsync();
                    float[] logits = output.DownloadToArray();
                    tensor.Dispose();
                    output.Dispose();

                    int nextToken = ArgMax(logits, inputSlice.Length - 1, vocabSize);

                    // Break on newline or empty token
                    if (tokenizer.idToToken[nextToken] == "\n" || tokenizer.idToToken[nextToken] == "<PAD>")
                        break;

                    completeOutput.Add(nextToken);
                    outputTokens.Add(nextToken);
                    currIters += 1;
                }

                isGenerating = false;

                string generated = tokenizer.Decode(outputTokens);
                Logger.Info("🧠 Generated Output:\n" + generated);

                if (generated != null)
                {
                    var splittedActs = generated.Trim().Split(" ");
                    foreach (var action in splittedActs)
                    {
                        string act = null;
                        float dur = 0;

                        if (action.StartsWith("FWD"))
                        {
                            act = "Accelerate";
                            dur = float.Parse(action.Substring(3, action.Length - 3));
                        }
                        else if (action.StartsWith("TL"))
                        {
                            act = "TurnLeft";
                            dur = float.Parse(action.Substring(2, action.Length - 2));
                        }
                        else if (action.StartsWith("TR"))
                        {
                            act = "TurnRight";
                            dur = float.Parse(action.Substring(2, action.Length - 2));
                        }
                        else if (action.StartsWith("SK"))
                        {
                            if (api.MyRobot.Skill.IsSkillOnCooldown)
                                continue;

                            act = "Skill";
                        }
                        else if (action.StartsWith("DS"))
                        {
                            if (api.MyRobot.IsDashOnCooldown)
                                continue;

                            act = "Dash";
                        }

                        if (act == null)
                        {
                            continue;
                        }

                        if (dur < 0.1)
                        {
                            dur = 0.1f;
                        }

                        Logger.Info($"action: {act} {dur}");
                        var parsedAct = GetAction(act, dur);
                        Enqueue(parsedAct);
                    }
                }
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
                    return new AccelerateAction(InputType.Script, Mathf.Max(api.BattleInfo.MinActionTime, duration));
                case "TurnLeft":
                    return new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(api.BattleInfo.MinActionTime, duration));
                case "TurnRight":
                    return new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(api.BattleInfo.MinActionTime, duration));
                case "Dash":
                    return new DashAction(InputType.Script);
                case "Skill":
                    return new SkillAction(InputType.Script);
            }
            return new AccelerateAction(InputType.Script, api.BattleInfo.MinActionTime);
        }

        private void CreateEngine()
        {
            if (modelLoadState == ModelState.Initializing)
                return;

            modelLoadState = ModelState.Initializing;

            engine?.Dispose();

            if (runtimeModel == null)
            {
                ModelAsset modelAsset = Resources.Load($"ML/Models/SLM/action_gpt_fp16") as ModelAsset;
                runtimeModel = ModelLoader.Load(modelAsset);
            }

            engine = new Worker(runtimeModel, BackendType.CPU);
            modelLoadState = ModelState.Initialzed;
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
}