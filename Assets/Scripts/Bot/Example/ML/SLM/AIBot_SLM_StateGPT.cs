// using System.Collections.Generic;
// using System.Linq;
// using System.Threading.Tasks;
// using SumoBot;
// using SumoCore;
// using SumoInput;
// using SumoManager;
// using Unity.InferenceEngine;
// using UnityEngine;

// namespace ML.LanguageModels
// {
//     class AIBot_SLM_StateGPT : Bot
//     {
//         public override string ID => "Sumo_StateGPT";

//         public override SkillType SkillType => SkillType.Boost;

//         public Model runtimeModel;
//         public Worker engine;
//         private SumoAPI api;
//         private bool isInitializing = false;

//         private BPETokenizer tokenizer;
//         private BattleState state;
//         private List<ISumoAction> sequentialActions = new();
//         private Queue<ISumoAction> ongoingAction = new();
//         private bool isGenerating = false;

//         private ISumoAction currentAction = null;
//         private float actionTimer = 0f;
//         private string prompt;

//         public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
//         {
//             this.state = state;
//             if (state == BattleState.Battle_End)
//             {
//                 engine.Dispose();
//                 sequentialActions.Clear();
//                 prompt = null;
//                 ClearCommands();
//             }
//             else if (state == BattleState.Battle_Countdown)
//             {
//                 prompt = GeneratePrompt();

//                 CreateEngine();
//                 // SetRoutine(RunInference());
//             }
//         }

//         public override void OnBotCollision(BounceEvent bounceEvent)
//         {
//         }

//         public override void OnBotInit(SumoAPI botAPI)
//         {
//             api = botAPI;
//             tokenizer = new("Assets/Resources/Models/ML/LanguageModels/state_tokenizer.json");
//         }


//         public override void OnBotUpdate()
//         {
//             if (isGenerating) return;

//             // // If no action is active, start the next one
//             // if (currentAction == null && sequentialActions.Count > 0)
//             // {
//             //     currentAction = sequentialActions[0];
//             //     sequentialActions.RemoveAt(0);
//             //     actionTimer = currentAction.Duration;
//             //     Enqueue(currentAction);
//             //     Submit();
//             // }

//             // // Countdown the active action
//             // if (currentAction != null)
//             // {
//             //     actionTimer = api.MyRobot.ActiveActions.TryGetValue(currentAction.Type, out float duration) ? duration : 0;
//             //     Debug.Log($"Current Action: {currentAction.FullName}, Timer: {actionTimer:F2}");
//             //     if (actionTimer <= 0f)
//             //     {
//             //         sequentialActions.Clear();
//             //         currentAction = null; // Done, move to next in queue next frame
//             //     }
//             // }
//             prompt = GeneratePrompt();
//             _ = RunInference();
//             // Submit();
//         }

//         public string GeneratePrompt()
//         {
//             return $"Given: BotPos=[{api.MyRobot.Position.x:F2},{api.MyRobot.Position.y:F2}], BotRot={Normalize360(api.MyRobot.Rotation):F2}, EnemyPos=[{api.EnemyRobot.Position.x:F2},{api.EnemyRobot.Position.y:F2}], EnemyRot={Normalize360(api.EnemyRobot.Rotation):F2},";
//         }

//         async Task RunInference()
//         {
//             if (sequentialActions.Count == 0 && currentAction == null && prompt != null)
//             {
//                 Debug.Log($"prompt {prompt}");

//                 int[] input = tokenizer.Encode(prompt);
//                 int blockSize = 128;
//                 int vocabSize = tokenizer.vocab.Count();

//                 List<int> completeOutput = new(input);
//                 List<int> outputTokens = new();

//                 var currIters = 0;
//                 while (currIters < 50 && state == BattleState.Battle_Ongoing)
//                 {
//                     isGenerating = true;
//                     int[] inputSlice = completeOutput
//                             .Skip(Mathf.Max(0, completeOutput.Count - blockSize))
//                             .ToArray();

//                     var tensor = new Tensor<int>(new TensorShape(1, inputSlice.Length), inputSlice);

//                     engine?.SetInput("input_ids", tensor);
//                     engine?.Schedule();

//                     using var output = (Tensor<float>)engine.PeekOutput(0).ReadbackAndClone();
//                     float[] logits = output.DownloadToArray();
//                     tensor.Dispose();
//                     output.Dispose();

//                     int nextToken = ArgMax(logits, inputSlice.Length - 1, vocabSize);

//                     if (tokenizer.idToToken[nextToken] == "<PAD>")
//                         break;

//                     completeOutput.Add(nextToken);
//                     outputTokens.Add(nextToken);
//                     await Task.Yield();
//                 }

//                 isGenerating = false;

//                 string generated = tokenizer.Decode(outputTokens);
//                 Debug.Log("🧠 Generated Output:\n" + generated);
//             }

//             // yield return null;
//         }

//         float Normalize360(float angle)
//         {
//             angle %= 360f;
//             if (angle < 0) angle += 360f;
//             return angle;
//         }

//         private ISumoAction GetAction(string predictedAction, float duration)
//         {
//             switch (predictedAction)
//             {
//                 case "Accelerate":
//                     return new AccelerateAction(InputType.Script, Mathf.Max(0.1f, duration));
//                 case "TurnLeft":
//                     return new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(0.1f, duration));
//                 case "TurnRight":
//                     return new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(0.1f, duration));
//                 case "Dash":
//                     return new DashAction(InputType.Script);
//                 case "SkillStone":
//                     return new SkillAction(InputType.Script, ActionType.SkillStone);
//                 case "SkillBoost":
//                     return new SkillAction(InputType.Script, ActionType.SkillBoost);
//             }
//             return new AccelerateAction(InputType.Script, 0.1f);
//         }

//         private void CreateEngine()
//         {
//             if (isInitializing)
//                 return;

//             isInitializing = true;

//             engine?.Dispose();

//             if (runtimeModel == null)
//             {
//                 ModelAsset modelAsset = Resources.Load("Models/ML/LanguageModels/state_gpt") as ModelAsset;
//                 runtimeModel = ModelLoader.Load(modelAsset);
//             }

//             engine = new Worker(runtimeModel, BackendType.CPU);
//             isInitializing = false;
//         }

//         public override void OnBotDestroy()
//         {
//             engine?.Dispose();
//         }

//         int ArgMax(float[] logits, int seqIndex, int vocabSize)
//         {
//             int start = seqIndex * vocabSize;
//             float max = float.MinValue;
//             int argmax = 0;

//             for (int i = 0; i < vocabSize; i++)
//             {
//                 float val = logits[start + i];
//                 if (val > max)
//                 {
//                     max = val;
//                     argmax = i;
//                 }
//             }
//             return argmax;
//         }


//     }
// }