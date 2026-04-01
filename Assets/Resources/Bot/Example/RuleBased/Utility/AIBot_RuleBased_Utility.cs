using System.Collections.Generic;
using UnityEngine;
using SumoCore;
using SumoManager;
using SumoLog;
using System.Linq;
using System;
using SumoInput;

namespace SumoBot.RuleBased.Utility
{
    public class AIBot_RuleBased_Utility : Bot
    {
        public override string ID => "Bot_Utility";
        public override SkillType DefaultSkillType => SkillType.Boost;

        private SumoAPI api;

        [SerializeField]
        private UtilityAI utility;

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;

            utility = new(Evaluator);
            utility.Actions.GenerateUtilityAction();
        }

        public override void OnBotUpdate()
        {
            ClearCommands();

            var scoredActions = utility.Run(2);

            foreach (var (action, _, reason) in scoredActions)
            {
                if (action is TurnAction && api.IsActionActive(action))
                    continue;

                action.Reason = reason;
                Enqueue(action);
            }

            Submit();
        }

        public float Evaluator(List<ConsiderationType> types)
        {
            float result = 0;

            float angleToEnemy = api.Angle();
            float distanceFromArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition).magnitude;
            bool isNearArena = distanceFromArena >= (api.BattleInfo.ArenaRadius * 0.9f);
            foreach (var type in types)
            {
                switch (type)
                {
                    case ConsiderationType.DistanceFromCenter:
                        result += 1 - api.DistanceNormalized(targetPos: api.BattleInfo.ArenaPosition);
                        break;
                    case ConsiderationType.DistanceToEnemy:
                        result += 1 - api.DistanceNormalized();
                        break;
                    case ConsiderationType.EnemyInSideRight:
                        {
                            float delta = Mathf.Abs(Mathf.DeltaAngle(270, angleToEnemy));
                            result += Mathf.Clamp01((90 - delta) / 90);
                            break;
                        }
                    case ConsiderationType.EnemyInSideLeft:
                        {
                            float delta = Mathf.Abs(Mathf.DeltaAngle(90, angleToEnemy));
                            result += Mathf.Clamp01((90 - delta) / 90);
                            break;
                        }
                    case ConsiderationType.EnemyInBack:
                        {
                            float delta = Mathf.Abs(Mathf.DeltaAngle(180, angleToEnemy));
                            result += Mathf.Clamp01((45 - delta) / 45);
                            break;
                        }
                    case ConsiderationType.SkillIsReady:
                        result += api.MyRobot.Skill.IsSkillOnCooldown ? -1 : 1;
                        break;
                    case ConsiderationType.DashIsReady:
                        result += api.MyRobot.IsDashOnCooldown ? -1 : 1;
                        break;
                    case ConsiderationType.NearArena:
                        {
                            result += isNearArena ? 1 : -1;
                            break;
                        }
                    case ConsiderationType.NotNearArena:
                        {
                            result += !isNearArena ? 1 : 0.01f;
                            break;
                        }
                    case ConsiderationType.EnemyInFront:
                        {
                            float angle = api.Angle(normalized: true);
                            result += angle > 0.7f ? angle : -1;
                            break;
                        }
                    case ConsiderationType.EnemyInFrontAndClose:
                        {
                            float dist = 1 - api.DistanceNormalized();
                            float angle = api.Angle(normalized: true);
                            float val = (angle + dist) / 2;
                            result += val > 0.8f ? val : -1;
                            break;
                        }
                    default:
                        break;
                }
            }
            return result / types.Count;
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
            ClearCommands();
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
        }
    }

    [Serializable]
    public class UtilityAI
    {
        public List<UtilityAction> Actions = new();
        public Func<List<ConsiderationType>, float> Evaluator;

        public UtilityAI(Func<List<ConsiderationType>, float> evaluator)
        {
            Evaluator = evaluator;
        }

        public List<(ISumoAction, float score, string Reason)> Run(int topNumAction = 2)
        {
            List<(ISumoAction action, float score, string Reason)> result = new();

            Actions.ForEach((x) =>
            {
                List<string> reasons = x.Considerations.Select((types) => string.Join(", ", types.InputType.Select((type) => type.ToString()))).ToList();
                result.Add((x.Action, x.Evaluate(Evaluator), string.Join(":", reasons)));
            });

            return result.OrderByDescending(x => x.score)
                .Take(topNumAction)
                .ToList();
        }
    }

    public enum ConsiderationType
    {
        DistanceFromCenter,
        DistanceToEnemy,
        EnemyInSideLeft,
        EnemyInSideRight,
        EnemyInFront,
        EnemyInBack,
        EnemyInFrontAndClose,
        DashIsReady,
        SkillIsReady,
        NearArena,
        NotNearArena,
    }

    [Serializable]
    public class Consideration
    {
        public List<ConsiderationType> InputType;
        public AnimationCurve ResponseCurve = AnimationCurve.Linear(0, 0, 1, 1);

        public float Evaluate(Func<List<ConsiderationType>, float> evaluator)
        {
            float rawValue = evaluator.Invoke(InputType);
            return ResponseCurve.Evaluate(Mathf.Clamp01(rawValue));
        }
    }

    [Serializable]
    public class UtilityAction
    {
        public ISumoAction Action;
        public List<Consideration> Considerations = new();

        public float Evaluate(Func<List<ConsiderationType>, float> evaluator)
        {
            float score = 1f;
            foreach (var c in Considerations)
            {
                float value = c.Evaluate(evaluator);
                score *= value;

                if (score == 0f) break; // Short-circuit if any is 0
            }
            return score;
        }
    }

    public static class UtilityAIExtension
    {
        public static void GenerateUtilityAction(this List<UtilityAction> actions) => actions.AddRange(new List<UtilityAction>
            {
                #region Accelerating
                new() {
                    Action = new AccelerateAction(InputType.Script, 0.1f),
                    Considerations = new List<Consideration>
                    {
                        new() {
                            InputType = new (){
                                ConsiderationType.DistanceToEnemy,
                                ConsiderationType.NotNearArena,
                            },
                            ResponseCurve = AnimationCurve.Linear(0f, 0f, 1f, 1f)
                        }
                    }
                },
                new() {
                    Action = new DashAction(InputType.Script),
                    Considerations = new List<Consideration>
                    {
                        new() {
                            InputType = new (){
                                ConsiderationType.EnemyInFront,
                                ConsiderationType.DashIsReady,
                                ConsiderationType.NotNearArena,
                            },
                            ResponseCurve = AnimationCurve.Linear(0f, 0f, 1f, 1f)
                        }
                    }
                },
            #endregion
                #region Turn
                new() {
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.35f),
                    Considerations = new List<Consideration>
                    {
                        new() {
                            InputType = new() { ConsiderationType.EnemyInSideLeft},
                            ResponseCurve = AnimationCurve.Linear(0f, 0f, 1f, 1f)
                        }
                    }
                },
                new() {
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.45f),
                    Considerations = new List<Consideration>
                    {
                        new() {
                            InputType = new() { ConsiderationType.EnemyInSideRight},
                            ResponseCurve = AnimationCurve.Linear(0f, 0f, 1f, 1f)
                        }
                    }
                },
                new() {
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.9f),
                    Considerations = new List<Consideration>
                    {
                        new() {
                            InputType = new() { ConsiderationType.EnemyInBack },
                            ResponseCurve = AnimationCurve.Linear(0f, 0f, 1f, 1f)
                        }
                    }
                },
                #endregion
                #region Skill
                new() {
                    Action = new SkillAction(InputType.Script),
                    Considerations = new List<Consideration>
                    {
                        new() {
                            InputType = new() {
                                ConsiderationType.SkillIsReady,
                                ConsiderationType.NearArena,
                            },
                            ResponseCurve = AnimationCurve.Linear(0f, 0f, 1f, 1f)
                        }
                    }
                },
                new() {
                    Action = new SkillAction(InputType.Script),
                    Considerations = new List<Consideration>
                    {
                        new() {
                            InputType = new() {
                                ConsiderationType.EnemyInFrontAndClose,
                                ConsiderationType.SkillIsReady,
                            },
                            ResponseCurve = AnimationCurve.Linear(0f, 0f, 1f, 1f)
                        }
                    }
                }
                #endregion
            });
    }
}
