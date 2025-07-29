using System;
using System.Collections.Generic;
using System.Linq;
using SumoCore;
using SumoInput;
using UnityEngine;

namespace SumoBot.RuleBased.Utility
{
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
