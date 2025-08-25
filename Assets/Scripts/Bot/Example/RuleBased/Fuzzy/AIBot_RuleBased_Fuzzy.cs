
using System.Collections.Generic;
using System.Linq;
using SumoCore;
using SumoInput;
using SumoLog;
using SumoManager;
using UnityEngine;

namespace SumoBot.RuleBased.Fuzzy
{
    class AIBot_RuleBased_Fuzzy : Bot
    {
        public override string ID => Name;
        public override SkillType SkillType => DefaultSkillType;

        public string Name = "Fuzzy";
        public SkillType DefaultSkillType = SkillType.Boost;

        public FuzzySugeno Fuzzy = new();
        public int topActions = 2;

        private SumoAPI api;


        public override void OnBotUpdate()
        {
            ClearCommands();

            float distanceFromEnemy = api.DistanceNormalized();
            float enemyPosition = api.Angle() / 180;
            float enemySide = api.Angle(normalized: true);
            float distanceFromArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition).magnitude / api.BattleInfo.ArenaRadius;

            var centerToMe = api.Distance(targetPos: api.MyRobot.Position, oriPos: api.BattleInfo.ArenaPosition).normalized;

            var zRot = api.MyRobot.Rotation % 360f;
            if (zRot < 0) zRot += 360f;
            Vector2 facingDir = Quaternion.Euler(0, 0, zRot) * Vector2.up;

            var facingToOutside = Vector2.Dot(facingDir, centerToMe);

            List<float> inputs = new() {
                distanceFromEnemy,
                enemyPosition,
                enemySide,
                distanceFromArena,
                facingToOutside };

            var result = Fuzzy.Run(
                inputs: inputs,
                topActionsNum: topActions);

            foreach (var act in result)
            {
                if (act is SkillAction)
                {
                    if (act.Type != DefaultSkillType.ToActionType() || api.MyRobot.Skill.IsSkillOnCooldown)
                    {
                        continue;
                    }
                }

                if (act.Type == ActionType.Dash && api.MyRobot.IsDashOnCooldown)
                {
                    continue;
                }

                if (act is TurnAction && api.IsActionActive(act))
                {
                    continue;
                }

                Enqueue(act);
            }

            Submit();
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            Fuzzy.Rules.GenerateSugenoRule();
            Fuzzy.Membership.GenerateTriangular();
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
        }
    }
}

public class FuzzySugeno
{
    public List<SugenoRule> Rules = new();
    public List<FuzzyTriangleSet> Membership = new();

    public virtual List<ISumoAction> Run(List<float> inputs, int topActionsNum = 2)
    {
        var fuzzificationResult = Fuzzification(inputs);
        var defuzzificationResult = Defuzzification(fuzzificationResult);
        var topActions = defuzzificationResult
            .OrderByDescending(x => x.score)
            .Take(topActionsNum)
            .Select(x => x.action);
        return topActions.ToList();
    }

    public virtual Dictionary<string, Dictionary<string, float>> Fuzzification(
               List<float> inputs)
    {
        var result = new Dictionary<string, Dictionary<string, float>>();

        for (int i = 0; i < inputs.Count; i++)
        {
            result[Membership[i].Name] = Membership[i].EvaluateMembership(inputs[i]);
        }
        return result;
    }

    public List<(ISumoAction action, float score)> Defuzzification(Dictionary<string, Dictionary<string, float>> inputResult)
    {
        Dictionary<ISumoAction, (float sumWeightScore, float sumWeight)> actionMap = new();

        foreach (var rule in Rules)
        {
            float z = rule.GetOutputScore(inputResult, out float w);

            if (w > 0f)
            {
                if (!actionMap.TryGetValue(rule.Action, out var agg))
                    agg = (0f, 0f);

                agg.sumWeightScore += w * z;
                agg.sumWeight += w;

                actionMap[rule.Action] = agg;
            }
        }

        List<(ISumoAction action, float score)> scoredActions = new();
        foreach (var (action, (wz, w)) in actionMap)
        {
            if (w > 0f)
                scoredActions.Add((action, wz / w));
        }

        return scoredActions;
    }
}

public class FuzzyTriangleSet
{
    public string Name;
    public List<FuzzyTriangleValue> Sets;

    public FuzzyTriangleSet(string name, List<FuzzyTriangleValue> values)
    {
        Name = name;
        Sets = values;
    }

    public Dictionary<string, float> EvaluateMembership(float input)
    {
        var result = new Dictionary<string, float>();
        foreach (var cat in Sets)
        {
            result[cat.Label] = cat.EvaluateSet(input);
        }
        return result;
    }
}

public class FuzzyTriangleValue
{
    public string Label;
    public FuzzyTriangleValue(string label, float center, float spread)
    {
        Label = label;
        Center = center;
        Spread = spread;
    }
    public float Center;
    public float Spread;

    public float EvaluateSet(float input)
    {
        return Mathf.Clamp01(1f - Mathf.Abs(input - Center) / Spread);
    }
}

public class SugenoRule
{
    public Dictionary<string, string> Conditions = new();
    public ISumoAction Action;
    public float CrispOutput;

    public float GetOutputScore(
        Dictionary<string, Dictionary<string, float>> fuzzyInputs,
        out float strength)
    {
        strength = 1f;
        foreach (var (varName, label) in Conditions)
        {
            if (!fuzzyInputs.TryGetValue(varName, out var v) || !v.TryGetValue(label, out float membershipMiu))
            {
                strength = 0f;
                return 0f;
            }
            strength *= membershipMiu;
        }
        return CrispOutput;
    }
}

public static class FuzzySugenoExtension
{
    public static void GenerateSugenoRule(this List<SugenoRule> value)
    {
        value.AddRange(new List<SugenoRule>
            {
                new() {
                    Conditions = new()
                    {
                        { "distance_from_arena", "near_border" },
                        { "facing_to_arena", "front" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.Accelerate, 0.2f),
                    CrispOutput = 0.9f
                },
                new() {
                    Conditions = new()
                    {
                        { "distance_from_arena", "near_border" },
                        { "distance_from_enemy", "far" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.5f),
                    CrispOutput = 0.9f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "back_left" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 065f),
                    CrispOutput = 0.9f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "back_right" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.65f),
                    CrispOutput = 0.9f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "back_right" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.65f),
                    CrispOutput = 0.9f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "back" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.9f),
                    CrispOutput = 1.5f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "front_left" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.3f),
                    CrispOutput = 0.9f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "left" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.5f),
                    CrispOutput = 0.9f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "right" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.5f),
                    CrispOutput = 0.9f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "front_left" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.25f),
                    CrispOutput = 0.9f
                },

                new() {
                    Conditions = new()
                    {
                        { "enemy_position", "front_right" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.2f),
                    CrispOutput = 0.9f
                },
                #region Accelerate
                new() {
                    Conditions = new()
                    {
                        { "distance_from_enemy", "far" },
                        { "enemy_side", "front" },
                    },
                    Action = new AccelerateAction(InputType.Script, 0.2f),
                    CrispOutput = 3f
                },
                new() {
                    Conditions = new()
                    {
                        { "distance_from_enemy", "close" },
                        { "enemy_side", "front" },
                    },
                    Action = new DashAction(InputType.Script),
                    CrispOutput = 5f
                },
                new() {
                    Conditions = new()
                    {
                        { "distance_from_enemy", "medium" },
                        { "enemy_side", "front" },
                    },
                    Action = new AccelerateAction(InputType.Script, 0.3f),
                    CrispOutput = 2.5f
                },
                new() {
                    Conditions = new()
                    {
                        { "distance_from_enemy", "medium" },
                        { "enemy_side", "front" },
                    },
                    Action = new AccelerateAction(InputType.Script, 0.3f),
                    CrispOutput = 2.5f
                },
                #endregion Accelerate
                #region Skill
                new() {
                    Conditions = new()
                    {
                        { "distance_from_arena", "near_border" },
                        { "distance_from_enemy", "medium" },
                    },
                    Action = new SkillAction(InputType.Script, ActionType.SkillStone),
                    CrispOutput = 1f
                },
                new() {
                    Conditions = new()
                    {
                        { "enemy_side", "front" },
                        { "distance_from_enemy", "medium" },
                    },
                    Action = new SkillAction(InputType.Script, ActionType.SkillBoost),
                    CrispOutput = 1f
                },
                #endregion Skill
            });
    }

    public static void GenerateTriangular(this List<FuzzyTriangleSet> value)
    {
        value.AddRange(new List<FuzzyTriangleSet>
            {
                new("distance_from_enemy", new()
                    {
                        new FuzzyTriangleValue ( label : "far", center : 0f, spread : 0.3f ),
                        new FuzzyTriangleValue ( label : "medium", center : 0.6f, spread : 0.25f ),
                        new FuzzyTriangleValue ( label : "close", center : 1f, spread : 0.6f ),
                    }),
                new("enemy_position", new()
                    {
                        new FuzzyTriangleValue(label: "front", center: 0, spread: 0.25f),
                        new FuzzyTriangleValue(label: "right", center: -0.5f, spread: 0.25f),
                        new FuzzyTriangleValue(label: "left", center: 0.5f, spread: 0.25f),
                        new FuzzyTriangleValue(label: "back", center: 1, spread: 0.25f),
                    
                        new FuzzyTriangleValue(label: "front_left", center: 0.25f, spread: 0.25f),
                        new FuzzyTriangleValue(label: "front_right", center: -0.25f, spread: 0.25f),

                        new FuzzyTriangleValue(label: "back_left", center: 0.75f, spread: 0.25f),
                        new FuzzyTriangleValue(label: "back_right", center: -0.75f, spread: 0.25f),
                    }),
                new("enemy_side", new()
                    {
                        new FuzzyTriangleValue(label: "behind", center: 0f, spread: 0.4f),
                        new FuzzyTriangleValue(label: "side", center: 0.7f, spread: 0.3f),
                        new FuzzyTriangleValue(label: "front", center: 1f, spread: 0.7f),
                    }),
                new("distance_from_arena", new()
                    {
                        new FuzzyTriangleValue(label: "near_border", center: 0f, spread: 0.3f),
                        new FuzzyTriangleValue(label: "close_border", center: 0.6f, spread: 0.2f),
                        new FuzzyTriangleValue(label: "center", center: 1f, spread: 0.8f),
                    }),
                new("facing_to_arena", new()
                    {
                        new FuzzyTriangleValue("front",  center: -1f, spread: 1f),
                        new FuzzyTriangleValue("behind", center:  1f, spread: 1f),
                    }),
            });
    }
}