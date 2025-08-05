
using System.Collections.Generic;
using SumoCore;
using SumoInput;

namespace SumoBot.RuleBased.Fuzzy
{
    [System.Serializable]
    public class FuzzySugeno : FuzzyBase
    {
        public override List<(ISumoAction action, float score)> Defuzzification(Dictionary<string, Dictionary<string, float>> inputResult)
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


    [System.Serializable]
    public class SugenoRule : FuzzyRuleBase
    {
        public float CrispOutput;

        public override float GetOutputScore(
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
        public static void GenerateSugenoRule(this List<FuzzyRuleBase> value)
        {
            value.AddRange(new List<FuzzyRuleBase>
            {
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "distance_from_arena", "near_border" },
                        { "facing_to_arena", "front" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.Accelerate, 0.2f),
                    CrispOutput = 0.9f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "distance_from_arena", "near_border" },
                        { "distance_from_enemy", "far" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.5f),
                    CrispOutput = 0.9f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "back_left" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 065f),
                    CrispOutput = 0.9f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "back_right" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.65f),
                    CrispOutput = 0.9f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "back_right" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.65f),
                    CrispOutput = 0.9f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "back" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.9f),
                    CrispOutput = 1.5f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "front_left" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.3f),
                    CrispOutput = 0.9f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "left" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.5f),
                    CrispOutput = 0.9f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "right" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.5f),
                    CrispOutput = 0.9f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "front_left" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnLeft, 0.25f),
                    CrispOutput = 0.9f
                },

                new SugenoRule() {
                    Conditions = new()
                    {
                        { "enemy_position", "front_right" },
                    },
                    Action = new TurnAction(InputType.Script, ActionType.TurnRight, 0.2f),
                    CrispOutput = 0.9f
                },
                #region Accelerate
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "distance_from_enemy", "far" },
                        { "angle_to_enemy", "front" },
                    },
                    Action = new AccelerateAction(InputType.Script, 0.2f),
                    CrispOutput = 3f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "distance_from_enemy", "close" },
                        { "angle_to_enemy", "front" },
                    },
                    Action = new DashAction(InputType.Script),
                    CrispOutput = 5f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "distance_from_enemy", "medium" },
                        { "angle_to_enemy", "front" },
                    },
                    Action = new AccelerateAction(InputType.Script, 0.3f),
                    CrispOutput = 2.5f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "distance_from_enemy", "medium" },
                        { "angle_to_enemy", "front" },
                    },
                    Action = new AccelerateAction(InputType.Script, 0.3f),
                    CrispOutput = 2.5f
                },
                #endregion Accelerate
                #region Skill
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "distance_from_arena", "near_border" },
                        { "distance_from_enemy", "medium" },
                    },
                    Action = new SkillAction(InputType.Script, ActionType.SkillStone),
                    CrispOutput = 1f
                },
                new SugenoRule() {
                    Conditions = new()
                    {
                        { "angle_to_enemy", "front" },
                        { "distance_from_enemy", "medium" },
                    },
                    Action = new SkillAction(InputType.Script, ActionType.SkillBoost),
                    CrispOutput = 1f
                },
                #endregion Skill
            });

        }
    }
}