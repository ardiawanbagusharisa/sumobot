// StrategyToActionMapper.cs
using SumoCore;
using SumoInput;
using UnityEngine;

public static class StrategyToActionMapper
{

    public static ISumoAction Map(string strategy)
    {
        if (string.IsNullOrEmpty(strategy))
            return null;

        if (strategy.StartsWith("turn_left_"))
            return new TurnAction(InputType.Script, ActionType.TurnLeftWithAngle, ParseAngle(strategy));
        if (strategy.StartsWith("turn_right_"))
            return new TurnAction(InputType.Script, ActionType.TurnRightWithAngle, ParseAngle(strategy));

        switch (strategy)
        {
            case "accelerate":
                return new AccelerateAction(InputType.Script);

            case "dash":
                return new DashAction(InputType.Script);

            case "boost":
                return new SkillAction(InputType.Script);

            case "turn_left":
                return new TurnAction(InputType.Script, ActionType.TurnLeft);

            case "turn_right":
                return new TurnAction(InputType.Script, ActionType.TurnRight);

            case "stay":
                return null;

            default:
                Debug.LogWarning($"[StrategyToActionMapper] Unknown strategy: '{strategy}'. No action will be taken.");
                return null;
        }
    }

    private static float ParseAngle(string strategy)
    {
        var parts = strategy.Split('_');
        if (parts.Length >= 3 && float.TryParse(parts[2], out float angle))
            return angle;
        return 45f;
    }
}
