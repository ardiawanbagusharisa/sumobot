// StrategyToActionMapper.cs
using CoreSumo;
using UnityEngine;

/// <summary>
/// Helper class untuk mapping string strategi hasil SLM (Python) ke ISumoAction Unity.
/// Sudah support turn_left_ANGLE, turn_right_ANGLE, dan aksi standar lainnya.
/// </summary>
public static class StrategyToActionMapper
{
    /// <summary>
    /// Mapping string strategi ke ISumoAction.
    /// Jika strategi "stay" atau tidak dikenal, return null (robot diam).
    /// Untuk strategi dengan angle (turn_left_90) akan dibuatkan action angle.
    /// </summary>
    public static ISumoAction Map(string strategy)
    {
        // Mapping untuk aksi turn angle, format: turn_left_45, turn_right_90, dst
        if (strategy.StartsWith("turn_left_"))
            return new TurnAction(InputType.Script, ActionType.TurnLeftWithAngle, ParseAngle(strategy));
        if (strategy.StartsWith("turn_right_"))
            return new TurnAction(InputType.Script, ActionType.TurnRightWithAngle, ParseAngle(strategy));

        switch (strategy)
        {
            case "accelerate":
                return new AccelerateAction(InputType.Script);

            case "avoid":
                // Contoh: mapping avoid ke turn left biasa
                return new TurnAction(InputType.Script, ActionType.TurnLeft);

            case "boost":
                return new SkillAction(InputType.Script);

            case "dash":
                return new DashAction(InputType.Script);

            case "stay":
                // Tidak melakukan aksi apapun
                return null;

            case "turn_left":
                return new TurnAction(InputType.Script, ActionType.TurnLeft);

            case "turn_right":
                return new TurnAction(InputType.Script, ActionType.TurnRight);

            default:
                Debug.LogWarning($"[StrategyToActionMapper] Unknown strategy: '{strategy}'. No action will be taken.");
                return null;
        }
    }

    /// <summary>
    /// Parse nilai angle dari string strategi, misal "turn_left_90" → 90f.
    /// </summary>
    private static float ParseAngle(string strategy)
    {
        // Pecah dengan underscore, ambil bagian terakhir
        var parts = strategy.Split('_');
        if (parts.Length >= 3 && float.TryParse(parts[2], out float angle))
            return angle;
        return 45f; // default fallback
    }
}
