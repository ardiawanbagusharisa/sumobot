namespace SumoBot.Graph
{
    /// <summary>The kind of action an Action module wants to fire this tick. Mirrors the
    /// action.* module TypeIds; the runtime (GraphBot) maps each to a concrete ISumoAction.</summary>
    public enum GraphActionKind
    {
        Accelerate,
        TurnLeft,
        TurnRight,
        Dash,
        SkillBoost,
        SkillStone,
    }

    /// <summary>
    /// One action the interpreter decided to fire this tick, with its duration (seconds; ignored
    /// by Dash/Skill). Kept engine-free so the pure interpreter can produce it; the runtime turns
    /// it into an ISumoAction.
    /// </summary>
    public readonly struct GraphActionIntent
    {
        public readonly GraphActionKind Kind;
        public readonly float Duration;

        public GraphActionIntent(GraphActionKind kind, float duration)
        {
            Kind = kind;
            Duration = duration;
        }
    }
}
