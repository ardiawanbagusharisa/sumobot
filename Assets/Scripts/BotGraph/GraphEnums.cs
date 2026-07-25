namespace SumoBot.Graph
{
    /// <summary>The value type carried on a port or a parameter. Number is a float; Bool is a
    /// true/false. Connections are only valid between ports of the same PortType.</summary>
    public enum PortType
    {
        Number,
        Bool,
    }

    /// <summary>Whether a port produces a value (Out) or consumes one (In).</summary>
    public enum PortDirection
    {
        In,
        Out,
    }

    /// <summary>Palette grouping / role of a module: a Sensor reads the world, Logic combines
    /// values, an Action drives the bot when its trigger is true.</summary>
    public enum ModuleKind
    {
        Sensor,
        Logic,
        Action,
    }
}
