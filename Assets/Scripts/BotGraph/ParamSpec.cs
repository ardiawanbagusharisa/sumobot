namespace SumoBot.Graph
{
    /// <summary>
    /// A configurable constant on a module — e.g. a Constant module's value, or an Accelerate
    /// module's duration. Unlike a port, a parameter is not wired; the player sets it, and its
    /// value is stored per-node in <see cref="GraphNode.Params"/> keyed by <see cref="Id"/>.
    ///
    /// Values are stored as float for a uniform, engine-free data shape. Bool parameters use
    /// 0 (false) / non-zero (true); Min/Max bound Number parameters for editor clamping.
    /// </summary>
    public class ParamSpec
    {
        public string Id;
        public string DisplayName;
        public PortType Type;
        public float Default;
        public float Min;
        public float Max;

        public ParamSpec() { }

        public ParamSpec(string id, string displayName, PortType type, float def, float min, float max)
        {
            Id = id;
            DisplayName = displayName;
            Type = type;
            Default = def;
            Min = min;
            Max = max;
        }
    }
}
