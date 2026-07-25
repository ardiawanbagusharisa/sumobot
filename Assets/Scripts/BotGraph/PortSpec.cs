namespace SumoBot.Graph
{
    /// <summary>
    /// Describes one port on a <see cref="ModuleDefinition"/>: its id (unique within the module),
    /// a display label, its value type, and its direction. Input ports may be marked Required —
    /// a required input with no incoming connection is a validation error (an Action whose
    /// trigger is never wired would never fire).
    ///
    /// This is library metadata, not saved graph data: ports come from the module definition, so
    /// a saved graph only stores connections between (node, port id) pairs.
    /// </summary>
    public class PortSpec
    {
        public string Id;
        public string DisplayName;
        public PortType Type;
        public PortDirection Direction;
        public bool Required;

        public PortSpec() { }

        public PortSpec(string id, string displayName, PortType type, PortDirection direction, bool required = false)
        {
            Id = id;
            DisplayName = displayName;
            Type = type;
            Direction = direction;
            Required = required;
        }
    }
}
