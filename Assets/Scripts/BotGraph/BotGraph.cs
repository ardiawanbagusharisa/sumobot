using System.Collections.Generic;

namespace SumoBot.Graph
{
    /// <summary>
    /// One node placed in a graph: an instance of a <see cref="ModuleDefinition"/> (by TypeId)
    /// with a unique NodeId, an editor position, and its parameter values keyed by ParamSpec.Id.
    /// Bool parameters use 0 (false) / non-zero (true).
    /// </summary>
    public class GraphNode
    {
        public string NodeId;
        public string TypeId;
        public float X;
        public float Y;
        public Dictionary<string, float> Params = new();
    }

    /// <summary>A directed wire from an output port to an input port.</summary>
    public class GraphConnection
    {
        public string FromNodeId;
        public string FromPortId;
        public string ToNodeId;
        public string ToPortId;
    }

    /// <summary>
    /// A player's bot creation as pure data: the saved module graph (decision-8). This is the
    /// payload a publish (decision-7) carries, what the editor (E3) saves/loads, and what the
    /// interpreter (E2) runs. <see cref="FormatVersion"/> is explicit from day one because this
    /// shape persists into player saves and Market listings — bump it on incompatible changes.
    /// </summary>
    public class BotGraph
    {
        public const int CurrentVersion = 1;

        public int FormatVersion = CurrentVersion;
        public string Name;
        public List<GraphNode> Nodes = new();
        public List<GraphConnection> Connections = new();
    }
}
