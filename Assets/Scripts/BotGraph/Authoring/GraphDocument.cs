using System;
using System.Collections.Generic;

namespace SumoBot.Graph.Authoring
{
    /// <summary>
    /// The live editing surface for one bot graph (E3.1 data bridge, decision-10). It wraps a
    /// <see cref="BotGraph"/> as the single source of truth: every editor gesture — placing or
    /// removing a node, wiring or cutting a connection, moving a node, changing a parameter — goes
    /// through here and mutates that one model. The MonoBehaviour visuals (E3.2) are a view keyed
    /// by <see cref="GraphNode.NodeId"/>; resolve a visual back to its backing node with
    /// <see cref="TryGetNode"/>.
    ///
    /// Because the model stays authoritative, the board can be validated continuously
    /// (<see cref="Validate"/>) and saved with a one-line <see cref="ToJson"/>. This type is pure
    /// data — no Unity, no scene — so it is unit-testable on its own.
    /// </summary>
    public class GraphDocument
    {
        private readonly ModuleLibrary library;
        private readonly Dictionary<string, GraphNode> nodesById = new();
        private int nextNodeSeq;

        /// <summary>The authoritative graph. Prefer the mutation methods over editing it directly,
        /// so the NodeId index stays in sync.</summary>
        public BotGraph Graph { get; }

        /// <summary>Raised after a structural change (node or connection added/removed) so the editor
        /// can re-validate. Not raised for cosmetic moves or parameter edits, which cannot change
        /// validity.</summary>
        public event Action Changed;

        public GraphDocument(ModuleLibrary library, BotGraph graph = null)
        {
            this.library = library ?? throw new ArgumentNullException(nameof(library));
            Graph = graph ?? new BotGraph();
            ReindexFromGraph();
        }

        /// <summary>Load an existing draft's graph into a fresh document.</summary>
        public static GraphDocument FromJson(string json, ModuleLibrary library)
            => new(library, GraphSerializer.FromJson(json));

        public string Name
        {
            get => Graph.Name;
            set => Graph.Name = value;
        }

        public IReadOnlyList<GraphNode> Nodes => Graph.Nodes;
        public IReadOnlyList<GraphConnection> Connections => Graph.Connections;

        /// <summary>Resolve a visual node (by NodeId) back to its backing <see cref="GraphNode"/>.</summary>
        public bool TryGetNode(string nodeId, out GraphNode node) => nodesById.TryGetValue(nodeId ?? "", out node);

        /// <summary>
        /// Place a node of the given module type at (x, y): assigns a unique NodeId, seeds every
        /// parameter to its definition default, and adds it to the graph. Returns the new node, or
        /// null if the type is not in the library (the board only builds from the safe vocabulary).
        /// </summary>
        public GraphNode AddNode(string typeId, float x, float y)
        {
            var def = library.Get(typeId);
            if (def == null) return null;

            var node = new GraphNode { NodeId = NewNodeId(), TypeId = typeId, X = x, Y = y };
            foreach (var param in def.Parameters)
                node.Params[param.Id] = param.Default;

            Graph.Nodes.Add(node);
            nodesById[node.NodeId] = node;
            Changed?.Invoke();
            return node;
        }

        /// <summary>Remove a node and every connection touching it. Returns false if unknown.</summary>
        public bool RemoveNode(string nodeId)
        {
            if (nodeId == null || !nodesById.TryGetValue(nodeId, out var node)) return false;

            nodesById.Remove(nodeId);
            Graph.Nodes.Remove(node);
            Graph.Connections.RemoveAll(c => c != null && (c.FromNodeId == nodeId || c.ToNodeId == nodeId));
            Changed?.Invoke();
            return true;
        }

        /// <summary>Update a node's editor position. Returns false if the node is unknown.</summary>
        public bool MoveNode(string nodeId, float x, float y)
        {
            if (!TryGetNode(nodeId, out var node)) return false;
            node.X = x;
            node.Y = y;
            return true;
        }

        /// <summary>Set a node parameter value. Returns false if the node is unknown.</summary>
        public bool SetParam(string nodeId, string paramId, float value)
        {
            if (paramId == null || !TryGetNode(nodeId, out var node)) return false;
            node.Params[paramId] = value;
            return true;
        }

        /// <summary>
        /// Wire an output port to an input port. Both endpoints must be existing nodes; port
        /// legality (direction, type, fan-in, cycles) is not enforced here — the board stays
        /// permissive and <see cref="Validate"/> surfaces any problem to the editor. Returns the
        /// new connection, or null if either node is unknown.
        /// </summary>
        public GraphConnection AddConnection(string fromNodeId, string fromPortId, string toNodeId, string toPortId)
        {
            if (!nodesById.ContainsKey(fromNodeId ?? "") || !nodesById.ContainsKey(toNodeId ?? "")) return null;

            var conn = new GraphConnection
            {
                FromNodeId = fromNodeId,
                FromPortId = fromPortId,
                ToNodeId = toNodeId,
                ToPortId = toPortId,
            };
            Graph.Connections.Add(conn);
            Changed?.Invoke();
            return conn;
        }

        /// <summary>Cut a specific wire. Returns false if no matching connection exists.</summary>
        public bool RemoveConnection(string fromNodeId, string fromPortId, string toNodeId, string toPortId)
        {
            int removed = Graph.Connections.RemoveAll(c => c != null &&
                c.FromNodeId == fromNodeId && c.FromPortId == fromPortId &&
                c.ToNodeId == toNodeId && c.ToPortId == toPortId);
            if (removed > 0) Changed?.Invoke();
            return removed > 0;
        }

        /// <summary>Every problem in the current graph (empty = valid). Cheap enough to call
        /// after each edit so the editor can highlight errors live.</summary>
        public IReadOnlyList<GraphError> Validate() => GraphValidator.Validate(Graph, library);

        public bool IsValid => GraphValidator.IsValid(Graph, library);

        /// <summary>Serialize the current graph to the on-disk / publish JSON form.</summary>
        public string ToJson() => GraphSerializer.ToJson(Graph);

        private void ReindexFromGraph()
        {
            nodesById.Clear();
            foreach (var node in Graph.Nodes)
                if (node != null && !string.IsNullOrEmpty(node.NodeId))
                    nodesById[node.NodeId] = node;
        }

        // Unique within this graph regardless of ids a loaded draft already uses.
        private string NewNodeId()
        {
            string id;
            do { id = "node-" + nextNodeSeq++; } while (nodesById.ContainsKey(id));
            return id;
        }
    }
}
