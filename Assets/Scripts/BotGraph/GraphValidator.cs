using System.Collections.Generic;

namespace SumoBot.Graph
{
    public enum GraphErrorCode
    {
        UnknownModuleType,
        DuplicateNodeId,
        ConnectionUnknownNode,
        ConnectionUnknownPort,
        PortDirectionMismatch,
        PortTypeMismatch,
        InputAlreadyConnected,
        RequiredInputUnconnected,
        Cycle,
    }

    /// <summary>One problem found in a graph. Code lets callers branch (e.g. the editor can
    /// highlight the offending node); Message is human-readable.</summary>
    public class GraphError
    {
        public GraphErrorCode Code;
        public string Message;

        public GraphError(GraphErrorCode code, string message)
        {
            Code = code;
            Message = message;
        }

        public override string ToString() => $"{Code}: {Message}";
    }

    /// <summary>
    /// Validates a <see cref="BotGraph"/> against a <see cref="ModuleLibrary"/>, returning every
    /// problem found (empty list = valid). The interpreter (E2) refuses to run an invalid graph;
    /// the editor (E3) surfaces these to the player. Data must flow acyclically from sensors
    /// through logic into actions, so cycles and type/direction mismatches are rejected.
    /// </summary>
    public static class GraphValidator
    {
        public static bool IsValid(BotGraph graph, ModuleLibrary library) => Validate(graph, library).Count == 0;

        public static IReadOnlyList<GraphError> Validate(BotGraph graph, ModuleLibrary library)
        {
            var errors = new List<GraphError>();
            if (graph == null || library == null) return errors;

            // Index nodes; catch missing/duplicate ids and unknown module types.
            var nodesById = new Dictionary<string, GraphNode>();
            foreach (var node in graph.Nodes)
            {
                if (node == null) continue;
                if (string.IsNullOrEmpty(node.NodeId) || nodesById.ContainsKey(node.NodeId))
                {
                    errors.Add(new GraphError(GraphErrorCode.DuplicateNodeId,
                        $"Duplicate or missing node id '{node.NodeId}'."));
                    continue;
                }
                nodesById[node.NodeId] = node;
                if (!library.Contains(node.TypeId))
                    errors.Add(new GraphError(GraphErrorCode.UnknownModuleType,
                        $"Node '{node.NodeId}' uses unknown module '{node.TypeId}'."));
            }

            // Validate connections; count fan-in per input port (must be at most 1).
            var inboundCount = new Dictionary<string, int>();
            foreach (var conn in graph.Connections)
            {
                if (conn == null) continue;

                if (!nodesById.TryGetValue(conn.FromNodeId, out var fromNode) ||
                    !nodesById.TryGetValue(conn.ToNodeId, out var toNode))
                {
                    errors.Add(new GraphError(GraphErrorCode.ConnectionUnknownNode,
                        $"Connection references unknown node(s): '{conn.FromNodeId}' -> '{conn.ToNodeId}'."));
                    continue;
                }

                var fromDef = library.Get(fromNode.TypeId);
                var toDef = library.Get(toNode.TypeId);
                if (fromDef == null || toDef == null) continue; // unknown type already reported

                var fromPort = fromDef.FindPort(conn.FromPortId, PortDirection.Out);
                var toPort = toDef.FindPort(conn.ToPortId, PortDirection.In);

                // Distinguish a wrong-direction wire (port exists, wrong side) from a missing port.
                if (fromPort == null || toPort == null)
                {
                    bool directionMismatch =
                        (fromPort == null && fromDef.FindPort(conn.FromPortId, PortDirection.In) != null) ||
                        (toPort == null && toDef.FindPort(conn.ToPortId, PortDirection.Out) != null);
                    errors.Add(directionMismatch
                        ? new GraphError(GraphErrorCode.PortDirectionMismatch,
                            $"Connection must go Out -> In: '{conn.FromNodeId}.{conn.FromPortId}' -> '{conn.ToNodeId}.{conn.ToPortId}'.")
                        : new GraphError(GraphErrorCode.ConnectionUnknownPort,
                            $"Connection uses unknown port: '{conn.FromNodeId}.{conn.FromPortId}' -> '{conn.ToNodeId}.{conn.ToPortId}'."));
                    continue;
                }

                if (fromPort.Type != toPort.Type)
                    errors.Add(new GraphError(GraphErrorCode.PortTypeMismatch,
                        $"Type mismatch: '{conn.FromNodeId}.{conn.FromPortId}' ({fromPort.Type}) -> " +
                        $"'{conn.ToNodeId}.{conn.ToPortId}' ({toPort.Type})."));

                string inKey = conn.ToNodeId + "/" + conn.ToPortId;
                inboundCount.TryGetValue(inKey, out int count);
                inboundCount[inKey] = count + 1;
                if (count + 1 > 1)
                    errors.Add(new GraphError(GraphErrorCode.InputAlreadyConnected,
                        $"Input '{conn.ToNodeId}.{conn.ToPortId}' has more than one incoming connection."));
            }

            // Required inputs must be connected (e.g. an Action whose trigger is never wired).
            foreach (var node in nodesById.Values)
            {
                var def = library.Get(node.TypeId);
                if (def == null) continue;
                foreach (var port in def.Inputs)
                {
                    if (!port.Required) continue;
                    if (!inboundCount.ContainsKey(node.NodeId + "/" + port.Id))
                        errors.Add(new GraphError(GraphErrorCode.RequiredInputUnconnected,
                            $"Required input '{node.NodeId}.{port.Id}' is not connected."));
                }
            }

            if (HasCycle(nodesById, graph.Connections))
                errors.Add(new GraphError(GraphErrorCode.Cycle,
                    "The graph contains a cycle; data must flow acyclically from sensors to actions."));

            return errors;
        }

        private static bool HasCycle(Dictionary<string, GraphNode> nodesById, List<GraphConnection> connections)
        {
            var adjacency = new Dictionary<string, List<string>>();
            foreach (var conn in connections)
            {
                if (conn == null) continue;
                if (!nodesById.ContainsKey(conn.FromNodeId) || !nodesById.ContainsKey(conn.ToNodeId)) continue;
                if (!adjacency.TryGetValue(conn.FromNodeId, out var list))
                {
                    list = new List<string>();
                    adjacency[conn.FromNodeId] = list;
                }
                list.Add(conn.ToNodeId);
            }

            // 0 = unvisited, 1 = on the current DFS stack, 2 = fully explored.
            var state = new Dictionary<string, int>();
            foreach (var id in nodesById.Keys)
                if (Visit(id, adjacency, state))
                    return true;
            return false;
        }

        private static bool Visit(string node, Dictionary<string, List<string>> adjacency, Dictionary<string, int> state)
        {
            state.TryGetValue(node, out int s);
            if (s == 1) return true;  // back-edge onto the stack -> cycle
            if (s == 2) return false;

            state[node] = 1;
            if (adjacency.TryGetValue(node, out var next))
                foreach (var n in next)
                    if (Visit(n, adjacency, state))
                        return true;
            state[node] = 2;
            return false;
        }
    }
}
