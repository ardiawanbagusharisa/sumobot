using System.Collections.Generic;

namespace SumoBot.Graph
{
    /// <summary>
    /// Evaluates a bot graph against a sensor snapshot and returns the action intents to fire this
    /// tick (decision-8 E2). Pure and deterministic: no Unity, no side effects — output depends
    /// only on the graph plus the sensor values, and Action nodes are visited in graph order.
    ///
    /// All port values are carried as float (Bool = 0/1, threshold 0.5), which is safe because
    /// GraphValidator guarantees Number/Bool ports are only wired to matching ports. The
    /// interpreter still guards against cycles and unknown nodes so an unvalidated/bad graph fails
    /// safe (produces no intents) rather than throwing mid-battle.
    /// </summary>
    public class GraphInterpreter
    {
        private const float BoolThreshold = 0.5f;

        private readonly BotGraph graph;
        private readonly ModuleLibrary library;
        private readonly Dictionary<string, GraphNode> nodesById = new();
        private readonly Dictionary<string, (string nodeId, string portId)> inputSource = new();

        private Dictionary<string, float> memo;
        private HashSet<string> visiting;
        private IBotSensors sensors;

        public GraphInterpreter(BotGraph graph, ModuleLibrary library)
        {
            this.graph = graph;
            this.library = library;

            foreach (var node in graph.Nodes)
                if (node != null && !string.IsNullOrEmpty(node.NodeId))
                    nodesById[node.NodeId] = node;

            // Each input port has at most one source (validation enforces fan-in <= 1).
            foreach (var conn in graph.Connections)
                if (conn != null)
                    inputSource[Key(conn.ToNodeId, conn.ToPortId)] = (conn.FromNodeId, conn.FromPortId);
        }

        /// <summary>Compute the actions to fire for the given sensor snapshot.</summary>
        public IReadOnlyList<GraphActionIntent> Evaluate(IBotSensors sensors)
        {
            this.sensors = sensors;
            memo = new Dictionary<string, float>();
            visiting = new HashSet<string>();

            var intents = new List<GraphActionIntent>();
            foreach (var node in graph.Nodes) // graph order -> deterministic
            {
                if (node == null) continue;
                var def = library.Get(node.TypeId);
                if (def == null || def.Kind != ModuleKind.Action) continue;

                if (IsTrue(ResolveInput(node.NodeId, "when")) && TryMapAction(node, def, out var intent))
                    intents.Add(intent);
            }
            return intents;
        }

        private bool TryMapAction(GraphNode node, ModuleDefinition def, out GraphActionIntent intent)
        {
            float duration = Param(node, def, "duration");
            switch (node.TypeId)
            {
                case "action.accelerate": intent = new GraphActionIntent(GraphActionKind.Accelerate, duration); return true;
                case "action.turnLeft": intent = new GraphActionIntent(GraphActionKind.TurnLeft, duration); return true;
                case "action.turnRight": intent = new GraphActionIntent(GraphActionKind.TurnRight, duration); return true;
                case "action.dash": intent = new GraphActionIntent(GraphActionKind.Dash, duration); return true;
                case "action.skillBoost": intent = new GraphActionIntent(GraphActionKind.SkillBoost, duration); return true;
                case "action.skillStone": intent = new GraphActionIntent(GraphActionKind.SkillStone, duration); return true;
                default: intent = default; return false;
            }
        }

        private float ResolveInput(string nodeId, string portId)
        {
            if (inputSource.TryGetValue(Key(nodeId, portId), out var src))
                return EvalOutput(src.nodeId, src.portId);
            return 0f; // unconnected input -> false/0 (validation flags required ones)
        }

        private float EvalOutput(string nodeId, string portId)
        {
            string key = Key(nodeId, portId);
            if (memo.TryGetValue(key, out var cached)) return cached;
            if (!visiting.Add(nodeId)) return 0f; // cycle guard -> fail safe

            float result = Compute(nodeId);
            visiting.Remove(nodeId);
            memo[key] = result;
            return result;
        }

        // Every module here has a single output ("value"), so the port id is not needed to compute.
        private float Compute(string nodeId)
        {
            if (!nodesById.TryGetValue(nodeId, out var node)) return 0f;
            var def = library.Get(node.TypeId);
            if (def == null) return 0f;

            switch (node.TypeId)
            {
                case "sensor.enemyDistance": return sensors.EnemyDistance;
                case "sensor.enemyAngle": return sensors.EnemyAngle;
                case "sensor.edgeProximity": return sensors.EdgeProximity;
                case "sensor.dashReady": return Bool(sensors.DashReady);
                case "sensor.skillReady": return Bool(sensors.SkillReady);
                case "sensor.selfOutOfArena": return Bool(sensors.SelfOutOfArena);
                case "sensor.enemyOutOfArena": return Bool(sensors.EnemyOutOfArena);

                case "logic.constant": return Param(node, def, "value");
                case "logic.greaterThan": return Bool(ResolveInput(nodeId, "a") > ResolveInput(nodeId, "b"));
                case "logic.lessThan": return Bool(ResolveInput(nodeId, "a") < ResolveInput(nodeId, "b"));
                case "logic.and": return Bool(IsTrue(ResolveInput(nodeId, "a")) && IsTrue(ResolveInput(nodeId, "b")));
                case "logic.or": return Bool(IsTrue(ResolveInput(nodeId, "a")) || IsTrue(ResolveInput(nodeId, "b")));
                case "logic.not": return Bool(!IsTrue(ResolveInput(nodeId, "a")));

                default: return 0f;
            }
        }

        private static float Param(GraphNode node, ModuleDefinition def, string paramId)
        {
            if (node.Params != null && node.Params.TryGetValue(paramId, out var value)) return value;
            foreach (var p in def.Parameters)
                if (p.Id == paramId) return p.Default;
            return 0f;
        }

        private static bool IsTrue(float value) => value >= BoolThreshold;
        private static float Bool(bool b) => b ? 1f : 0f;
        private static string Key(string nodeId, string portId) => nodeId + "/" + portId;
    }
}
