using System.Linq;
using NUnit.Framework;

namespace SumoBot.Graph.Tests
{
    // E1: validation is what makes the graph a SAFE representation the interpreter (E2) can trust.
    // A valid graph passes; each failure mode is detected.
    public class BotGraphValidationTests
    {
        private static ModuleLibrary Lib => ModuleLibrary.BuildDefault();

        // enemyDistance > constant -> accelerate: minimal valid, typed, acyclic graph.
        private static BotGraph ValidGraph()
        {
            var g = new BotGraph { Name = "Chaser" };
            g.Nodes.Add(new GraphNode { NodeId = "dist", TypeId = "sensor.enemyDistance" });
            var k = new GraphNode { NodeId = "k", TypeId = "logic.constant" };
            k.Params["value"] = 0.5f;
            g.Nodes.Add(k);
            g.Nodes.Add(new GraphNode { NodeId = "cmp", TypeId = "logic.greaterThan" });
            g.Nodes.Add(new GraphNode { NodeId = "acc", TypeId = "action.accelerate" });
            g.Connections.Add(new GraphConnection { FromNodeId = "dist", FromPortId = "value", ToNodeId = "cmp", ToPortId = "a" });
            g.Connections.Add(new GraphConnection { FromNodeId = "k", FromPortId = "value", ToNodeId = "cmp", ToPortId = "b" });
            g.Connections.Add(new GraphConnection { FromNodeId = "cmp", FromPortId = "value", ToNodeId = "acc", ToPortId = "when" });
            return g;
        }

        private static bool HasCode(BotGraph g, GraphErrorCode code)
            => GraphValidator.Validate(g, Lib).Any(e => e.Code == code);

        [Test]
        public void ValidGraph_HasNoErrors()
        {
            var errors = GraphValidator.Validate(ValidGraph(), Lib);
            Assert.IsEmpty(errors, string.Join("; ", errors.Select(e => e.ToString())));
        }

        [Test]
        public void UnknownModuleType_IsDetected()
        {
            var g = ValidGraph();
            g.Nodes.Add(new GraphNode { NodeId = "x", TypeId = "sensor.doesNotExist" });
            Assert.IsTrue(HasCode(g, GraphErrorCode.UnknownModuleType));
        }

        [Test]
        public void DuplicateNodeId_IsDetected()
        {
            var g = ValidGraph();
            g.Nodes.Add(new GraphNode { NodeId = "acc", TypeId = "action.dash" });
            Assert.IsTrue(HasCode(g, GraphErrorCode.DuplicateNodeId));
        }

        [Test]
        public void PortTypeMismatch_IsDetected()
        {
            var g = ValidGraph();
            // enemyDistance (Number out) wired into a fresh Bool input.
            g.Nodes.Add(new GraphNode { NodeId = "dash", TypeId = "action.dash" });
            g.Connections.Add(new GraphConnection { FromNodeId = "dist", FromPortId = "value", ToNodeId = "dash", ToPortId = "when" });
            Assert.IsTrue(HasCode(g, GraphErrorCode.PortTypeMismatch));
        }

        [Test]
        public void UnknownPort_IsDetected()
        {
            var g = ValidGraph();
            g.Connections.Add(new GraphConnection { FromNodeId = "dist", FromPortId = "nope", ToNodeId = "cmp", ToPortId = "a" });
            Assert.IsTrue(HasCode(g, GraphErrorCode.ConnectionUnknownPort));
        }

        [Test]
        public void WrongDirectionConnection_IsDetected()
        {
            var g = ValidGraph();
            // Wire into cmp's OUTPUT port "value" (an input-side use of an output) -> direction mismatch.
            g.Nodes.Add(new GraphNode { NodeId = "dash", TypeId = "action.dash" });
            g.Connections.Add(new GraphConnection { FromNodeId = "dash", FromPortId = "when", ToNodeId = "cmp", ToPortId = "a" });
            Assert.IsTrue(HasCode(g, GraphErrorCode.PortDirectionMismatch));
        }

        [Test]
        public void ConnectionToUnknownNode_IsDetected()
        {
            var g = ValidGraph();
            g.Connections.Add(new GraphConnection { FromNodeId = "ghost", FromPortId = "value", ToNodeId = "acc", ToPortId = "when" });
            Assert.IsTrue(HasCode(g, GraphErrorCode.ConnectionUnknownNode));
        }

        [Test]
        public void RequiredInputUnconnected_IsDetected()
        {
            var g = ValidGraph();
            g.Connections.RemoveAll(c => c.ToNodeId == "acc" && c.ToPortId == "when");
            Assert.IsTrue(HasCode(g, GraphErrorCode.RequiredInputUnconnected));
        }

        [Test]
        public void InputFanInGreaterThanOne_IsDetected()
        {
            var g = ValidGraph();
            g.Nodes.Add(new GraphNode { NodeId = "ready", TypeId = "sensor.dashReady" });
            g.Connections.Add(new GraphConnection { FromNodeId = "ready", FromPortId = "value", ToNodeId = "acc", ToPortId = "when" });
            Assert.IsTrue(HasCode(g, GraphErrorCode.InputAlreadyConnected));
        }

        [Test]
        public void Cycle_IsDetected()
        {
            // n1.not -> n2.not -> n1.not: a two-node Bool cycle with both required inputs fed.
            var g = new BotGraph();
            g.Nodes.Add(new GraphNode { NodeId = "n1", TypeId = "logic.not" });
            g.Nodes.Add(new GraphNode { NodeId = "n2", TypeId = "logic.not" });
            g.Connections.Add(new GraphConnection { FromNodeId = "n1", FromPortId = "value", ToNodeId = "n2", ToPortId = "a" });
            g.Connections.Add(new GraphConnection { FromNodeId = "n2", FromPortId = "value", ToNodeId = "n1", ToPortId = "a" });
            Assert.IsTrue(HasCode(g, GraphErrorCode.Cycle));
        }
    }
}
