using System.Linq;
using NUnit.Framework;

namespace SumoBot.Graph.Tests
{
    // E2: the pure interpreter turns a graph + sensor snapshot into action intents. These use a
    // fake sensor set so the logic is tested with no Unity/battle runtime.
    public class GraphInterpreterTests
    {
        private static ModuleLibrary Lib => ModuleLibrary.BuildDefault();

        private class FakeSensors : IBotSensors
        {
            public float EnemyDistance { get; set; }
            public float EnemyAngle { get; set; }
            public float EdgeProximity { get; set; }
            public bool DashReady { get; set; }
            public bool SkillReady { get; set; }
            public bool SelfOutOfArena { get; set; }
            public bool EnemyOutOfArena { get; set; }
        }

        // enemyDistance > constant(0.5) -> accelerate(duration 0.3)
        private static BotGraph ChaseWhenFar()
        {
            var g = new BotGraph { Name = "Chaser" };
            g.Nodes.Add(new GraphNode { NodeId = "dist", TypeId = "sensor.enemyDistance" });
            var k = new GraphNode { NodeId = "k", TypeId = "logic.constant" };
            k.Params["value"] = 0.5f;
            g.Nodes.Add(k);
            g.Nodes.Add(new GraphNode { NodeId = "cmp", TypeId = "logic.greaterThan" });
            var acc = new GraphNode { NodeId = "acc", TypeId = "action.accelerate" };
            acc.Params["duration"] = 0.3f;
            g.Nodes.Add(acc);
            g.Connections.Add(new GraphConnection { FromNodeId = "dist", FromPortId = "value", ToNodeId = "cmp", ToPortId = "a" });
            g.Connections.Add(new GraphConnection { FromNodeId = "k", FromPortId = "value", ToNodeId = "cmp", ToPortId = "b" });
            g.Connections.Add(new GraphConnection { FromNodeId = "cmp", FromPortId = "value", ToNodeId = "acc", ToPortId = "when" });
            return g;
        }

        [Test]
        public void Accelerate_FiresWhenDistanceAboveThreshold()
        {
            var interp = new GraphInterpreter(ChaseWhenFar(), Lib);

            var intents = interp.Evaluate(new FakeSensors { EnemyDistance = 0.8f });

            Assert.AreEqual(1, intents.Count);
            Assert.AreEqual(GraphActionKind.Accelerate, intents[0].Kind);
            Assert.AreEqual(0.3f, intents[0].Duration);
        }

        [Test]
        public void Accelerate_DoesNotFireWhenDistanceBelowThreshold()
        {
            var interp = new GraphInterpreter(ChaseWhenFar(), Lib);

            var intents = interp.Evaluate(new FakeSensors { EnemyDistance = 0.2f });

            Assert.IsEmpty(intents);
        }

        [Test]
        public void Evaluate_IsDeterministic_AcrossRepeatedCalls()
        {
            var interp = new GraphInterpreter(ChaseWhenFar(), Lib);
            var sensors = new FakeSensors { EnemyDistance = 0.8f };

            var first = interp.Evaluate(sensors).Select(i => i.Kind).ToArray();
            var second = interp.Evaluate(sensors).Select(i => i.Kind).ToArray();

            CollectionAssert.AreEqual(first, second);
        }

        [Test]
        public void DashReadySensor_GatesDashAction()
        {
            // sensor.dashReady -> action.dash
            var g = new BotGraph();
            g.Nodes.Add(new GraphNode { NodeId = "ready", TypeId = "sensor.dashReady" });
            g.Nodes.Add(new GraphNode { NodeId = "dash", TypeId = "action.dash" });
            g.Connections.Add(new GraphConnection { FromNodeId = "ready", FromPortId = "value", ToNodeId = "dash", ToPortId = "when" });
            var interp = new GraphInterpreter(g, Lib);

            Assert.AreEqual(1, interp.Evaluate(new FakeSensors { DashReady = true }).Count);
            Assert.IsEmpty(interp.Evaluate(new FakeSensors { DashReady = false }));
        }

        [Test]
        public void NotGate_InvertsSensor()
        {
            // not(selfOutOfArena) -> accelerate: drive while I am safely inside.
            var g = new BotGraph();
            g.Nodes.Add(new GraphNode { NodeId = "out", TypeId = "sensor.selfOutOfArena" });
            g.Nodes.Add(new GraphNode { NodeId = "not", TypeId = "logic.not" });
            g.Nodes.Add(new GraphNode { NodeId = "acc", TypeId = "action.accelerate" });
            g.Connections.Add(new GraphConnection { FromNodeId = "out", FromPortId = "value", ToNodeId = "not", ToPortId = "a" });
            g.Connections.Add(new GraphConnection { FromNodeId = "not", FromPortId = "value", ToNodeId = "acc", ToPortId = "when" });
            var interp = new GraphInterpreter(g, Lib);

            Assert.AreEqual(1, interp.Evaluate(new FakeSensors { SelfOutOfArena = false }).Count);
            Assert.IsEmpty(interp.Evaluate(new FakeSensors { SelfOutOfArena = true }));
        }

        [Test]
        public void AndGate_RequiresBothInputs()
        {
            // dashReady AND enemyClose(distance < 0.3) -> dash
            var g = new BotGraph();
            g.Nodes.Add(new GraphNode { NodeId = "ready", TypeId = "sensor.dashReady" });
            g.Nodes.Add(new GraphNode { NodeId = "dist", TypeId = "sensor.enemyDistance" });
            var k = new GraphNode { NodeId = "k", TypeId = "logic.constant" };
            k.Params["value"] = 0.3f;
            g.Nodes.Add(k);
            g.Nodes.Add(new GraphNode { NodeId = "close", TypeId = "logic.lessThan" });
            g.Nodes.Add(new GraphNode { NodeId = "and", TypeId = "logic.and" });
            g.Nodes.Add(new GraphNode { NodeId = "dash", TypeId = "action.dash" });
            g.Connections.Add(new GraphConnection { FromNodeId = "dist", FromPortId = "value", ToNodeId = "close", ToPortId = "a" });
            g.Connections.Add(new GraphConnection { FromNodeId = "k", FromPortId = "value", ToNodeId = "close", ToPortId = "b" });
            g.Connections.Add(new GraphConnection { FromNodeId = "ready", FromPortId = "value", ToNodeId = "and", ToPortId = "a" });
            g.Connections.Add(new GraphConnection { FromNodeId = "close", FromPortId = "value", ToNodeId = "and", ToPortId = "b" });
            g.Connections.Add(new GraphConnection { FromNodeId = "and", FromPortId = "value", ToNodeId = "dash", ToPortId = "when" });
            var interp = new GraphInterpreter(g, Lib);

            Assert.AreEqual(1, interp.Evaluate(new FakeSensors { DashReady = true, EnemyDistance = 0.1f }).Count); // both true
            Assert.IsEmpty(interp.Evaluate(new FakeSensors { DashReady = true, EnemyDistance = 0.9f }));           // far
            Assert.IsEmpty(interp.Evaluate(new FakeSensors { DashReady = false, EnemyDistance = 0.1f }));          // on cooldown
        }

        [Test]
        public void UnconnectedTrigger_ProducesNoIntent()
        {
            var g = new BotGraph();
            g.Nodes.Add(new GraphNode { NodeId = "acc", TypeId = "action.accelerate" }); // "when" not wired
            var interp = new GraphInterpreter(g, Lib);

            Assert.IsEmpty(interp.Evaluate(new FakeSensors()));
        }

        [Test]
        public void UnconnectedComparatorInputs_TreatedAsZero()
        {
            // greaterThan with no inputs: 0 > 0 is false -> no accelerate.
            var g = new BotGraph();
            g.Nodes.Add(new GraphNode { NodeId = "cmp", TypeId = "logic.greaterThan" });
            g.Nodes.Add(new GraphNode { NodeId = "acc", TypeId = "action.accelerate" });
            g.Connections.Add(new GraphConnection { FromNodeId = "cmp", FromPortId = "value", ToNodeId = "acc", ToPortId = "when" });
            var interp = new GraphInterpreter(g, Lib);

            Assert.IsEmpty(interp.Evaluate(new FakeSensors()));
        }
    }
}
