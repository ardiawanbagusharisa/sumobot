using NUnit.Framework;

namespace SumoBot.Graph.Tests
{
    // E1: a saved bot creation is the BotGraph JSON. These cover that it round-trips losslessly
    // and always carries an explicit format version (the shape persists into player saves and
    // Market listings, decision-8).
    public class BotGraphSerializationTests
    {
        // "chase when far": enemyDistance > constant -> accelerate.
        private static BotGraph SampleGraph()
        {
            var graph = new BotGraph { Name = "Chaser" };
            graph.Nodes.Add(new GraphNode { NodeId = "dist", TypeId = "sensor.enemyDistance", X = 10, Y = 20 });

            var k = new GraphNode { NodeId = "k", TypeId = "logic.constant", X = 30, Y = 40 };
            k.Params["value"] = 0.5f;
            graph.Nodes.Add(k);

            graph.Nodes.Add(new GraphNode { NodeId = "cmp", TypeId = "logic.greaterThan", X = 50, Y = 60 });

            var acc = new GraphNode { NodeId = "acc", TypeId = "action.accelerate", X = 70, Y = 80 };
            acc.Params["duration"] = 0.3f;
            graph.Nodes.Add(acc);

            graph.Connections.Add(new GraphConnection { FromNodeId = "dist", FromPortId = "value", ToNodeId = "cmp", ToPortId = "a" });
            graph.Connections.Add(new GraphConnection { FromNodeId = "k", FromPortId = "value", ToNodeId = "cmp", ToPortId = "b" });
            graph.Connections.Add(new GraphConnection { FromNodeId = "cmp", FromPortId = "value", ToNodeId = "acc", ToPortId = "when" });
            return graph;
        }

        [Test]
        public void ToJson_FromJson_RoundTripsNodesConnectionsAndParams()
        {
            var original = SampleGraph();

            var back = GraphSerializer.FromJson(GraphSerializer.ToJson(original));

            Assert.AreEqual(original.FormatVersion, back.FormatVersion);
            Assert.AreEqual("Chaser", back.Name);
            Assert.AreEqual(4, back.Nodes.Count);
            Assert.AreEqual(3, back.Connections.Count);
            Assert.AreEqual("sensor.enemyDistance", back.Nodes.Find(n => n.NodeId == "dist").TypeId);
            Assert.AreEqual(0.5f, back.Nodes.Find(n => n.NodeId == "k").Params["value"]);
            Assert.AreEqual(0.3f, back.Nodes.Find(n => n.NodeId == "acc").Params["duration"]);
            Assert.AreEqual(70f, back.Nodes.Find(n => n.NodeId == "acc").X);

            var wire = back.Connections.Find(c => c.ToNodeId == "acc");
            Assert.AreEqual("cmp", wire.FromNodeId);
            Assert.AreEqual("when", wire.ToPortId);
        }

        [Test]
        public void SerializedGraph_CarriesFormatVersion()
        {
            StringAssert.Contains("FormatVersion", GraphSerializer.ToJson(new BotGraph()));
        }

        [Test]
        public void NewGraph_UsesCurrentFormatVersion()
        {
            Assert.AreEqual(BotGraph.CurrentVersion, new BotGraph().FormatVersion);
        }

        [Test]
        public void RoundTrippedSampleGraph_StaysValid()
        {
            var back = GraphSerializer.FromJson(GraphSerializer.ToJson(SampleGraph()));
            Assert.IsTrue(GraphValidator.IsValid(back, ModuleLibrary.BuildDefault()));
        }
    }
}
