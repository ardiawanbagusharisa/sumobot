using NUnit.Framework;
using SumoBot.Graph.Authoring;

namespace SumoBot.Graph.Tests
{
    // E3.1: the board holds a live BotGraph as the single source of truth. These cover that each
    // editor gesture mutates that model, that a visual node resolves to its GraphNode by NodeId,
    // and that the model validates + serializes straight off the wrapped graph (decision-10).
    public class GraphDocumentTests
    {
        private static GraphDocument NewDocument() => new(ModuleLibrary.BuildDefault());

        [Test]
        public void AddNode_MutatesGraph_AndIsResolvableByNodeId()
        {
            var doc = NewDocument();

            var node = doc.AddNode("sensor.enemyDistance", 10f, 20f);

            Assert.IsNotNull(node);
            Assert.AreEqual(1, doc.Graph.Nodes.Count);
            Assert.AreSame(node, doc.Graph.Nodes[0]);
            Assert.IsTrue(doc.TryGetNode(node.NodeId, out var resolved));
            Assert.AreSame(node, resolved);
        }

        [Test]
        public void AddNode_SeedsParameterDefaults()
        {
            var doc = NewDocument();

            var constant = doc.AddNode("logic.constant", 0f, 0f);

            // logic.constant declares one param "value" defaulting to 0.5.
            Assert.AreEqual(0.5f, constant.Params["value"]);
        }

        [Test]
        public void AddNode_UnknownType_ReturnsNullAndAddsNothing()
        {
            var doc = NewDocument();

            Assert.IsNull(doc.AddNode("does.not.exist", 0f, 0f));
            Assert.AreEqual(0, doc.Graph.Nodes.Count);
        }

        [Test]
        public void AddNode_GivesEachNodeAUniqueId()
        {
            var doc = NewDocument();

            var a = doc.AddNode("sensor.enemyDistance", 0f, 0f);
            var b = doc.AddNode("sensor.enemyDistance", 0f, 0f);

            Assert.AreNotEqual(a.NodeId, b.NodeId);
        }

        [Test]
        public void RemoveNode_RemovesNode_AndIncidentConnections()
        {
            var doc = NewDocument();
            var dist = doc.AddNode("sensor.enemyDistance", 0f, 0f);
            var cmp = doc.AddNode("logic.greaterThan", 0f, 0f);
            doc.AddConnection(dist.NodeId, "value", cmp.NodeId, "a");

            Assert.IsTrue(doc.RemoveNode(dist.NodeId));

            Assert.IsFalse(doc.TryGetNode(dist.NodeId, out _));
            Assert.AreEqual(1, doc.Graph.Nodes.Count);
            Assert.AreEqual(0, doc.Graph.Connections.Count, "connections touching a removed node must be dropped");
        }

        [Test]
        public void RemoveNode_UnknownId_ReturnsFalse()
        {
            Assert.IsFalse(NewDocument().RemoveNode("nope"));
        }

        [Test]
        public void MoveNode_UpdatesPosition()
        {
            var doc = NewDocument();
            var node = doc.AddNode("sensor.enemyDistance", 1f, 2f);

            Assert.IsTrue(doc.MoveNode(node.NodeId, 5f, 6f));
            Assert.AreEqual(5f, node.X);
            Assert.AreEqual(6f, node.Y);
        }

        [Test]
        public void SetParam_UpdatesValue()
        {
            var doc = NewDocument();
            var node = doc.AddNode("logic.constant", 0f, 0f);

            Assert.IsTrue(doc.SetParam(node.NodeId, "value", -0.25f));
            Assert.AreEqual(-0.25f, node.Params["value"]);
        }

        [Test]
        public void AddConnection_MutatesGraph()
        {
            var doc = NewDocument();
            var dist = doc.AddNode("sensor.enemyDistance", 0f, 0f);
            var cmp = doc.AddNode("logic.greaterThan", 0f, 0f);

            var conn = doc.AddConnection(dist.NodeId, "value", cmp.NodeId, "a");

            Assert.IsNotNull(conn);
            Assert.AreEqual(1, doc.Graph.Connections.Count);
        }

        [Test]
        public void AddConnection_UnknownEndpoint_ReturnsNull()
        {
            var doc = NewDocument();
            var dist = doc.AddNode("sensor.enemyDistance", 0f, 0f);

            Assert.IsNull(doc.AddConnection(dist.NodeId, "value", "ghost", "a"));
            Assert.AreEqual(0, doc.Graph.Connections.Count);
        }

        [Test]
        public void RemoveConnection_CutsMatchingWireOnly()
        {
            var doc = NewDocument();
            var dist = doc.AddNode("sensor.enemyDistance", 0f, 0f);
            var cmp = doc.AddNode("logic.greaterThan", 0f, 0f);
            doc.AddConnection(dist.NodeId, "value", cmp.NodeId, "a");

            Assert.IsTrue(doc.RemoveConnection(dist.NodeId, "value", cmp.NodeId, "a"));
            Assert.AreEqual(0, doc.Graph.Connections.Count);
            Assert.IsFalse(doc.RemoveConnection(dist.NodeId, "value", cmp.NodeId, "a"));
        }

        [Test]
        public void Validate_ReflectsLiveEdits()
        {
            var doc = NewDocument();
            // An Action with its required "when" trigger unconnected is invalid...
            doc.AddNode("action.dash", 0f, 0f);
            Assert.IsFalse(doc.IsValid);

            // ...wiring a bool source into the trigger makes it valid.
            var ready = doc.AddNode("sensor.dashReady", 0f, 0f);
            var dash = doc.Graph.Nodes.Find(n => n.TypeId == "action.dash");
            doc.AddConnection(ready.NodeId, "value", dash.NodeId, "when");
            Assert.IsTrue(doc.IsValid);
        }

        [Test]
        public void FromJson_RebuildsBoard_WithNodesResolvableById()
        {
            var source = NewDocument();
            var dist = source.AddNode("sensor.enemyDistance", 10f, 20f);
            var cmp = source.AddNode("logic.greaterThan", 30f, 40f);
            source.AddConnection(dist.NodeId, "value", cmp.NodeId, "a");
            source.Name = "Rebuilt";

            var loaded = GraphDocument.FromJson(source.ToJson(), ModuleLibrary.BuildDefault());

            Assert.AreEqual("Rebuilt", loaded.Name);
            Assert.AreEqual(2, loaded.Nodes.Count);
            Assert.AreEqual(1, loaded.Connections.Count);
            Assert.IsTrue(loaded.TryGetNode(dist.NodeId, out var back));
            Assert.AreEqual(10f, back.X);
            Assert.AreEqual(20f, back.Y);
        }

        [Test]
        public void AddNode_AfterLoad_DoesNotCollideWithLoadedIds()
        {
            // A loaded graph already using "node-0" must not get a duplicate from a later AddNode.
            var seeded = new BotGraph();
            seeded.Nodes.Add(new GraphNode { NodeId = "node-0", TypeId = "sensor.enemyDistance" });
            var doc = new GraphDocument(ModuleLibrary.BuildDefault(), seeded);

            var added = doc.AddNode("sensor.enemyDistance", 0f, 0f);

            Assert.AreNotEqual("node-0", added.NodeId);
            Assert.IsTrue(doc.IsValid, "no duplicate node ids after adding onto a loaded graph");
        }
    }
}
