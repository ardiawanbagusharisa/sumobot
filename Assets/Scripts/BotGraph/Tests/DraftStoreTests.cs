using NUnit.Framework;
using SumoBot.Graph.Authoring;

namespace SumoBot.Graph.Tests
{
    // E3.1: the draft store is the data layer behind My Creations (E3.4). These cover the seam
    // contract against the in-memory store: named drafts coexist, save upserts by Id, load/delete
    // by Id, and a saved draft rebuilds the board with no data loss (decision-10).
    public class DraftStoreTests
    {
        private static GraphDraft SampleDraft(string name)
        {
            var doc = new GraphDocument(ModuleLibrary.BuildDefault());
            var dist = doc.AddNode("sensor.enemyDistance", 10f, 20f);
            var k = doc.AddNode("logic.constant", 30f, 40f);
            doc.SetParam(k.NodeId, "value", 0.5f);
            var cmp = doc.AddNode("logic.greaterThan", 50f, 60f);
            var acc = doc.AddNode("action.accelerate", 70f, 80f);
            doc.AddConnection(dist.NodeId, "value", cmp.NodeId, "a");
            doc.AddConnection(k.NodeId, "value", cmp.NodeId, "b");
            doc.AddConnection(cmp.NodeId, "value", acc.NodeId, "when");
            doc.Name = name;
            return GraphDraft.NewDraft(doc.Graph);
        }

        [Test]
        public void Save_ThenLoadAll_ReturnsMultipleNamedDrafts()
        {
            var store = new InMemoryDraftStore();
            store.Save(SampleDraft("Chaser"));
            store.Save(SampleDraft("Camper"));

            var all = store.LoadAll();

            Assert.AreEqual(2, all.Count);
            CollectionAssert.AreEquivalent(
                new[] { "Chaser", "Camper" },
                new[] { all[0].Name, all[1].Name });
        }

        [Test]
        public void Save_WithSameId_Upserts()
        {
            var store = new InMemoryDraftStore();
            var draft = SampleDraft("First");
            store.Save(draft);

            draft.Graph.Name = "Renamed";
            store.Save(draft);

            Assert.AreEqual(1, store.LoadAll().Count);
            Assert.AreEqual("Renamed", store.Load(draft.Id).Name);
        }

        [Test]
        public void Load_UnknownId_ReturnsNull()
        {
            Assert.IsNull(new InMemoryDraftStore().Load("missing"));
        }

        [Test]
        public void Delete_RemovesDraft()
        {
            var store = new InMemoryDraftStore();
            var draft = SampleDraft("Doomed");
            store.Save(draft);

            store.Delete(draft.Id);

            Assert.IsNull(store.Load(draft.Id));
            Assert.AreEqual(0, store.LoadAll().Count);
        }

        [Test]
        public void SavedDraft_ReloadsBoard_WithNoDataLoss()
        {
            var store = new InMemoryDraftStore();
            var draft = SampleDraft("Chaser");
            store.Save(draft);

            // Reload through JSON, exactly as the editor will when reopening a draft.
            var reloadedDraft = store.Load(draft.Id);
            var doc = GraphDocument.FromJson(GraphSerializer.ToJson(reloadedDraft.Graph), ModuleLibrary.BuildDefault());

            Assert.AreEqual("Chaser", doc.Name);
            Assert.AreEqual(4, doc.Nodes.Count);
            Assert.AreEqual(3, doc.Connections.Count);
            var constant = doc.Graph.Nodes.Find(n => n.TypeId == "logic.constant");
            Assert.AreEqual(0.5f, constant.Params["value"]);
            var accel = doc.Graph.Nodes.Find(n => n.TypeId == "action.accelerate");
            Assert.AreEqual(70f, accel.X);
            Assert.AreEqual(80f, accel.Y);
            Assert.IsTrue(doc.IsValid, "a round-tripped valid draft stays valid");
        }
    }
}
