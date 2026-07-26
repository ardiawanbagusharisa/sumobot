using System.IO;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using SumoBot.Graph.Authoring;

namespace SumoBot.Graph.Tests
{
    // E3.1: exercises the production file-backed store (not just the in-memory fake), proving the
    // one-file-per-draft-Id scheme and the GraphSerializer round-trip on disk. Runs against a
    // throwaway temp directory, enabled by FileDraftStore's injectable storage root.
    public class FileDraftStoreTests
    {
        private string rootDir;
        private FileDraftStore store;

        [SetUp]
        public void SetUp()
        {
            rootDir = Path.Combine(Path.GetTempPath(), "BotDraftsTest_" + System.Guid.NewGuid().ToString("N"));
            store = new FileDraftStore(rootDir);
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(rootDir)) Directory.Delete(rootDir, recursive: true);
        }

        private static GraphDraft SampleDraft(string name)
        {
            var doc = new GraphDocument(ModuleLibrary.BuildDefault());
            var dist = doc.AddNode("sensor.enemyDistance", 10f, 20f);
            var cmp = doc.AddNode("logic.greaterThan", 30f, 40f);
            doc.AddConnection(dist.NodeId, "value", cmp.NodeId, "a");
            doc.Name = name;
            return GraphDraft.NewDraft(doc.Graph);
        }

        [Test]
        public void LoadAll_OnFirstRun_IsEmpty()
        {
            Assert.AreEqual(0, store.LoadAll().Count);
        }

        [Test]
        public void Save_WritesOneFilePerDraftId()
        {
            store.Save(SampleDraft("A"));
            store.Save(SampleDraft("B"));

            Assert.AreEqual(2, Directory.GetFiles(rootDir, "*.json").Length);
            Assert.AreEqual(2, store.LoadAll().Count);
        }

        [Test]
        public void Save_ThenLoad_RoundTripsThroughDisk()
        {
            var draft = SampleDraft("Chaser");
            store.Save(draft);

            var loaded = store.Load(draft.Id);

            Assert.IsNotNull(loaded);
            Assert.AreEqual(draft.Id, loaded.Id);
            Assert.AreEqual("Chaser", loaded.Name);
            Assert.AreEqual(2, loaded.Graph.Nodes.Count);
            Assert.AreEqual(1, loaded.Graph.Connections.Count);
        }

        [Test]
        public void Save_SameId_OverwritesFile()
        {
            var draft = SampleDraft("First");
            store.Save(draft);
            draft.Graph.Name = "Renamed";
            store.Save(draft);

            Assert.AreEqual(1, Directory.GetFiles(rootDir, "*.json").Length);
            Assert.AreEqual("Renamed", store.Load(draft.Id).Name);
        }

        [Test]
        public void Delete_RemovesFile()
        {
            var draft = SampleDraft("Doomed");
            store.Save(draft);

            store.Delete(draft.Id);

            Assert.IsNull(store.Load(draft.Id));
            Assert.AreEqual(0, store.LoadAll().Count);
        }

        [Test]
        public void LoadAll_SkipsCorruptFile()
        {
            store.Save(SampleDraft("Good"));
            File.WriteAllText(Path.Combine(rootDir, "broken.json"), "{ this is not valid json");

            // FileDraftStore logs the skip via Logger.Error (-> Debug.LogError); expect it so the
            // Unity test runner does not treat the logged error as a failure.
            LogAssert.Expect(LogType.Error, new Regex(@"\[BotDrafts\] Failed to read"));

            var all = store.LoadAll();

            Assert.AreEqual(1, all.Count, "a corrupt draft file is skipped, not fatal");
            Assert.AreEqual("Good", all[0].Name);
        }
    }
}
