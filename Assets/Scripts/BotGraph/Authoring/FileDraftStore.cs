using System;
using System.Collections.Generic;
using System.IO;

namespace SumoBot.Graph.Authoring
{
    /// <summary>
    /// Default <see cref="IDraftStore"/>: one JSON file per draft under a root directory, named
    /// <c>{Id}.json</c>. The draft Id lives in the filename; the file body is exactly the graph
    /// JSON produced by <see cref="GraphSerializer"/>, so a draft on disk is the same shape a
    /// publish (E5) will carry — no wrapper envelope.
    ///
    /// The storage root is injected (typically Application.persistentDataPath/BotDrafts, supplied
    /// by the UI composition point) so this class stays engine-free — <c>System.IO</c> only, no
    /// <c>UnityEngine</c> — keeping the data assembly's one-way boundary intact and letting the
    /// store be tested against a temp directory.
    ///
    /// A missing directory is a clean first run (no drafts). A corrupt/unreadable file is skipped
    /// (LoadAll) or treated as absent (Load) and logged, so one bad draft never breaks the My
    /// Creations list.
    /// </summary>
    public class FileDraftStore : IDraftStore
    {
        private readonly string rootDir;

        public FileDraftStore(string rootDir)
        {
            if (string.IsNullOrEmpty(rootDir))
                throw new ArgumentException("A draft store needs a non-empty root directory.", nameof(rootDir));
            this.rootDir = rootDir;
        }

        private string PathFor(string id) => Path.Combine(rootDir, id + ".json");

        public IReadOnlyList<GraphDraft> LoadAll()
        {
            var drafts = new List<GraphDraft>();
            if (!Directory.Exists(rootDir)) return drafts;

            foreach (string file in Directory.GetFiles(rootDir, "*.json"))
            {
                var draft = ReadFile(file);
                if (draft != null) drafts.Add(draft);
            }
            return drafts;
        }

        public GraphDraft Load(string id)
        {
            if (string.IsNullOrEmpty(id)) return null;
            string path = PathFor(id);
            return File.Exists(path) ? ReadFile(path) : null;
        }

        public void Save(GraphDraft draft)
        {
            if (draft == null || string.IsNullOrEmpty(draft.Id))
                throw new ArgumentException("A draft must have a non-empty Id.", nameof(draft));

            Directory.CreateDirectory(rootDir);
            File.WriteAllText(PathFor(draft.Id), GraphSerializer.ToJson(draft.Graph ?? new BotGraph()));
        }

        public void Delete(string id)
        {
            if (string.IsNullOrEmpty(id)) return;
            string path = PathFor(id);
            if (File.Exists(path)) File.Delete(path);
        }

        // The Id is the filename stem; the body is the graph JSON. A read failure degrades to null
        // so a single corrupt draft is skipped rather than crashing the whole list.
        private static GraphDraft ReadFile(string path)
        {
            try
            {
                var graph = GraphSerializer.FromJson(File.ReadAllText(path));
                if (graph == null) return null;
                return new GraphDraft(Path.GetFileNameWithoutExtension(path), graph);
            }
            catch (Exception e)
            {
                Logger.Error($"[BotDrafts] Failed to read {path}, skipping draft: {e.Message}");
                return null;
            }
        }
    }
}
