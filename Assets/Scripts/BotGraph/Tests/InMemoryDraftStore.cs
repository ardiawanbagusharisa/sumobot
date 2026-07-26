using System.Collections.Generic;
using System.Linq;
using SumoBot.Graph.Authoring;

namespace SumoBot.Graph.Tests
{
    /// <summary>
    /// In-memory <see cref="IDraftStore"/> for unit tests, mirroring InMemoryPlayerCatalogStore.
    /// Holds drafts keyed by Id so the editor and My Creations logic can be exercised without
    /// touching disk. FileDraftStore is the production implementation and is intentionally not
    /// unit-tested (same convention as FilePlayerCatalogStore).
    /// </summary>
    public class InMemoryDraftStore : IDraftStore
    {
        private readonly Dictionary<string, GraphDraft> byId = new();

        public IReadOnlyList<GraphDraft> LoadAll() => byId.Values.ToList();

        public GraphDraft Load(string id)
            => id != null && byId.TryGetValue(id, out var draft) ? draft : null;

        public void Save(GraphDraft draft) => byId[draft.Id] = draft;

        public void Delete(string id)
        {
            if (id != null) byId.Remove(id);
        }
    }
}
