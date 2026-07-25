using System.Collections.Generic;
using System.Linq;

namespace SumoServices.Tests
{
    /// <summary>
    /// In-memory IPlayerCatalogStore for unit tests. Mirrors InMemoryMarketStore: pass nothing
    /// to start "never persisted" (Load returns null), or an initial set to preload. Used to
    /// exercise LocalCatalogService's merge-on-load and the publish persistence round-trip
    /// without touching disk. Note: items are held by reference (not deep-cloned) because
    /// CatalogItem is polymorphic (BotScript/Skin) — the tests here don't mutate stored items.
    /// </summary>
    public class InMemoryPlayerCatalogStore : IPlayerCatalogStore
    {
        private List<CatalogItem> saved;

        public InMemoryPlayerCatalogStore(IEnumerable<CatalogItem> initial = null)
        {
            saved = initial?.ToList();
        }

        public List<CatalogItem> Load() => saved?.ToList();

        public void Save(IReadOnlyList<CatalogItem> items) => saved = items.ToList();
    }
}
