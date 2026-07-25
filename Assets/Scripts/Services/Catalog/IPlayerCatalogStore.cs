using System.Collections.Generic;

namespace SumoServices
{
    /// <summary>
    /// Persistence seam for player-authored (Source=Player) catalog items, split out of
    /// LocalCatalogService the same way IMarketStore was split out of LocalMarketService: so
    /// the catalog's merge/index logic can be unit-tested against an in-memory store with no
    /// file I/O. FilePlayerCatalogStore is the production implementation; a networked store
    /// can replace it later behind this interface.
    ///
    /// This is a GLOBAL store (all players' published items), not per-player — a buyer must be
    /// able to resolve the seller's authored item by id. Ownership stays per-player in
    /// PlayerData; authorship is carried on the item itself (CatalogItem.Author).
    /// </summary>
    public interface IPlayerCatalogStore
    {
        /// <summary>
        /// The persisted player-authored items, or null when nothing has ever been persisted
        /// (first run). Null-vs-empty is kept for symmetry with IMarketStore; the catalog
        /// treats both as "no player items yet".
        /// </summary>
        List<CatalogItem> Load();

        /// <summary>Persist the given player-authored items, replacing whatever was stored.</summary>
        void Save(IReadOnlyList<CatalogItem> items);
    }
}
