using System.Collections.Generic;

namespace SumoServices
{
    /// <summary>
    /// Persistence seam for the Market's listings, split out of LocalMarketService so the
    /// service's orchestration (list/unlist/reprice/buy) can be unit-tested against an
    /// in-memory store with no file I/O. FileMarketStore is the production implementation;
    /// a networked store can replace it later, mirroring the service-layer pattern.
    /// </summary>
    public interface IMarketStore
    {
        /// <summary>
        /// The persisted listings, or null when nothing has ever been persisted (first run).
        /// The null-vs-empty distinction lets the caller seed only on a genuine first run,
        /// not when a player has legitimately unlisted everything.
        /// </summary>
        List<MarketListing> Load();

        /// <summary>Persist the given listings, replacing whatever was stored.</summary>
        void Save(IReadOnlyList<MarketListing> listings);
    }
}
