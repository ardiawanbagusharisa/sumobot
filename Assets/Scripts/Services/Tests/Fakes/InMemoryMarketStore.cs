using System.Collections.Generic;
using System.Linq;

namespace SumoServices.Tests
{
    /// <summary>
    /// In-memory IMarketStore for unit tests. Clones on Load/Save so the service's working
    /// list is never aliased with the stored copy — mirroring how FileMarketStore round-trips
    /// through JSON. Pass an empty list to start "persisted but empty" (no first-run seeding);
    /// pass nothing to start "never persisted" (the service will seed).
    /// </summary>
    public class InMemoryMarketStore : IMarketStore
    {
        private List<MarketListing> saved;

        public InMemoryMarketStore(IEnumerable<MarketListing> initial = null)
        {
            saved = initial?.Select(Clone).ToList();
        }

        public List<MarketListing> Load() => saved?.Select(Clone).ToList();

        public void Save(IReadOnlyList<MarketListing> listings) => saved = listings.Select(Clone).ToList();

        private static MarketListing Clone(MarketListing l) => new()
        {
            ListingId = l.ListingId,
            ItemId = l.ItemId,
            SellerId = l.SellerId,
            Price = l.Price,
            IsActive = l.IsActive
        };
    }
}
