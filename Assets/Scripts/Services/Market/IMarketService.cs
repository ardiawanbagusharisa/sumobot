using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Player-to-player (P2P) Market: browse other players' listings and buy copies, or
    /// manage your own listings (list/unlist/reprice). Distinct from ITradeService, which
    /// covers the Shop (Official items, buy-only, no listing lifecycle). Local JSON store
    /// for now; a later networked implementation can replace it behind this interface,
    /// mirroring ICatalogService/ITradeService.
    /// </summary>
    public interface IMarketService
    {
        /// <summary>Load (or seed, on first run) persisted listings. Call once at startup.</summary>
        Task<ServiceResult> LoadAsync();

        /// <summary>All currently active listings (Community tab browse).</summary>
        IReadOnlyList<MarketListing> ActiveListings { get; }

        /// <summary>Raised after listings change (list/unlist/reprice/buy) and persist.</summary>
        event Action ListingsChanged;

        /// <summary>
        /// List an owned+authored item for sale at the given price. Fails if the item is
        /// unknown, not owned, its Author != the current player, it is an Official item,
        /// or the current player already has an active listing for it (use RepriceAsync).
        /// </summary>
        Task<ServiceResult> ListAsync(string itemId, int price);

        /// <summary>Deactivate one of the current player's own listings. No-op (Ok) if already inactive.</summary>
        Task<ServiceResult> UnlistAsync(string listingId);

        /// <summary>Change the price of one of the current player's own active listings.</summary>
        Task<ServiceResult> RepriceAsync(string listingId, int newPrice);

        /// <summary>
        /// Buy a copy from an active listing: buyer -Price, seller +90% (rounded down),
        /// the remaining 10% is destroyed (platform sink), buyer is granted the item as a
        /// use-only copy. Fails without charging if the listing is inactive/unknown, the
        /// buyer is the seller, the buyer already owns the item, or the buyer can't afford it.
        /// </summary>
        Task<ServiceResult> BuyAsync(string listingId);
    }
}
