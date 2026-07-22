using System;

namespace SumoServices
{
    /// <summary>
    /// One player's active or historical offer to sell copies of an item they authored,
    /// at a price they set. Distinct from CatalogItem: CatalogItem is the static item
    /// definition (shared by everyone); MarketListing is dynamic user-generated content —
    /// it lives in its own store (see IMarketService), not Resources/Catalog/items.json.
    /// The license/copy model is unlimited: buying does not deactivate a listing, it stays
    /// available for the next buyer. Only an explicit Unlist deactivates it.
    /// </summary>
    [Serializable]
    public class MarketListing
    {
        /// <summary>Stable listing id (distinct from ItemId — a seller could in principle relist).</summary>
        public string ListingId;

        /// <summary>The CatalogItem.Id being sold. Must resolve via ICatalogService and have Author == SellerId.</summary>
        public string ItemId;

        /// <summary>PlayerId of the seller. Must equal the item's CatalogItem.Author at list time.</summary>
        public string SellerId;

        /// <summary>Current asking price in coins. Mutable via RepriceAsync.</summary>
        public int Price;

        /// <summary>False once unlisted. Unlisting keeps the row (lightweight history, no id reuse); Browse/Buy only consider active listings.</summary>
        public bool IsActive = true;
    }
}
