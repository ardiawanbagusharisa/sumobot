using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Local implementation of IMarketService. Listings are global marketplace state (every
    /// player's offers), not per-player like PlayerData. Composes the catalog (to validate
    /// authorship and resolve items) and the player-data service (balance + inventory +
    /// cross-player crediting); listing persistence is delegated to an injected
    /// <see cref="IMarketStore"/> (FileMarketStore by default), so this class is pure
    /// orchestration and is unit-testable against an in-memory store.
    /// </summary>
    public class LocalMarketService : IMarketService
    {
        private readonly ICatalogService catalog;
        private readonly IPlayerDataService playerData;
        private readonly IMarketStore store;

        private List<MarketListing> listings = new();

        public IReadOnlyList<MarketListing> ActiveListings => listings.Where(l => l.IsActive).ToList();

        public event Action ListingsChanged;

        // store defaults to the file-backed implementation so runtime wiring in GameServices
        // (new LocalMarketService(Catalog, PlayerData)) is unchanged; tests inject an in-memory
        // store to exercise list/buy/unlist without touching disk.
        public LocalMarketService(ICatalogService catalog, IPlayerDataService playerData, IMarketStore store = null)
        {
            this.catalog = catalog;
            this.playerData = playerData;
            this.store = store ?? new FileMarketStore();
        }

        public Task<ServiceResult> LoadAsync()
        {
            try
            {
                var loaded = store.Load();
                if (loaded != null)
                {
                    listings = loaded;
                }
                else
                {
                    // First run (nothing ever persisted): seed and save so the flow is testable
                    // end-to-end. A persisted-but-empty store is left empty (no re-seeding).
                    listings = BuildSeedListings();
                    Persist();
                }

                Logger.Info($"[Market] Loaded {listings.Count} listing(s), {ActiveListings.Count} active.");
                return Task.FromResult(ServiceResult.Ok());
            }
            catch (Exception e)
            {
                Logger.Error($"[Market] Load failed: {e.Message}");
                return Task.FromResult(ServiceResult.Fail(e.Message));
            }
        }

        public Task<ServiceResult> ListAsync(string itemId, int price)
        {
            if (playerData.Current == null) return Task.FromResult(ServiceResult.Fail("No player loaded."));
            if (price < 0) return Task.FromResult(ServiceResult.Fail("price must be non-negative."));

            var item = catalog.GetById(itemId);
            if (item == null) return Task.FromResult(ServiceResult.Fail($"Unknown item '{itemId}'."));
            if (!playerData.Current.Owns(itemId)) return Task.FromResult(ServiceResult.Fail("Cannot list an item you do not own."));

            string playerId = playerData.Current.PlayerId;
            if (string.IsNullOrEmpty(item.Author) || item.Author != playerId)
                return Task.FromResult(ServiceResult.Fail("Only the item's author can list it for sale."));

            if (listings.Any(l => l.IsActive && l.ItemId == itemId && l.SellerId == playerId))
                return Task.FromResult(ServiceResult.Fail("Already listed — use RepriceAsync to change the price."));

            listings.Add(new MarketListing
            {
                ListingId = Guid.NewGuid().ToString("N"),
                ItemId = itemId,
                SellerId = playerId,
                Price = price,
                IsActive = true
            });

            return Task.FromResult(Finish());
        }

        public Task<ServiceResult> UnlistAsync(string listingId)
        {
            if (playerData.Current == null) return Task.FromResult(ServiceResult.Fail("No player loaded."));

            var listing = listings.FirstOrDefault(l => l.ListingId == listingId);
            if (listing == null) return Task.FromResult(ServiceResult.Fail($"Unknown listing '{listingId}'."));
            if (listing.SellerId != playerData.Current.PlayerId) return Task.FromResult(ServiceResult.Fail("Only the seller can unlist this."));

            if (!listing.IsActive) return Task.FromResult(ServiceResult.Ok()); // already inactive — no change, no event

            listing.IsActive = false;
            return Task.FromResult(Finish());
        }

        public Task<ServiceResult> RepriceAsync(string listingId, int newPrice)
        {
            if (playerData.Current == null) return Task.FromResult(ServiceResult.Fail("No player loaded."));
            if (newPrice < 0) return Task.FromResult(ServiceResult.Fail("newPrice must be non-negative."));

            var listing = listings.FirstOrDefault(l => l.ListingId == listingId);
            if (listing == null) return Task.FromResult(ServiceResult.Fail($"Unknown listing '{listingId}'."));
            if (listing.SellerId != playerData.Current.PlayerId) return Task.FromResult(ServiceResult.Fail("Only the seller can reprice this."));
            if (!listing.IsActive) return Task.FromResult(ServiceResult.Fail("Cannot reprice an inactive listing."));

            listing.Price = newPrice;
            return Task.FromResult(Finish());
        }

        public async Task<ServiceResult> BuyAsync(string listingId)
        {
            if (playerData.Current == null) return ServiceResult.Fail("No player loaded.");

            var listing = listings.FirstOrDefault(l => l.ListingId == listingId);
            if (listing == null || !listing.IsActive) return ServiceResult.Fail("Listing is not available.");

            var item = catalog.GetById(listing.ItemId);
            if (item == null) return ServiceResult.Fail($"Unknown item '{listing.ItemId}'.");

            string buyerId = playerData.Current.PlayerId;
            if (buyerId == listing.SellerId) return ServiceResult.Fail("Cannot buy your own listing.");
            if (playerData.Current.Owns(listing.ItemId)) return ServiceResult.Fail("Item already owned.");

            var spend = await playerData.TrySpendCoinsAsync(listing.Price);
            if (!spend.Success) return spend;

            // 90/10 split: seller gets 90% (rounded down), the remainder is a platform
            // sink — destroyed, not credited anywhere. E.g. price=101 -> seller 90, sink 11.
            int sellerCut = listing.Price * 90 / 100;
            var credit = await playerData.AddCoinsToPlayerAsync(listing.SellerId, sellerCut);
            if (!credit.Success)
            {
                // Compensate the buyer so a failed seller-credit never silently takes their coins.
                await playerData.AddCoinsAsync(listing.Price);
                return credit;
            }

            var grant = await playerData.GrantItemAsync(listing.ItemId);
            if (!grant.Success)
            {
                // Buyer's coins are refunded, but the seller-credit/sink split is not
                // reversed — a rare-failure edge case accepted as out of scope for this MVP.
                await playerData.AddCoinsAsync(listing.Price);
                Logger.Warning($"[Market] Grant failed after seller credit for listing '{listingId}': {grant.Error}");
                return grant;
            }

            return ServiceResult.Ok();
        }

        private ServiceResult Finish()
        {
            var persist = Persist();
            if (!persist.Success) return persist;
            ListingsChanged?.Invoke();
            return ServiceResult.Ok();
        }

        private ServiceResult Persist()
        {
            try
            {
                store.Save(listings);
                return ServiceResult.Ok();
            }
            catch (Exception e)
            {
                Logger.Error($"[Market] Save failed: {e.Message}");
                return ServiceResult.Fail(e.Message);
            }
        }

        // Hand-seeded listings so the buy/list/unlist/reprice flow is testable end-to-end
        // before a real player-upload system exists (Phase 2). Mirrors the two BotScript
        // rows in items.json hand-tagged Source=Player, Author=seed_author_sumobot_labs.
        private List<MarketListing> BuildSeedListings()
        {
            return new List<MarketListing>
            {
                new() { ListingId = Guid.NewGuid().ToString("N"), ItemId = "bot_script_genetic_algorithm", SellerId = "seed_author_sumobot_labs", Price = 100, IsActive = true },
                new() { ListingId = Guid.NewGuid().ToString("N"), ItemId = "bot_script_behavior_tree", SellerId = "seed_author_sumobot_labs", Price = 150, IsActive = true },
            };
        }
    }
}
