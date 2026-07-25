using System.Linq;
using System.Threading.Tasks;
using NUnit.Framework;

namespace SumoServices.Tests
{
    // item 3: with the storage seam (IMarketStore) the P2P Market is unit-testable end-to-end
    // on one "machine" — the fake player-data service switches acting identity so a buyer and
    // seller can both exist, which the file-backed single-device store could never simulate.
    public class MarketServiceTests
    {
        private const string Seller = "seller-1";
        private const string Buyer = "buyer-1";
        private const string ItemId = "bot_authored";

        // A market whose catalog has one Player-authored item owned/authored by Seller, backed
        // by an empty (persisted, non-null) store so LoadAsync does not seed demo listings.
        private static (LocalMarketService market, FakePlayerDataService player) NewMarket()
        {
            var catalog = new FakeCatalogService(
                new CatalogItem { Id = ItemId, Price = 0, Source = "Player", Author = Seller, IsDefault = false });
            var player = new FakePlayerDataService();
            var market = new LocalMarketService(catalog, player, new InMemoryMarketStore(new MarketListing[0]));
            market.LoadAsync().GetAwaiter().GetResult();
            return (market, player);
        }

        private static async Task<string> ListAsSeller(LocalMarketService market, FakePlayerDataService player, int price)
        {
            await player.LoadAsync(Seller);
            await player.GrantItemAsync(ItemId); // seller owns the item they authored
            var list = await market.ListAsync(ItemId, price);
            Assert.IsTrue(list.Success, list.Error);
            return market.ActiveListings.Single(l => l.ItemId == ItemId && l.SellerId == Seller).ListingId;
        }

        [Test]
        public async Task ListAsync_NonAuthor_Fails()
        {
            var (market, player) = NewMarket();
            await player.LoadAsync(Buyer);
            await player.GrantItemAsync(ItemId); // owns a copy, but is not the author

            var result = await market.ListAsync(ItemId, 100);

            Assert.IsFalse(result.Success);
        }

        [Test]
        public async Task BuyAsync_SplitsNinetyTen_TransfersItem_AndKeepsListingActive()
        {
            var (market, player) = NewMarket();
            var listingId = await ListAsSeller(market, player, 100);

            await player.LoadAsync(Buyer);
            player.Current.Coins = 500;
            int sellerBefore = player.SaveFor(Seller).Coins;

            var result = await market.BuyAsync(listingId);

            Assert.IsTrue(result.Success, result.Error);
            Assert.AreEqual(400, player.Current.Coins);                       // buyer charged full price
            Assert.IsTrue(player.Current.Owns(ItemId));                       // buyer got a copy
            Assert.AreEqual(sellerBefore + 90, player.SaveFor(Seller).Coins); // seller +90% (sink keeps 10)
            Assert.IsTrue(market.ActiveListings.Any(l => l.ListingId == listingId)); // unlimited-copy model
        }

        [Test]
        public async Task BuyAsync_OwnListing_Fails()
        {
            var (market, player) = NewMarket();
            var listingId = await ListAsSeller(market, player, 100); // still acting as seller

            var result = await market.BuyAsync(listingId);

            Assert.IsFalse(result.Success);
        }

        [Test]
        public async Task BuyAsync_InsufficientCoins_FailsWithoutCharging()
        {
            var (market, player) = NewMarket();
            var listingId = await ListAsSeller(market, player, 100);

            await player.LoadAsync(Buyer);
            player.Current.Coins = 50;

            var result = await market.BuyAsync(listingId);

            Assert.IsFalse(result.Success);
            Assert.AreEqual(50, player.Current.Coins);
            Assert.IsFalse(player.Current.Owns(ItemId));
        }

        [Test]
        public async Task UnlistAsync_BySeller_RemovesFromActive_AndBlocksBuy()
        {
            var (market, player) = NewMarket();
            var listingId = await ListAsSeller(market, player, 100);

            var unlist = await market.UnlistAsync(listingId);
            Assert.IsTrue(unlist.Success, unlist.Error);
            Assert.IsFalse(market.ActiveListings.Any(l => l.ListingId == listingId));

            await player.LoadAsync(Buyer);
            player.Current.Coins = 500;
            var buy = await market.BuyAsync(listingId);
            Assert.IsFalse(buy.Success); // unlisted listings are not buyable
        }

        [Test]
        public async Task UnlistAsync_ByNonSeller_Fails()
        {
            var (market, player) = NewMarket();
            var listingId = await ListAsSeller(market, player, 100);

            await player.LoadAsync(Buyer);
            var result = await market.UnlistAsync(listingId);

            Assert.IsFalse(result.Success);
            Assert.IsTrue(market.ActiveListings.Any(l => l.ListingId == listingId)); // still active
        }

        [Test]
        public async Task RepriceAsync_BySeller_ChangesBuyPrice()
        {
            var (market, player) = NewMarket();
            var listingId = await ListAsSeller(market, player, 100);

            var reprice = await market.RepriceAsync(listingId, 200);
            Assert.IsTrue(reprice.Success, reprice.Error);

            await player.LoadAsync(Buyer);
            player.Current.Coins = 500;
            var buy = await market.BuyAsync(listingId);

            Assert.IsTrue(buy.Success, buy.Error);
            Assert.AreEqual(300, player.Current.Coins); // charged the repriced 200, not the original 100
        }

        [Test]
        public async Task RepriceAsync_ByNonSeller_Fails()
        {
            var (market, player) = NewMarket();
            var listingId = await ListAsSeller(market, player, 100);

            await player.LoadAsync(Buyer);
            var result = await market.RepriceAsync(listingId, 200);

            Assert.IsFalse(result.Success);
        }
    }
}
