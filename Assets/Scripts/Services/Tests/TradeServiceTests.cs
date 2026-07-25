using System.Threading.Tasks;
using NUnit.Framework;

namespace SumoServices.Tests
{
    // item 2: LocalTradeService owns no data — it composes catalog (prices) + player-data
    // (balance/inventory). With in-memory fakes we can assert its guards and the charge+grant
    // path with no scene, no PlayerPrefs, no file I/O.
    public class TradeServiceTests
    {
        private static (LocalTradeService trade, FakePlayerDataService player) NewTrade(int coins, params CatalogItem[] items)
        {
            var catalog = new FakeCatalogService(items);
            var player = new FakePlayerDataService();
            player.LoadAsync("buyer").GetAwaiter().GetResult();
            player.Current.Coins = coins;
            return (new LocalTradeService(catalog, player), player);
        }

        [Test]
        public async Task BuyAsync_UnknownItem_Fails()
        {
            var (trade, player) = NewTrade(1000);
            var result = await trade.BuyAsync("nope");

            Assert.IsFalse(result.Success);
            Assert.AreEqual(1000, player.Current.Coins);
        }

        [Test]
        public async Task BuyAsync_AlreadyOwned_Fails()
        {
            var (trade, player) = NewTrade(1000, new CatalogItem { Id = "wheel_x", Price = 100 });
            await player.GrantItemAsync("wheel_x");

            var result = await trade.BuyAsync("wheel_x");

            Assert.IsFalse(result.Success);
            Assert.AreEqual(1000, player.Current.Coins);
        }

        [Test]
        public async Task BuyAsync_InsufficientCoins_FailsWithoutCharging()
        {
            var (trade, player) = NewTrade(50, new CatalogItem { Id = "wheel_x", Price = 100 });

            var result = await trade.BuyAsync("wheel_x");

            Assert.IsFalse(result.Success);
            Assert.AreEqual(50, player.Current.Coins);
            Assert.IsFalse(player.Current.Owns("wheel_x"));
        }

        [Test]
        public async Task BuyAsync_Success_ChargesAndGrants()
        {
            var (trade, player) = NewTrade(1000, new CatalogItem { Id = "wheel_x", Price = 100 });

            var result = await trade.BuyAsync("wheel_x");

            Assert.IsTrue(result.Success, result.Error);
            Assert.AreEqual(900, player.Current.Coins);
            Assert.IsTrue(player.Current.Owns("wheel_x"));
        }
    }
}
