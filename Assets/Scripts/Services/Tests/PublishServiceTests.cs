using System.Linq;
using System.Threading.Tasks;
using NUnit.Framework;

namespace SumoServices.Tests
{
    // TASK-11 / decision-7: publishing turns a player's creation into an owned, Source=Player,
    // Author==self catalog item, which the TASK-7 Market flow can then list/sell. These tests
    // use the same fakes as MarketServiceTests: FakeCatalogService doubles as the read model and
    // the IPlayerCatalogWriter publish targets, and FakePlayerDataService switches identity so a
    // separate buyer can purchase the author's published item on one "machine".
    public class PublishServiceTests
    {
        private const string Author = "author-1";
        private const string Buyer = "buyer-1";

        private static (LocalPublishService publish, FakeCatalogService catalog, FakePlayerDataService player) NewPublish()
        {
            var catalog = new FakeCatalogService();
            var player = new FakePlayerDataService();
            var publish = new LocalPublishService(catalog, player);
            return (publish, catalog, player);
        }

        [Test]
        public async Task PublishAsync_Skin_StampsSourceAndAuthor_GrantsOwnership_AndIsResolvable()
        {
            var (publish, catalog, player) = NewPublish();
            await player.LoadAsync(Author);

            var result = await publish.PublishAsync(new SkinItem { DisplayName = "Neon Wheels", Slot = "Wheel", PartSprite = "Sprites/Character/Wheel_3" });

            Assert.IsTrue(result.Success, result.Error);
            string id = result.Value;
            StringAssert.StartsWith("player_", id);              // globally-unique, namespaced id

            var item = catalog.GetById(id);
            Assert.IsNotNull(item);                              // immediately resolvable in the read model
            Assert.AreEqual("Player", item.Source);
            Assert.AreEqual(Author, item.Author);
            Assert.AreEqual("Skin", item.Type);                 // discriminator stamped for store round-trip
            Assert.IsFalse(item.IsDefault);
            Assert.IsTrue(player.Current.Owns(id));             // author owns their creation
        }

        [Test]
        public async Task PublishedSkin_CanBeListedAndBought_AuthorStillOwns()
        {
            var (publish, catalog, player) = NewPublish();
            await player.LoadAsync(Author);
            string id = (await publish.PublishAsync(new SkinItem { DisplayName = "Neon Wheels", Slot = "Wheel" })).Value;

            var market = new LocalMarketService(catalog, player, new InMemoryMarketStore(new MarketListing[0]));
            await market.LoadAsync();

            var list = await market.ListAsync(id, 100);
            Assert.IsTrue(list.Success, list.Error);            // author can list what they published
            string listingId = market.ActiveListings.Single(l => l.ItemId == id).ListingId;

            await player.LoadAsync(Buyer);
            player.Current.Coins = 500;
            var buy = await market.BuyAsync(listingId);

            Assert.IsTrue(buy.Success, buy.Error);
            Assert.IsTrue(player.Current.Owns(id));             // buyer got a copy
            Assert.IsTrue(player.SaveFor(Author).Owns(id));     // author keeps their original (license/copy model)
        }

        [Test]
        public async Task PublishAsync_BotScript_IsGated_AndGrantsNothing()
        {
            var (publish, catalog, player) = NewPublish();
            await player.LoadAsync(Author);

            var result = await publish.PublishAsync(new BotScriptItem { DisplayName = "Genetic Algorithm", WinRate = 0.62f });

            Assert.IsFalse(result.Success);                     // gated until the script sandbox exists
            Assert.IsEmpty(catalog.AllItems);                   // nothing published
            Assert.IsFalse(player.Current.OwnedItemIds.Any());  // nothing granted
        }

        [Test]
        public async Task PublishAsync_NoPlayerLoaded_Fails()
        {
            var (publish, _, _) = NewPublish();

            var result = await publish.PublishAsync(new SkinItem { DisplayName = "Orphan" });

            Assert.IsFalse(result.Success);
        }

        [Test]
        public async Task PublishAsync_NullDraft_Fails()
        {
            var (publish, _, player) = NewPublish();
            await player.LoadAsync(Author);

            var result = await publish.PublishAsync(null);

            Assert.IsFalse(result.Success);
        }

        // Persistence + merge: a published item written to the player-catalog store is present
        // again after a "restart" (a fresh LocalCatalogService loading the same store). This
        // exercises LocalCatalogService's real merge-on-load path rather than the fake.
        [Test]
        public async Task PublishedItem_Persists_AcrossCatalogReload()
        {
            var store = new InMemoryPlayerCatalogStore();
            var player = new FakePlayerDataService();
            await player.LoadAsync(Author);

            var catalog = new LocalCatalogService(store);
            await catalog.LoadAsync();
            var publish = new LocalPublishService(catalog, player);

            string id = (await publish.PublishAsync(new SkinItem { DisplayName = "Persisted Skin", Slot = "Accessory" })).Value;
            Assert.IsNotNull(catalog.GetById(id));              // visible in the session that published it

            var reloaded = new LocalCatalogService(store);
            await reloaded.LoadAsync();
            Assert.IsNotNull(reloaded.GetById(id));             // and after a fresh load from the same store
        }
    }
}
