using NUnit.Framework;

namespace SumoServices.Tests
{
    // First EditMode test — proves the Sumobot.Services asmdef + test harness compile and
    // run against pure service-layer logic (no scene, no PlayerPrefs, no file I/O, no fakes).
    // Targets PlayerData's model logic; the loadout migration has an idempotency contract
    // (decision-5) that is easy to break, which is exactly what a unit test should guard.
    public class PlayerDataTests
    {
        [Test]
        public void CreateDefault_StartsWithTwoLoadouts_MappedToBothSides()
        {
            var data = PlayerData.CreateDefault("player-1");

            Assert.AreEqual("player-1", data.PlayerId);
            Assert.AreEqual(2, data.Loadouts.Count);
            Assert.IsNotNull(data.GetLoadoutForSide(PlayerData.SideLeft));
            Assert.IsNotNull(data.GetLoadoutForSide(PlayerData.SideRight));
        }

        [Test]
        public void Owns_ReflectsInventory()
        {
            var data = PlayerData.CreateDefault("player-1");
            Assert.IsFalse(data.Owns("skin-1"));

            data.OwnedItemIds.Add("skin-1");
            Assert.IsTrue(data.Owns("skin-1"));
        }

        [Test]
        public void MigrateToLoadouts_CarriesLegacyEquippedSet_AndIsIdempotent()
        {
            var data = new PlayerData();
            data.EquippedBySlot["Wheel"] = "skin-wheel-3";

            data.MigrateToLoadouts();

            var left = data.GetLoadoutForSide(PlayerData.SideLeft);
            Assert.IsNotNull(left);
            Assert.AreEqual("skin-wheel-3", left.EquippedBySlot["Wheel"]);
            Assert.AreEqual(0, data.EquippedBySlot.Count, "legacy field should be consumed once migrated");

            // Second migration must not duplicate loadouts or resurrect the legacy set.
            data.MigrateToLoadouts();
            Assert.AreEqual(2, data.Loadouts.Count);
        }
    }
}
