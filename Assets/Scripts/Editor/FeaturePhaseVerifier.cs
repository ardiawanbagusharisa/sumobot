using System;
using System.Collections.Generic;
using System.IO;
using SumoLeaderboard;
using SumoServices;
using UnityEditor;
using UnityEngine;

namespace SumoFeatures.Editor
{
    public static class FeaturePhaseVerifier
    {
        [MenuItem("Sumobot/Verify Local Feature Phases")]
        public static void VerifyLocalFeaturePhases()
        {
            string root = Path.Combine("Library", "FeaturePhaseVerifier", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);

            var catalog = new LocalCatalogService();
            Require(catalog.LoadAsync().GetAwaiter().GetResult().Success, "Catalog failed to load.");
            Require(catalog.AllItems.Count > 0, "Catalog is empty.");

            var playerData = new LocalPlayerDataService(root);
            Require(playerData.LoadAsync("feature_test_player").GetAwaiter().GetResult().Success,
                "Player data failed to load.");

            CatalogItem equippable = null;
            foreach (CatalogItem item in catalog.AllItems)
            {
                if (!item.IsDefault && item.Price > 0 && !string.IsNullOrEmpty(item.Slot))
                {
                    equippable = item;
                    break;
                }
            }

            Require(equippable != null, "No paid equippable catalog item was found.");
            int coinsBefore = playerData.Current.Coins;
            var trade = new LocalTradeService(catalog, playerData);
            Require(trade.BuyAsync(equippable.Id).GetAwaiter().GetResult().Success,
                "Market purchase failed.");
            Require(playerData.Current.Owns(equippable.Id), "Purchased item was not granted.");
            Require(playerData.Current.Coins == coinsBefore - equippable.Price,
                "Purchase did not deduct the expected coin amount.");

            Require(playerData.EquipAsync(equippable.Slot, equippable.Id)
                .GetAwaiter().GetResult().Success, "Equipment operation failed.");
            Require(playerData.Current.EquippedBySlot.TryGetValue(equippable.Slot, out string equipped) &&
                equipped == equippable.Id, "Equipped item was not persisted in memory.");

            var reloaded = new LocalPlayerDataService(root);
            Require(reloaded.LoadAsync("feature_test_player").GetAwaiter().GetResult().Success,
                "Persisted player data failed to reload.");
            Require(reloaded.Current.Owns(equippable.Id), "Inventory did not survive reload.");
            Require(reloaded.Current.EquippedBySlot.TryGetValue(equippable.Slot, out equipped) &&
                equipped == equippable.Id, "Equipment did not survive reload.");

            (int winner, int loser) = EloCalculator.UpdatePair(1000, 1000, EloCalculator.Win);
            Require(winner == 1016 && loser == 984, "Elo calculation returned an unexpected result.");

            var leaderboardStore = new LeaderboardStore(root);
            var data = new LeaderboardData
            {
                Tables = new List<LeaderboardTable>
                {
                    new()
                    {
                        GameMode = GameMode.Multiplayer,
                        Mode = PlayerMode.PvP,
                        Control = ControlCategory.Buttons,
                        Entries = new List<LeaderboardEntry>
                        {
                            new() { ProfileID = "feature_test_player", PlayerName = "Verifier" }
                        }
                    }
                }
            };
            leaderboardStore.Save(data);
            LeaderboardData loadedLeaderboard = leaderboardStore.Load();
            Require(loadedLeaderboard.Tables.Count == 1 &&
                loadedLeaderboard.Tables[0].Entries.Count == 1,
                "Leaderboard persistence failed.");

            Debug.Log($"[FeatureVerifier] PASS: login data, catalog, market, inventory, equipment, and leaderboard verified at {root}.");
        }

        private static void Require(bool condition, string message)
        {
            if (!condition)
                throw new InvalidOperationException("[FeatureVerifier] " + message);
        }
    }
}
