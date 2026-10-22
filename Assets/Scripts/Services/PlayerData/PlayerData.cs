using System;
using System.Collections.Generic;

namespace SumoServices
{
    /// <summary>
    /// The persistent, per-player save blob. This is the "local database row" for a
    /// player: owned items (the Inventory) + equipped costume. It is loaded on sign-in
    /// and saved after any change. Kept as plain serializable data so the same shape can
    /// later live in Cloud Save with no model changes.
    ///
    /// NOTE: no currency/economy here — the game has none yet. When the PM defines how
    /// items are acquired (coins, win-unlocks, etc.), add exactly that model then.
    /// </summary>
    [Serializable]
    public class PlayerData
    {
        // TODO(economy): placeholder starting balance so the Market is testable. Replace
        // with a real earning model (match rewards, daily grant, etc.) once the PM defines it.
        private const int StartingCoins = 5000;

        /// <summary>Owning account id. Ties this save to a PlayerAccount.</summary>
        public string PlayerId;

        /// <summary>Soft-currency balance. Spent in the Market, granted by earning flows.</summary>
        public int Coins;

        /// <summary>Ids of items the player owns — the backing set for the Inventory UI.</summary>
        public List<string> OwnedItemIds = new();

        /// <summary>Currently equipped costume item ids, keyed by slot name (e.g. "Wheel", "Eye").</summary>
        public Dictionary<string, string> EquippedBySlot = new();

        public bool Owns(string itemId) => OwnedItemIds.Contains(itemId);

        public static PlayerData CreateDefault(string playerId)
        {
            return new PlayerData
            {
                PlayerId = playerId,
                Coins = StartingCoins,
                OwnedItemIds = new List<string>(),
                EquippedBySlot = new Dictionary<string, string>()
            };
        }
    }
}
