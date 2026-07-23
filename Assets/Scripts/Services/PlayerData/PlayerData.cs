using System;
using System.Collections.Generic;

namespace SumoServices
{
    /// <summary>
    /// One named costume set: a slot -> item id map (e.g. "Wheel" -> "skin-wheel-3").
    /// A player owns several of these (see <see cref="PlayerData.Loadouts"/>) and local
    /// play maps each side to one, so P1 and P2 can wear different skins from the same
    /// shared inventory. See decision-5.
    /// </summary>
    [Serializable]
    public class Loadout
    {
        public string Id;
        public string Name;

        /// <summary>Equipped costume item ids for this loadout, keyed by slot name ("Wheel", "Body", ...).</summary>
        public Dictionary<string, string> EquippedBySlot = new();
    }

    /// <summary>
    /// The persistent, per-player save blob. This is the "local database row" for a
    /// player: owned items (the Inventory) + costume loadouts. It is loaded on sign-in
    /// and saved after any change. Kept as plain serializable data so the same shape can
    /// later live in Cloud Save with no model changes.
    ///
    /// decision-5: one account per device. <see cref="OwnedItemIds"/> and <see cref="Coins"/>
    /// are account-scoped and shared by every local participant; the *equipped set* is
    /// per side, held as named <see cref="Loadouts"/> with <see cref="ActiveLoadoutBySide"/>
    /// mapping each local seat to one.
    /// </summary>
    [Serializable]
    public class PlayerData
    {
        // TODO(economy): placeholder starting balance so the Market is testable. Replace
        // with a real earning model (match rewards, daily grant, etc.) once the PM defines it.
        private const int StartingCoins = 5000;

        // Side keys for ActiveLoadoutBySide. Kept as strings (not SumoCore.PlayerSide) so this
        // save model stays free of a SumoServices -> SumoCore dependency (SumoCore already
        // depends on SumoServices). Callers pass PlayerSide.ToString(), which matches these.
        public const string SideLeft = "Left";
        public const string SideRight = "Right";

        /// <summary>Owning account id. Ties this save to a PlayerAccount.</summary>
        public string PlayerId;

        /// <summary>Soft-currency balance. Spent in the Market, granted by earning flows. Account-scoped.</summary>
        public int Coins;

        /// <summary>Ids of items the player owns — the backing set for the Inventory UI. Account-scoped.</summary>
        public List<string> OwnedItemIds = new();

        /// <summary>The player's costume loadouts. One per local side today; see <see cref="ActiveLoadoutBySide"/>.</summary>
        public List<Loadout> Loadouts = new();

        /// <summary>Maps a local seat (<see cref="SideLeft"/>/<see cref="SideRight"/>) to a loadout id.</summary>
        public Dictionary<string, string> ActiveLoadoutBySide = new();

        /// <summary>
        /// LEGACY (pre-loadout saves): a single equipped set lived here. Read once by
        /// <see cref="MigrateToLoadouts"/> into <see cref="Loadouts"/> and then cleared.
        /// Do not read or write this going forward — use a <see cref="Loadout"/> instead.
        /// </summary>
        public Dictionary<string, string> EquippedBySlot = new();

        public bool Owns(string itemId) => OwnedItemIds.Contains(itemId);

        /// <summary>The loadout with this id, or null if unknown.</summary>
        public Loadout GetLoadout(string id)
        {
            if (string.IsNullOrEmpty(id) || Loadouts == null) return null;
            return Loadouts.Find(l => l != null && l.Id == id);
        }

        /// <summary>The loadout assigned to a local side ("Left"/"Right"), or null if unmapped.</summary>
        public Loadout GetLoadoutForSide(string side)
        {
            if (side == null || ActiveLoadoutBySide == null) return null;
            return ActiveLoadoutBySide.TryGetValue(side, out var id) ? GetLoadout(id) : null;
        }

        /// <summary>
        /// Bring a save up to the loadout model (decision-5). Idempotent: does nothing once
        /// <see cref="Loadouts"/> is populated. On first run it turns the legacy single
        /// <see cref="EquippedBySlot"/> into loadout "A" assigned to Left, adds an empty "B"
        /// assigned to Right, and clears the legacy field. This reproduces exactly what a
        /// pre-change save rendered (Left = its equipped set, Right = default parts).
        /// </summary>
        public void MigrateToLoadouts()
        {
            Loadouts ??= new();
            ActiveLoadoutBySide ??= new();
            EquippedBySlot ??= new();

            if (Loadouts.Count == 0)
            {
                Loadouts.Add(new Loadout
                {
                    Id = "loadout-a",
                    Name = "A",
                    EquippedBySlot = EquippedBySlot // carry the legacy set forward as-is
                });
                Loadouts.Add(new Loadout { Id = "loadout-b", Name = "B" });
                EquippedBySlot = new(); // consumed — no longer a source of truth
            }

            // Ensure both seats point at a real loadout (defensive against hand-edited saves).
            if (!ActiveLoadoutBySide.ContainsKey(SideLeft))
                ActiveLoadoutBySide[SideLeft] = Loadouts[0].Id;
            if (!ActiveLoadoutBySide.ContainsKey(SideRight))
                ActiveLoadoutBySide[SideRight] = Loadouts.Count > 1 ? Loadouts[1].Id : Loadouts[0].Id;
        }

        public static PlayerData CreateDefault(string playerId)
        {
            var data = new PlayerData
            {
                PlayerId = playerId,
                Coins = StartingCoins,
                OwnedItemIds = new List<string>()
            };
            data.MigrateToLoadouts(); // start with the two default loadouts + side mapping
            return data;
        }
    }
}
