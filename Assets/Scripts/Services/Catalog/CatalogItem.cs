using System;

namespace SumoServices
{
    /// <summary>
    /// One purchasable/ownable thing (a costume part, a skin, etc.). The catalog is the
    /// shared definition that both the Market (what's for sale) and the Inventory
    /// (what a player owns, resolved by Id) read from.
    /// </summary>
    [Serializable]
    public class CatalogItem
    {
        /// <summary>Stable id stored in PlayerData.OwnedItemIds. Must be unique across the catalog.</summary>
        public string Id;

        public string DisplayName;

        /// <summary>Equip slot this item goes into (e.g. "Wheel", "Eye", "Accessory"). Empty = not equippable.</summary>
        public string Slot;

        /// <summary>
        /// True for items every player owns from the start (the current default parts).
        /// Kept as a neutral flag on purpose: the game has no economy yet, so there is no
        /// price/acquisition model here. Add that when the PM defines how items are earned.
        /// </summary>
        public bool IsDefault;

        /// <summary>Resources path to the item's sprite/icon, loaded on demand by the UI.</summary>
        public string IconResourcePath;
    }
}
