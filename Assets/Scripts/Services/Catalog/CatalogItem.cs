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
        /// <summary>
        /// Discriminator used by CatalogItemConverter to pick the concrete type when
        /// deserializing items.json (e.g. "BotScript" -> BotScriptItem). Empty/unknown
        /// values deserialize as the base CatalogItem.
        /// </summary>
        public string Type;

        /// <summary>Stable id stored in PlayerData.OwnedItemIds. Must be unique across the catalog.</summary>
        public string Id;

        public string DisplayName;

        /// <summary>Equip slot this item goes into (e.g. "Wheel", "Eye", "Accessory"). Empty = not equippable.</summary>
        public string Slot;

        /// <summary>
        /// True for items every player owns from the start (the current default parts).
        /// Default items are granted for free at session start regardless of Price.
        /// </summary>
        public bool IsDefault;

        /// <summary>
        /// Coin cost in the Market. Sell refunds the same amount (full refund) for now.
        /// Ignored for IsDefault items. Zero means free.
        /// </summary>
        public int Price;

        /// <summary>Resources path to the item's sprite/icon, loaded on demand by the UI.</summary>
        public string IconResourcePath;

        /// <summary>
        /// Optional tint applied to the icon (e.g. "#FF00BD"), for items that reuse one base
        /// sprite in multiple colors (Skin - Body / Skin - Accessory). Empty = no tint (white).
        /// </summary>
        public string IconColor;

        /// <summary>
        /// Optional author/creator handle shown in the detail panel (also what the "Ask"
        /// button targets). Common to every category — a skin, a module or a bot script can
        /// all credit a creator. Empty = the Creator line and Ask button are hidden.
        /// </summary>
        public string Creator;

        /// <summary>
        /// Optional long description shown in the detail panel. Common to every category.
        /// Empty = the description section is hidden.
        /// </summary>
        public string Description;
    }
}
