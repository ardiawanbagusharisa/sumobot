using System;

namespace SumoServices
{
    /// <summary>
    /// A cosmetic skin (Type == "Skin" in items.json). Skins are equippable costume parts:
    /// the <see cref="CatalogItem.Slot"/> (e.g. "Wheel", "Accessory") maps 1:1 to a SumoPart,
    /// and <see cref="PartSprite"/> points at the *real* character sprite the bot renders.
    ///
    /// Why a separate path from <see cref="CatalogItem.IconResourcePath"/>: the icon is a
    /// cropped/normalized MarketIcons copy sized for the shop grid, which is NOT what should
    /// render on the bot. PartSprite is the un-cropped Sprites/Character source used in battle.
    /// A skin with an empty PartSprite (or empty Slot) is inert — it shows in the Market/Inventory
    /// but cannot be equipped (this is how Body skins stay non-equippable for now; see task-5).
    ///
    /// The equipped tint reuses <see cref="CatalogItem.IconColor"/>, so a color variant that
    /// shares one base sprite looks the same on the bot as it does on its icon.
    /// </summary>
    [Serializable]
    public class SkinItem : CatalogItem
    {
        /// <summary>
        /// Resources path to the character sprite equipping this skin puts on the bot
        /// (e.g. "Sprites/Character/Wheel_3"). Empty = not equippable.
        /// </summary>
        public string PartSprite;
    }
}
