using System.Collections.Generic;
using UnityEngine;

namespace SumoCore
{
    /// <summary>
    /// The bot's base ("naked") costume: the sprite each equippable part falls back to when the
    /// player has no skin equipped in that slot. Single source of truth so the Garage preview
    /// (GaragePreviewController) and the Garage inventory's None/Unequip cell
    /// (GarageInventoryController) show the same base and never drift.
    ///
    /// These are the real Sprites/Character source sprites (what the bot renders in battle), not
    /// the cropped MarketIcons used by purchasable skin cells.
    /// </summary>
    public static class DefaultCostume
    {
        // Resource names under Sprites/Character/. Wheel/Eye/Accessory are numbered variants
        // whose "_1" is the base; Body is a single un-numbered sprite.
        private static readonly Dictionary<SumoPart, string> Names = new()
        {
            { SumoPart.Wheel, "Wheel_1" },
            { SumoPart.Eye, "Eye_1" },
            { SumoPart.Accessory, "Accessory_1" },
            { SumoPart.Body, "Body" },
        };

        /// <summary>Base sprite for a part, or null if it has no default (e.g. FaceSide).</summary>
        public static Sprite SpriteFor(SumoPart part)
        {
            if (!Names.TryGetValue(part, out var name))
                return null;

            var sprite = Resources.Load<Sprite>($"Sprites/Character/{name}");
            if (sprite == null)
                Logger.Error($"[Costume] Default sprite not found at Resources/Sprites/Character/{name}");
            return sprite;
        }

        /// <summary>Base sprite for a slot name (e.g. "Wheel"), or null if the name isn't an equippable part.</summary>
        public static Sprite SpriteForSlot(string slot)
        {
            return System.Enum.TryParse(slot, out SumoPart part) ? SpriteFor(part) : null;
        }
    }
}
