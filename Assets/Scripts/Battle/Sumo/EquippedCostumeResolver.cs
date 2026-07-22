using System;
using System.Collections.Generic;
using SumoServices;
using UnityEngine;

namespace SumoCore
{
    /// <summary>
    /// Overlays the signed-in player's equipped Catalog skins (PlayerData.EquippedBySlot)
    /// onto a base Parts/PartTints pair for the equippable slots. The bridge from a
    /// Catalog skin id to a costume sprite is <see cref="SkinItem.PartSprite"/>.
    ///
    /// Shared by GameManager.PlayerProfile (battle-start costume, scoped to the local
    /// player) and the menu's Garage bot preview (always "the local signed-in player" —
    /// no opponent/ID-matching concept needed there). Equippable slots: Wheel, Accessory,
    /// Body. FaceSide is never equipped — it is the red/green side marker (UpdateSideColor).
    /// </summary>
    public static class EquippedCostumeResolver
    {
        /// <summary>
        /// Mutates <paramref name="parts"/>/<paramref name="tints"/> in place. No-ops if
        /// PlayerData or Catalog aren't ready yet.
        /// </summary>
        public static void ApplyEquipped(Dictionary<SumoPart, Sprite> parts, Dictionary<SumoPart, Color> tints)
        {
            var data = GameServices.PlayerData?.Current;
            if (data == null)
                return;

            var catalog = GameServices.Catalog;
            if (catalog == null)
                return;

            foreach (var pair in data.EquippedBySlot)
            {
                if (!Enum.TryParse(pair.Key, out SumoPart part) || !parts.ContainsKey(part))
                    continue; // slot name doesn't map to an equippable SumoPart — skip

                if (catalog.GetById(pair.Value) is not SkinItem skin || string.IsNullOrEmpty(skin.PartSprite))
                    continue;

                var sprite = Resources.Load<Sprite>(skin.PartSprite);
                if (sprite == null)
                {
                    Logger.Warning($"[Costume] Equipped skin '{skin.Id}' sprite not found at Resources/{skin.PartSprite}");
                    continue;
                }

                parts[part] = sprite;
                tints[part] = !string.IsNullOrEmpty(skin.IconColor)
                    && ColorUtility.TryParseHtmlString(skin.IconColor, out var tint)
                    ? tint
                    : Color.white;
            }
        }
    }
}
