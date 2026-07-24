using System;
using System.Collections.Generic;
using SumoServices;
using UnityEngine;

namespace SumoCore
{
    /// <summary>
    /// Overlays a loadout's equipped Catalog skins onto a base Parts/PartTints pair for the
    /// equippable slots. The bridge from a Catalog skin id to a costume sprite is
    /// <see cref="SkinItem.PartSprite"/>.
    ///
    /// The caller supplies *which* loadout to apply — that is deliberate (decision-5): a local
    /// match resolves each seat from its own side's loadout, so the resolver must not read a
    /// single "signed-in equipped set" implicitly. Callers: GameManager.PlayerProfile (battle,
    /// this side's loadout) and the Garage bot preview (the loadout being edited).
    ///
    /// Equippable slots: Wheel, Accessory, Body. FaceSide is never equipped — it is the
    /// red/green side marker (UpdateSideColor).
    /// </summary>
    public static class EquippedCostumeResolver
    {
        /// <summary>
        /// Mutates <paramref name="parts"/>/<paramref name="tints"/> in place, applying the
        /// slot -> item-id map in <paramref name="equippedBySlot"/>. No-ops if that map is null/empty
        /// or the Catalog isn't ready yet, leaving the base (default) parts untouched.
        /// </summary>
        public static void ApplyEquipped(
            Dictionary<SumoPart, Sprite> parts,
            Dictionary<SumoPart, Color> tints,
            IReadOnlyDictionary<string, string> equippedBySlot)
        {
            if (equippedBySlot == null || equippedBySlot.Count == 0)
                return;

            var catalog = GameServices.Catalog;
            if (catalog == null)
                return;

            foreach (var pair in equippedBySlot)
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
                var resolvedTint = !string.IsNullOrEmpty(skin.IconColor)
                    && ColorUtility.TryParseHtmlString(skin.IconColor, out var tint)
                    ? tint
                    : Color.white;
                tints[part] = resolvedTint;
            }
        }
    }
}
