using System.Collections.Generic;
using System.Linq;
using SumoCore;
using SumoServices;
using UnityEngine;
using UnityEngine.UI;

public class PartSwitcher : MonoBehaviour
{
    public SumoPart part;
    public Sprite[] sprites;
    public Image targetImage;
    public Image targetPreviewImage;

    private sealed class PartOption
    {
        public Sprite Sprite;
        public Color Color;
        public CatalogItem Item;
    }

    private readonly List<PartOption> availableOptions = new();
    private int currentIndex;

    void Start()
    {
        BotCreatorInventoryController.EnsureCreated();

        if (targetImage == null)
            return;

        var profile = GameManager.Instance.GetProfileById();
        if (profile == null)
            return;

        var ownedIds = GameServices.PlayerData?.Current?.OwnedItemIds;
        if (ownedIds != null && GameServices.Catalog != null)
        {
            foreach (CatalogItem item in GameServices.Catalog.AllItems)
            {
                if (item == null ||
                    !ownedIds.Contains(item.Id) ||
                    !string.Equals(item.Slot, part.ToString(), System.StringComparison.OrdinalIgnoreCase) ||
                    string.IsNullOrWhiteSpace(item.IconResourcePath))
                {
                    continue;
                }

                Sprite sprite = Resources.Load<Sprite>(item.BotSpriteResourcePath);
                if (sprite == null)
                    continue;

                Color color = !string.IsNullOrEmpty(item.IconColor) &&
                    ColorUtility.TryParseHtmlString(item.IconColor, out Color tint)
                        ? tint
                        : Color.white;
                if (availableOptions.Any(option => option.Item?.Id == item.Id))
                    continue;

                availableOptions.Add(new PartOption { Sprite = sprite, Color = color, Item = item });
            }
        }

        // Keep the scene's original part as a fallback for old/local profiles
        // that do not own any catalog entry for this slot. Once inventory data
        // exists, every arrow choice maps to a persistable owned item.
        if (availableOptions.Count == 0 &&
            profile.Parts.TryGetValue(part, out Sprite currentSprite) &&
            currentSprite != null)
        {
            Color currentColor = profile.PartColors.TryGetValue(part, out Color savedColor)
                ? savedColor
                : Color.white;
            availableOptions.Add(new PartOption { Sprite = currentSprite, Color = currentColor });
        }

        if (availableOptions.Count == 0 && sprites != null)
        {
            foreach (Sprite sprite in sprites.Where(sprite => sprite != null).Take(1))
                availableOptions.Add(new PartOption { Sprite = sprite, Color = Color.white });
        }

        if (availableOptions.Count == 0)
            return;

        string equippedId = null;
        GameServices.PlayerData?.Current?.EquippedBySlot.TryGetValue(part.ToString(), out equippedId);
        int equippedIndex = availableOptions.FindIndex(option => option.Item?.Id == equippedId);
        currentIndex = equippedIndex >= 0 ? equippedIndex : 0;
        ApplyOption(availableOptions[currentIndex]);
    }

    public void UpdateSprite(int direction)
    {
        SFXManager.Instance.Play2D("ui_accept_small");
        PlayerProfile profile = GameManager.Instance.GetProfileById();
        if (profile == null)
            return;

        if (availableOptions.Count == 0 || targetImage == null)
            return;

        currentIndex = (currentIndex + direction + availableOptions.Count) % availableOptions.Count;
        PartOption option = availableOptions[currentIndex];
        ApplyToProfile(option);

        if (option.Item != null &&
            GameServices.PlayerData?.Current?.Owns(option.Item.Id) == true &&
            profile.ID == GameServices.Auth?.Current?.PlayerId)
        {
            _ = EquipOwnedItemAsync(option.Item);
        }
    }

    public bool ApplyOwnedItem(CatalogItem item)
    {
        if (item == null || !string.Equals(
                item.Slot,
                part.ToString(),
                System.StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        int index = availableOptions.FindIndex(option => option.Item?.Id == item.Id);
        if (index < 0)
            return false;

        currentIndex = index;
        ApplyToProfile(availableOptions[currentIndex]);
        return true;
    }

    private void ApplyToProfile(PartOption option)
    {
        ApplyOption(option);
        PlayerProfile profile = GameManager.Instance.GetProfileById();
        if (profile == null)
            return;
        profile.Parts[part] = option.Sprite;
        profile.PartColors[part] = option.Color;
    }

    private void ApplyOption(PartOption option)
    {
        targetImage.sprite = option.Sprite;
        targetImage.color = option.Color;
        if (targetPreviewImage != null)
        {
            targetPreviewImage.sprite = option.Sprite;
            targetPreviewImage.color = option.Color;
        }
    }

    private async System.Threading.Tasks.Task EquipOwnedItemAsync(CatalogItem item)
    {
        ServiceResult result = await GameServices.PlayerData.EquipAsync(part.ToString(), item.Id);
        if (!result.Success)
            Logger.Warning($"[BotCreator] Could not equip '{item.Id}': {result.Error}");
    }
}
