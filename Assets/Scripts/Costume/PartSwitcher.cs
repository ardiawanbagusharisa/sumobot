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
        if (targetImage == null)
            return;

        var profile = GameManager.Instance.GetProfileById();
        if (profile == null)
            return;

        // Always keep the profile's current/default part available. All extra
        // choices come from the signed-in player's owned inventory.
        if (profile.Parts.TryGetValue(part, out Sprite currentSprite) && currentSprite != null)
        {
            Color currentColor = profile.PartColors.TryGetValue(part, out Color savedColor)
                ? savedColor
                : Color.white;
            availableOptions.Add(new PartOption { Sprite = currentSprite, Color = currentColor });
        }

        var ownedIds = GameServices.PlayerData?.Current?.OwnedItemIds;
        if (ownedIds != null && GameServices.Catalog != null)
        {
            foreach (string itemId in ownedIds)
            {
                CatalogItem item = GameServices.Catalog.GetById(itemId);
                if (item == null ||
                    !string.Equals(item.Slot, part.ToString(), System.StringComparison.OrdinalIgnoreCase) ||
                    string.IsNullOrWhiteSpace(item.IconResourcePath))
                {
                    continue;
                }

                Sprite sprite = Resources.Load<Sprite>(item.IconResourcePath);
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
        if (GameManager.Instance.GetProfileById() == null)
            return;

        if (availableOptions.Count == 0 || targetImage == null)
            return;

        currentIndex = (currentIndex + direction + availableOptions.Count) % availableOptions.Count;
        PartOption option = availableOptions[currentIndex];
        ApplyOption(option);

        var profile = GameManager.Instance.GetProfileById();
        profile.Parts[part] = option.Sprite;
        profile.PartColors[part] = option.Color;

        if (option.Item != null &&
            GameServices.PlayerData?.Current?.Owns(option.Item.Id) == true &&
            profile.ID == GameServices.Auth?.Current?.PlayerId)
        {
            _ = EquipOwnedItemAsync(option.Item);
        }
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
