using System.Collections.Generic;
using SumoCore;
using SumoServices;
using UnityEngine;
using UnityEngine.UI;

// The BotCreator's per-part ◀▶ arrow switcher. Kept as the legacy UI, but its data now lives
// in PlayerData loadouts — the same source of truth as the Garage (decision-5 / TASK-15.4).
//
// The arrows cycle: [None (default part / unequip)] + the account's OWNED skins for this slot.
// Unowned catalog items are NOT reachable here (they're acquired in the Store). Because
// BotCreator is opened per side (GameManager.EditingID -> Left/Right profile), this edits that
// side's loadout, so P1 and P2 dress their own bots from the one shared inventory.
public class PartSwitcher : MonoBehaviour
{
    public SumoPart part;
    public Image targetImage;
    public Image targetPreviewImage;

    // options[0] is always the "None" sentinel (default part / unequip); the rest are owned skins.
    private readonly List<CatalogItem> options = new();
    private int currentIndex;
    private bool busy;
    private IPlayerDataService subscribed;

    private string Slot => part.ToString();

    void OnEnable()
    {
        subscribed = GameServices.PlayerData;
        if (subscribed != null)
        {
            subscribed.EquipmentChanged += Refresh;
            subscribed.InventoryChanged += Refresh;
            SelectEditingSideLoadout(); // point the service at the side BotCreator opened for
        }
        Refresh();
    }

    void OnDisable()
    {
        if (subscribed != null)
        {
            subscribed.EquipmentChanged -= Refresh;
            subscribed.InventoryChanged -= Refresh;
            subscribed = null;
        }
    }

    // BotCreator edits GameManager.EditingID's profile; select that side's loadout so equip/
    // unequip act on the right bot. No-op if there's no editing profile or data isn't ready.
    private void SelectEditingSideLoadout()
    {
        var profile = GameManager.Instance.GetProfileById();
        var data = subscribed?.Current;
        if (profile == null || data == null) return;

        string loadoutId = data.GetLoadoutForSide(profile.Side.ToString())?.Id;
        if (!string.IsNullOrEmpty(loadoutId))
            subscribed.SelectLoadout(loadoutId);
    }

    // Rebuild the owned-skins cycle and sync the arrows/preview to what's currently equipped in
    // the active loadout for this slot. Reads only — never equips, so it's safe to call from the
    // EquipmentChanged/InventoryChanged events without a feedback loop.
    private void Refresh()
    {
        BuildOptions();

        var equipped = GameServices.PlayerData?.ActiveLoadout?.EquippedBySlot;
        string equippedId = equipped != null && equipped.TryGetValue(Slot, out var id) ? id : null;

        currentIndex = 0; // None
        if (!string.IsNullOrEmpty(equippedId))
        {
            int idx = options.FindIndex(o => o != null && o.Id == equippedId);
            if (idx >= 0) currentIndex = idx;
        }

        ApplyDisplay(options[currentIndex]);
    }

    private void BuildOptions()
    {
        options.Clear();
        options.Add(null); // [0] = None / default / unequip

        var data = GameServices.PlayerData?.Current;
        var catalog = GameServices.Catalog;
        if (data == null || catalog == null) return;

        foreach (string id in data.OwnedItemIds)
        {
            // Only equippable skins for this slot (a Slot + a real PartSprite).
            if (catalog.GetById(id) is SkinItem skin
                && skin.Slot == Slot
                && !string.IsNullOrEmpty(skin.PartSprite))
                options.Add(skin);
        }
    }

    // Called by the ◀ / ▶ buttons (direction = -1 / +1).
    public async void UpdateSprite(int direction)
    {
        SFXManager.Instance.Play2D("ui_accept_small");

        var playerData = GameServices.PlayerData;
        if (busy || playerData == null) return;

        if (options.Count == 0) BuildOptions();
        if (options.Count <= 1) return; // only None — nothing to cycle to

        int newIndex = (currentIndex + direction + options.Count) % options.Count;
        var target = options[newIndex];

        busy = true;
        try
        {
            ServiceResult result = target == null
                ? await playerData.UnequipAsync(Slot)
                : await playerData.EquipAsync(Slot, target.Id);

            if (result.Success)
            {
                currentIndex = newIndex;
                ApplyDisplay(target);
                // EquipAsync/UnequipAsync also raise EquipmentChanged -> Refresh, which reconciles.
            }
            else
            {
                Logger.Warning($"[BotCreator] Cycling {part} failed: {result.Error}");
            }
        }
        finally
        {
            busy = false;
        }
    }

    // item == null -> the default (unequipped) part.
    private void ApplyDisplay(CatalogItem item)
    {
        Sprite sprite;
        Color tint;

        if (item is SkinItem skin && !string.IsNullOrEmpty(skin.PartSprite))
        {
            sprite = Resources.Load<Sprite>(skin.PartSprite);
            tint = !string.IsNullOrEmpty(skin.IconColor)
                   && ColorUtility.TryParseHtmlString(skin.IconColor, out var c)
                ? c
                : Color.white;
        }
        else
        {
            sprite = DefaultCostume.SpriteFor(part);
            tint = Color.white;
        }

        if (targetImage != null) { targetImage.sprite = sprite; targetImage.color = tint; }
        if (targetPreviewImage != null) { targetPreviewImage.sprite = sprite; targetPreviewImage.color = tint; }
    }
}
