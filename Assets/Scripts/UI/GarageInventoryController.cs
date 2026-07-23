using System.Collections.Generic;
using SumoCore;
using SumoServices;
using UnityEngine;

// The Garage's owned-skins list for the currently selected loadout slot. Driven by
// GarageLoadoutController.SetSlot. Shows a fixed "None / Unequip" cell first, then the
// player's owned SkinItems for that slot; tapping a skin equips it, tapping None clears the
// slot. The currently equipped item is marked. When the player owns no skins for the slot,
// the list is just the "Default" cell (already the equipped state) — no separate empty panel.
//
// All three views of equipped state (this list's mark, the preview, the tabs) derive from
// the active loadout's EquippedBySlot and refresh via EquipmentChanged — no cached copy here.
public class GarageInventoryController : MonoBehaviour
{
    [Tooltip("Row prefab spawned once per owned skin. Assets/Prefabs/ItemCell.")]
    [SerializeField] private ItemCellView itemCellPrefab;

    [Tooltip("Where cells are spawned as children. This is the object with the GridLayoutGroup, " +
             "i.e. Scroll View > Viewport > Content.")]
    [SerializeField] private RectTransform content;

    [Tooltip("Fallback icon for the 'None / Unequip' cell. Normally the cell shows the current " +
             "slot's base 'naked' sprite (via DefaultCostume); this is only used if that can't be " +
             "resolved. Leave empty for a blank cell.")]
    [SerializeField] private Sprite unequipIcon;

    private string currentSlot;
    private IPlayerDataService subscribed;

    void OnEnable()
    {
        subscribed = GameServices.PlayerData;
        if (subscribed != null)
        {
            subscribed.InventoryChanged += Rebuild;
            subscribed.EquipmentChanged += Rebuild;
        }
        Rebuild();
    }

    void OnDisable()
    {
        if (subscribed != null)
        {
            subscribed.InventoryChanged -= Rebuild;
            subscribed.EquipmentChanged -= Rebuild;
            subscribed = null;
        }
    }

    /// <summary>Select which equip slot this list shows (called by GarageLoadoutController).</summary>
    public void SetSlot(string slot)
    {
        currentSlot = slot;
        Rebuild();
    }

    private void Rebuild()
    {
        if (content == null || string.IsNullOrEmpty(currentSlot)) return;

        for (int i = content.childCount - 1; i >= 0; i--)
            Destroy(content.GetChild(i).gameObject);

        var data = GameServices.PlayerData?.Current;
        var catalog = GameServices.Catalog;
        if (data == null || catalog == null) return;

        var owned = CollectOwnedSkinsForSlot(data, catalog);

        var equipped = GameServices.PlayerData?.ActiveLoadout?.EquippedBySlot;
        string equippedId = equipped != null && equipped.TryGetValue(currentSlot, out var eq) ? eq : null;

        // Fixed first cell: None / Unequip. Always present (even with zero owned skins) so the
        // player can always clear the slot back to the default part. Shows the slot's base
        // "naked" sprite so it reads as "the default look". Marked active when nothing is
        // equipped in this slot.
        Sprite noneSprite = DefaultCostume.SpriteForSlot(currentSlot) ?? unequipIcon;
        ItemCellView unequip = Instantiate(itemCellPrefab, content);
        unequip.BindUnequip(noneSprite);
        unequip.SetEquipped(equippedId == null);
        unequip.Clicked += OnCellClicked;

        foreach (var item in owned)
        {
            ItemCellView cell = Instantiate(itemCellPrefab, content);
            cell.Bind(item);
            cell.SetEquipped(item.Id == equippedId);
            cell.Clicked += OnCellClicked;
        }
    }

    private List<CatalogItem> CollectOwnedSkinsForSlot(PlayerData data, ICatalogService catalog)
    {
        var result = new List<CatalogItem>();
        foreach (string id in data.OwnedItemIds)
        {
            if (catalog.GetById(id) is SkinItem skin && skin.Slot == currentSlot)
                result.Add(skin);
        }
        return result;
    }

    // item == null is the Unequip cell.
    private async void OnCellClicked(CatalogItem item)
    {
        var playerData = GameServices.PlayerData;
        if (playerData == null || string.IsNullOrEmpty(currentSlot)) return;

        ServiceResult result = item == null
            ? await playerData.UnequipAsync(currentSlot)
            : await playerData.EquipAsync(currentSlot, item.Id);

        if (!result.Success)
            Logger.Warning($"[Garage] {(item == null ? "Unequip" : "Equip")} on slot '{currentSlot}' failed: {result.Error}");
        // On success, EquipmentChanged -> Rebuild refreshes the marks.
    }
}
