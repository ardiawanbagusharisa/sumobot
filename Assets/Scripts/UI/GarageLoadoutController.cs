using System;
using SumoCore;
using SumoServices;
using UnityEngine;
using UnityEngine.UI;

// The Garage's per-slot loadout tabs. Each tab is a Toggle (share one ToggleGroup in the
// editor so exactly one is active). Selecting a tab tells the inventory list which slot to
// show. Locked slots (e.g. Eye — no catalog variants yet) are non-interactable.
//
// Slot names must match the SkinItem.Slot values in items.json ("Wheel", "Accessory", "Body")
// and, by design, the SumoPart enum names.
public class GarageLoadoutController : MonoBehaviour
{
    [Serializable]
    public struct SlotTab
    {
        public Toggle toggle;
        public string slot;   // e.g. "Wheel", "Accessory", "Body"
        public bool locked;   // Eye today: shown but not selectable
    }

    [SerializeField] private SlotTab[] tabs;
    [SerializeField] private GarageInventoryController inventory; // driven when a tab is selected

    /// <summary>Raised with the newly selected slot name (extensibility hook; inventory is driven directly).</summary>
    public event Action<string> SelectedSlotChanged;

    private IPlayerDataService subscribed;

    void OnEnable()
    {
        string firstSelectable = null;

        foreach (var tab in tabs)
        {
            if (tab.toggle == null) continue;

            tab.toggle.interactable = !tab.locked;

            // Capture the slot for the closure.
            string slot = tab.slot;
            bool locked = tab.locked;
            // Clear runtime listeners first so re-enabling the panel doesn't stack duplicates
            // (this removes only code-added listeners, not the inspector/persistent ones).
            tab.toggle.onValueChanged.RemoveAllListeners();
            tab.toggle.onValueChanged.AddListener(isOn =>
            {
                if (isOn && !locked) Select(slot);
            });

            if (!locked && firstSelectable == null)
                firstSelectable = slot;
        }

        // Default to the first selectable tab so the list isn't empty on open.
        if (firstSelectable != null)
        {
            SetActiveToggle(firstSelectable);
            Select(firstSelectable);
        }

        subscribed = GameServices.PlayerData;
        if (subscribed != null)
            subscribed.EquipmentChanged += RefreshIcons;
        RefreshIcons();
    }

    void OnDisable()
    {
        if (subscribed != null)
        {
            subscribed.EquipmentChanged -= RefreshIcons;
            subscribed = null;
        }
    }

    /// <summary>
    /// Shows each tab's currently equipped skin (falling back to the slot's default "naked"
    /// sprite when nothing is equipped), so the tab row previews the loadout without needing
    /// the inventory list open. Mirrors the icon/tint lookup ItemCellView.Bind uses. The icon
    /// Image is found by child name (ItemCellView's "ImgItem", already present on every
    /// toggle) rather than a serialized field, so no extra scene wiring is needed.
    /// </summary>
    private void RefreshIcons()
    {
        var data = GameServices.PlayerData?.Current;
        var catalog = GameServices.Catalog;

        foreach (var tab in tabs)
        {
            if (tab.toggle == null) continue;
            Image icon = tab.toggle.transform.Find("ImgItem")?.GetComponent<Image>();
            if (icon == null) continue;

            SkinItem equipped = null;
            if (data != null && catalog != null
                && data.EquippedBySlot.TryGetValue(tab.slot, out var itemId)
                && catalog.GetById(itemId) is SkinItem skin)
                equipped = skin;

            if (equipped != null)
            {
                icon.sprite = !string.IsNullOrEmpty(equipped.IconResourcePath)
                    ? Resources.Load<Sprite>(equipped.IconResourcePath)
                    : null;
                icon.color = !string.IsNullOrEmpty(equipped.IconColor) && ColorUtility.TryParseHtmlString(equipped.IconColor, out var color)
                    ? color
                    : Color.white;
            }
            else
            {
                icon.sprite = DefaultCostume.SpriteForSlot(tab.slot);
                icon.color = Color.white;
            }
        }
    }

    private void SetActiveToggle(string slot)
    {
        foreach (var tab in tabs)
        {
            if (tab.toggle != null)
                tab.toggle.SetIsOnWithoutNotify(tab.slot == slot);
        }
    }

    private void Select(string slot)
    {
        if (inventory != null) inventory.SetSlot(slot);
        SelectedSlotChanged?.Invoke(slot);
    }
}
