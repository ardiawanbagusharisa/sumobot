using System;
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
