using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Owns the CONTENT and PURCHASE behaviour of the Market's item-detail panel: it fills the
// name/price/icon for the clicked CatalogItem and runs the Buy through GameServices.Trade.
// Panel visibility (show/hide, SFX, the exit button) stays with MarketManager, which owns
// the panel GameObject and is still bound to legacy scene onClicks — this controller only
// borrows its Show call so there is a single open path.
//
// After a successful buy, the coin balance and inventory refresh themselves via the
// IPlayerDataService events (see CoinBalanceView / InventoryController); this controller
// only updates its own Buy button to reflect the now-owned state. The Market is buy-only:
// there is no sell path in the UI.
public class ItemDetailController : MonoBehaviour
{
    [SerializeField] private MarketManager marketManager;

    [Header("Detail fields (moved here from MarketManager)")]
    [SerializeField] private TMP_Text nameText;
    [SerializeField] private TMP_Text priceText;
    [SerializeField] private Image iconImage;

    [Header("Purchase")]
    [SerializeField] private Button buyButton;
    [SerializeField] private TMP_Text buyButtonLabel; // optional: flips to "Owned" when owned

    private CatalogItem current;
    private bool purchasing;

    void Awake()
    {
        if (buyButton != null)
            buyButton.onClick.AddListener(OnBuyClicked);
    }

    /// <summary>Show the detail panel for a catalog item (wired from MarketListController).</summary>
    public void Show(CatalogItem item)
    {
        current = item;

        if (nameText != null) nameText.text = item.DisplayName;
        if (priceText != null) priceText.text = $"{item.Price}";
        if (iconImage != null && !string.IsNullOrEmpty(item.IconResourcePath))
        {
            iconImage.sprite = Resources.Load<Sprite>(item.IconResourcePath);
            // Same tint rule as ItemCellView so color variants that share a base sprite
            // (Skin - Body / Accessory) look right in the detail view too.
            iconImage.color = !string.IsNullOrEmpty(item.IconColor)
                && ColorUtility.TryParseHtmlString(item.IconColor, out var tint)
                ? tint
                : Color.white;
        }

        RefreshBuyState();

        // Reuse MarketManager's single show path (plays SFX, activates the panel).
        if (marketManager != null)
            marketManager.ShowItemDetail();
    }

    // Buttons need a void handler; guard against re-entrancy so a double-tap can't double-buy.
    private async void OnBuyClicked()
    {
        if (current == null || purchasing) return;

        purchasing = true;
        if (buyButton != null) buyButton.interactable = false;

        try
        {
            var result = await GameServices.Trade.BuyAsync(current.Id);
            if (!result.Success)
                Logger.Warning($"[Market] Buy '{current.Id}' failed: {result.Error}");
        }
        finally
        {
            purchasing = false;
            RefreshBuyState(); // re-enables the button unless the item is now owned
        }
    }

    // A player can't buy what they already own (LocalTradeService rejects it too); reflect
    // that in the button so the affordance matches the rule.
    private void RefreshBuyState()
    {
        bool owned = current != null && (GameServices.PlayerData?.Current?.Owns(current.Id) ?? false);
        if (buyButton != null) buyButton.interactable = !owned;
        if (buyButtonLabel != null) buyButtonLabel.text = owned ? "Owned" : "Buy";
    }
}
