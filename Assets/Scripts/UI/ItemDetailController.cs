using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Sole owner of the Market's item-detail panel: its visibility (show/hide + exit button),
// its content (name/price/icon for the clicked CatalogItem), and the purchase (Trade.BuyAsync).
// The same panel serves both the Market listing and the Inventory — MarketListController and
// InventoryController just call Show(item); neither this class nor they depend on MarketManager
// anymore (which is now purely chat/inventory panel toggles).
//
// After a successful buy, the coin balance and inventory refresh themselves via the
// IPlayerDataService events (see CoinBalanceView / InventoryController); this controller only
// updates its own Buy button. The Market is buy-only: there is no sell path.
public class ItemDetailController : MonoBehaviour
{
    [SerializeField] private GameObject panel; // PanelItemDetail — this controller owns its visibility

    [Header("Detail fields")]
    [SerializeField] private TMP_Text nameText;
    [SerializeField] private TMP_Text priceText;
    [SerializeField] private Image iconImage;

    [Header("Metadata fields (optional per item)")]
    [SerializeField] private TMP_Text creatorText;        // any item may credit a Creator
    [SerializeField] private GameObject descriptionGroup; // the "Description" section (title + content); hidden when empty
    [SerializeField] private TMP_Text descriptionText;    // the Content body text inside descriptionGroup
    [SerializeField] private Button askButton;            // ask the creator — always shown
    [SerializeField] private TMP_Text winRateText;        // BotScriptItem only

    [Header("Buttons")]
    [SerializeField] private Button buyButton;
    [SerializeField] private TMP_Text buyButtonLabel; // optional: flips to "Owned" when owned
    [SerializeField] private Button exitButton;       // closes the panel

    private CatalogItem current;
    private bool purchasing;

    void Awake()
    {
        if (buyButton != null) buyButton.onClick.AddListener(OnBuyClicked);
        if (exitButton != null) exitButton.onClick.AddListener(Hide);
        if (askButton != null) askButton.onClick.AddListener(OnAskClicked);
    }

    /// <summary>
    /// Open the detail panel for a catalog item. Wired from MarketListController (buy) and
    /// InventoryController (view an owned item — Buy shows as "Owned").
    /// </summary>
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

        RefreshMetadataFields(item);
        RefreshBuyState();

        if (panel != null)
        {
            SFXManager.Instance.Play2D("ui_accept");
            // Draw on top of whatever opened it (Market list or the Inventory panel).
            panel.transform.SetAsLastSibling();
            panel.SetActive(true);
        }
    }

    public void Hide()
    {
        if (panel == null) return;
        SFXManager.Instance.Play2D("ui_accept");
        panel.SetActive(false);
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

    // Creator/Description are common to every category; each element shows only when the
    // item actually provides a value (so a bare skin hides them). WinRate is bot-script
    // only. Per-field toggles — no single wrapping group — so the panel's core fields
    // (name/price/icon/Buy) always stay visible regardless of item type.
    private void RefreshMetadataFields(CatalogItem item)
    {
        if (creatorText != null)
        {
            creatorText.text = $"Creator: {item.Creator}";
            creatorText.gameObject.SetActive(!string.IsNullOrEmpty(item.Creator));
        }
        // Ask is always available (any item, any type); it does not depend on Creator.

        bool hasDescription = !string.IsNullOrEmpty(item.Description);
        if (descriptionText != null) descriptionText.text = item.Description;
        // Toggle the whole section (title + content) when assigned; otherwise fall back to
        // just the content text so a missing group reference still hides something sensible.
        if (descriptionGroup != null) descriptionGroup.SetActive(hasDescription);
        else if (descriptionText != null) descriptionText.gameObject.SetActive(hasDescription);

        var botScript = item as BotScriptItem;
        if (winRateText != null)
        {
            if (botScript != null) winRateText.text = $"Win-rate: {botScript.WinRate:0.00}";
            winRateText.gameObject.SetActive(botScript != null);
        }
    }

    // TODO: no design yet for what "Ask" does (contact the creator? open a chat thread?).
    // Wired up so the button is functional once that's decided; for now it's a no-op.
    private void OnAskClicked()
    {
        Logger.Warning("[Market] Ask is not implemented yet.");
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
