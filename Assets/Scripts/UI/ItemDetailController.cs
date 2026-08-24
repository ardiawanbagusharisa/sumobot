using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Sole owner of the shared item-detail panel across the whole MainMenu scene: its
// visibility (show/hide + exit button), its content (name/price/icon for the clicked
// CatalogItem), and the transaction (Trade.BuyAsync for Shop items, Market.BuyAsync for
// Community listings). MarketListController (Shop), CommunityListController (Community),
// and InventoryController (Garage) all call Show(item) / Show(item, listing); none of them
// own their own detail panel instance.
//
// After a successful buy, the coin balance and inventory refresh themselves via the
// IPlayerDataService events (see CoinBalanceView / InventoryController); this controller only
// updates its own Buy button. The Shop is buy-only; the Community tab additionally supports
// Unlist/Reprice for the current player's own listings (see sellerControls).
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
    [SerializeField] private Button sellButton;       // list an owned + self-authored item that isn't listed yet (ListAsync)
    [SerializeField] private Button equipButton;      // equip an owned skin (hidden for non-skins)
    [SerializeField] private TMP_Text equipButtonLabel; // optional: flips to "Equipped"
    [SerializeField] private Button exitButton;       // closes the panel

    [Header("Seller controls (Community tab, own listing only)")]
    [SerializeField] private GameObject sellerControls; // Unlist/Reprice buttons — shown instead of Buy when viewing your own listing
    [SerializeField] private Button unlistButton;
    [SerializeField] private Button repriceUpButton;   // +RepriceStep coins
    [SerializeField] private Button repriceDownButton; // -RepriceStep coins (floored at 0)

    private const int RepriceStep = 10;

    // Opening price when an item is first listed via Sell. The item's own catalog Price is
    // used when it carries one (>0); otherwise this floor keeps the listing from starting free.
    // The seller can adjust it right after via the Reprice controls.
    private const int DefaultListPrice = 50;

    private CatalogItem current;
    private MarketListing currentListing; // set only when opened from the Community tab
    private bool purchasing;
    private bool equipping;
    private bool listingBusy;

    void Awake()
    {
        if (buyButton != null) buyButton.onClick.AddListener(OnBuyClicked);
        if (sellButton != null) sellButton.onClick.AddListener(OnSellClicked);
        if (equipButton != null) equipButton.onClick.AddListener(OnEquipClicked);
        if (exitButton != null) exitButton.onClick.AddListener(Hide);
        if (askButton != null) askButton.onClick.AddListener(OnAskClicked);
        if (unlistButton != null) unlistButton.onClick.AddListener(OnUnlistClicked);
        if (repriceUpButton != null) repriceUpButton.onClick.AddListener(() => OnRepriceClicked(RepriceStep));
        if (repriceDownButton != null) repriceDownButton.onClick.AddListener(() => OnRepriceClicked(-RepriceStep));
    }

    /// <summary>
    /// Open the detail panel for a Shop item or an owned Inventory item (no Market listing
    /// involved — Buy goes through the Shop's ITradeService). Wired from MarketListController
    /// (buy) and InventoryController (view an owned item — Buy shows as "Owned").
    /// </summary>
    public void Show(CatalogItem item) => Show(item, null);

    /// <summary>
    /// Open the detail panel for a Community-tab listing. If the current player is the
    /// listing's seller, shows Unlist/Reprice controls instead of Buy; otherwise behaves
    /// like a normal purchase but through GameServices.Market.BuyAsync. Wired from
    /// CommunityListController.
    /// </summary>
    public void Show(CatalogItem item, MarketListing listing)
    {
        current = item;
        currentListing = listing;

        if (nameText != null) nameText.text = item.DisplayName;
        if (priceText != null) priceText.text = $"{(listing != null ? listing.Price : item.Price)}";
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
        RefreshEquipState();

        if (panel != null)
        {
            SFXManager.Instance.Play2D("ui_accept");
            // Draw on top of whatever opened it (Shop list, Community list, or Inventory panel).
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
    // Shop items buy through ITradeService; a Community listing buys through IMarketService
    // (which also handles the 90/10 seller/sink split) instead.
    private async void OnBuyClicked()
    {
        if (current == null || purchasing) return;

        purchasing = true;
        if (buyButton != null) buyButton.interactable = false;

        try
        {
            var result = currentListing != null
                ? await GameServices.Market.BuyAsync(currentListing.ListingId)
                : await GameServices.Trade.BuyAsync(current.Id);
            if (!result.Success)
                Logger.Warning($"[Market] Buy '{current.Id}' failed: {result.Error}");
        }
        finally
        {
            purchasing = false;
            RefreshBuyState(); // re-enables the button unless the item is now owned
        }
    }

    // Unlist/Reprice only ever act on currentListing, which is only set when Show(item,
    // listing) opened this panel from the Community tab and RefreshBuyState confirmed the
    // current player is the seller (sellerControls is only active in that case).
    private async void OnUnlistClicked()
    {
        if (currentListing == null || listingBusy) return;

        listingBusy = true;
        if (unlistButton != null) unlistButton.interactable = false;

        try
        {
            var result = await GameServices.Market.UnlistAsync(currentListing.ListingId);
            if (!result.Success)
                Logger.Warning($"[Market] Unlist '{currentListing.ListingId}' failed: {result.Error}");
            else
                Hide(); // the listing is gone from the active set — nothing left to show here
        }
        finally
        {
            listingBusy = false;
            if (unlistButton != null) unlistButton.interactable = true;
        }
    }

    private async void OnRepriceClicked(int delta)
    {
        if (currentListing == null || listingBusy) return;

        int newPrice = Mathf.Max(0, currentListing.Price + delta);
        listingBusy = true;

        try
        {
            var result = await GameServices.Market.RepriceAsync(currentListing.ListingId, newPrice);
            if (!result.Success)
                Logger.Warning($"[Market] Reprice '{currentListing.ListingId}' failed: {result.Error}");
            else if (priceText != null)
                priceText.text = $"{currentListing.Price}"; // mutated in place by RepriceAsync
        }
        finally
        {
            listingBusy = false;
        }
    }

    // List an owned, self-authored item that has no active listing yet. On success the item
    // is now the player's own listing, so re-open the panel in listing mode (Show(item,
    // listing)) — that swaps the Sell button for the Unlist/Reprice controls, and the
    // opening price can be tuned from there. Guarded against re-entrancy like Buy/Reprice.
    private async void OnSellClicked()
    {
        if (current == null || listingBusy) return;

        listingBusy = true;
        if (sellButton != null) sellButton.interactable = false;

        try
        {
            int price = current.Price > 0 ? current.Price : DefaultListPrice;
            var result = await GameServices.Market.ListAsync(current.Id, price);
            if (!result.Success)
            {
                Logger.Warning($"[Market] List '{current.Id}' failed: {result.Error}");
                return;
            }

            var listing = FindOwnActiveListing(current.Id);
            if (listing != null) Show(current, listing); // flip to Unlist/Reprice on the new listing
        }
        finally
        {
            listingBusy = false;
            if (sellButton != null) sellButton.interactable = true;
        }
    }

    // The current player's active listing for an item, or null. Used both to decide whether
    // Sell is offered (none yet) and to re-open the panel on the freshly-created listing.
    private static MarketListing FindOwnActiveListing(string itemId)
    {
        var market = GameServices.Market;
        string selfId = GameServices.PlayerData?.Current?.PlayerId;
        if (market == null || string.IsNullOrEmpty(selfId)) return null;

        foreach (var listing in market.ActiveListings)
            if (listing.ItemId == itemId && listing.SellerId == selfId)
                return listing;
        return null;
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

    // Equip an owned skin into its slot. Buttons need a void handler; guard re-entrancy so a
    // double-tap can't fire two equips. The costume applies on the next bot spawn (SetCostume
    // reads the seat's loadout); persistence is handled by EquipAsync -> SaveAsync.
    private async void OnEquipClicked()
    {
        if (current == null || equipping || !IsEquippable(current)) return;

        equipping = true;
        if (equipButton != null) equipButton.interactable = false;

        try
        {
            var result = await GameServices.PlayerData.EquipAsync(current.Slot, current.Id);
            if (!result.Success)
                Logger.Warning($"[Market] Equip '{current.Id}' failed: {result.Error}");
        }
        finally
        {
            equipping = false;
            RefreshEquipState();
        }
    }

    // Only owned skins that map to a costume part (SkinItem with a Slot + PartSprite) can be
    // equipped. Bot scripts, modules and Body skins (no PartSprite) hide the Equip button.
    private static bool IsEquippable(CatalogItem item)
        => item is SkinItem skin
           && !string.IsNullOrEmpty(skin.Slot)
           && !string.IsNullOrEmpty(skin.PartSprite);

    private void RefreshEquipState()
    {
        bool owned = current != null && (GameServices.PlayerData?.Current?.Owns(current.Id) ?? false);
        bool canEquip = owned && IsEquippable(current);
        if (equipButton != null) equipButton.gameObject.SetActive(canEquip);
        if (!canEquip) return;

        var equippedBySlot = GameServices.PlayerData.ActiveLoadout?.EquippedBySlot;
        bool equipped = equippedBySlot != null
            && equippedBySlot.TryGetValue(current.Slot, out var eq) && eq == current.Id;
        if (equipButton != null) equipButton.interactable = !equipped;
        if (equipButtonLabel != null) equipButtonLabel.text = equipped ? "Equipped" : "Equip";
    }

    // TODO: no design yet for what "Ask" does (contact the creator? open a chat thread?).
    // Wired up so the button is functional once that's decided; for now it's a no-op.
    private void OnAskClicked()
    {
        Logger.Warning("[Market] Ask is not implemented yet.");
    }

    // A player can't buy what they already own (LocalTradeService/LocalMarketService reject
    // it too); reflect that in the button so the affordance matches the rule. Three mutually
    // exclusive modes drive which control shows:
    //   - own listing (Community tab, your own row) -> Unlist/Reprice, no Buy;
    //   - owned + self-authored + not yet listed    -> Sell (ListAsync), no Buy;
    //   - everything else                            -> Buy (disabled/"Owned" when owned).
    private void RefreshBuyState()
    {
        string selfId = GameServices.PlayerData?.Current?.PlayerId;
        bool owned = current != null && (GameServices.PlayerData?.Current?.Owns(current.Id) ?? false);

        bool isOwnListing = currentListing != null && currentListing.SellerId == selfId;

        // Sell only when viewing the item outside the Community tab (no listing carried in),
        // it's ours to sell (owned + Author == self), and it isn't already listed.
        bool canSell = !isOwnListing
            && currentListing == null
            && owned
            && current != null
            && !string.IsNullOrEmpty(selfId)
            && current.Author == selfId
            && FindOwnActiveListing(current.Id) == null;

        if (sellerControls != null) sellerControls.SetActive(isOwnListing);
        if (sellButton != null) sellButton.gameObject.SetActive(canSell);
        if (buyButton != null) buyButton.gameObject.SetActive(!isOwnListing && !canSell);
        if (isOwnListing || canSell) return;

        if (buyButton != null) buyButton.interactable = !owned;
        if (buyButtonLabel != null) buyButtonLabel.text = owned ? "Owned" : "Buy";
    }
}
