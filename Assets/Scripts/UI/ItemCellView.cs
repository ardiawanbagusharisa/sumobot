using System;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// View for one row instantiated into a Scroll View / Content by MarketListController (Shop),
// CommunityListController (Community), InventoryController, and GarageInventoryController.
// Serialized refs are preferred; if left unassigned they fall back to the child names baked
// into Assets/Prefabs/ItemCell.prefab (ItemName / ItemImg / Price/CoinTxt) so existing prefab
// instances keep working without re-wiring.
public class ItemCellView : MonoBehaviour
{
    [SerializeField] private TMP_Text nameText;
    [SerializeField] private Image iconImage;
    [SerializeField] private TMP_Text priceText;
    [SerializeField] private GameObject priceRoot;        // optional: hidden for the Unequip cell
    [SerializeField] private GameObject equippedIndicator; // optional: check/highlight shown when equipped

    private Button button;

    public CatalogItem Item { get; private set; }

    /// <summary>True for the fixed "Empty / Unequip" cell (no bound item).</summary>
    public bool IsUnequip { get; private set; }

    /// <summary>Set when this cell was bound from a Community-tab Market listing (null for Shop/Inventory cells).</summary>
    public MarketListing Listing { get; private set; }

    /// <summary>Raised when this cell is clicked, carrying the bound item (null for the Unequip cell).</summary>
    public event Action<CatalogItem> Clicked;

    /// <summary>Raised when a Community-tab cell (bound with a listing) is clicked.</summary>
    public event Action<CatalogItem, MarketListing> ClickedWithListing;

    void Awake()
    {
        // Fall back to the legacy prefab child paths when serialized refs aren't assigned.
        if (nameText == null) nameText = transform.Find("ItemName")?.GetComponent<TMP_Text>();
        if (iconImage == null) iconImage = transform.Find("ItemImg")?.GetComponent<Image>();
        if (priceText == null) priceText = transform.Find("Price/CoinTxt")?.GetComponent<TMP_Text>();

        // The prefab's Button carries a stale persistent onClick copied from the original
        // static scene item; replace it with the dynamic click below.
        button = GetComponent<Button>();
        button.onClick = new Button.ButtonClickedEvent();
        button.onClick.AddListener(() =>
        {
            Clicked?.Invoke(Item);
            if (Listing != null) ClickedWithListing?.Invoke(Item, Listing);
        });
    }

    public void Bind(CatalogItem item) => Bind(item, null);

    /// <summary>
    /// Bind for a Community-tab row: shows the listing's asking price (which may differ
    /// from the item's catalog Price) instead of item.Price, and remembers the listing so
    /// ClickedWithListing can carry it to the detail panel.
    /// </summary>
    public void Bind(CatalogItem item, MarketListing listing)
    {
        Item = item;
        Listing = listing;
        IsUnequip = false;

        if (nameText != null) nameText.text = item.DisplayName;
        if (priceText != null) priceText.text = $"{(listing != null ? listing.Price : item.Price)}";
        if (priceRoot != null) priceRoot.SetActive(true);

        if (iconImage != null)
        {
            if (!string.IsNullOrEmpty(item.IconResourcePath))
                iconImage.sprite = Resources.Load<Sprite>(item.IconResourcePath);

            iconImage.color = !string.IsNullOrEmpty(item.IconColor) && ColorUtility.TryParseHtmlString(item.IconColor, out var color)
                ? color
                : Color.white;
        }

        SetEquipped(false);
    }

    /// <summary>
    /// Bind this cell as the fixed "Empty / Unequip" cell for the Garage: no item, no price,
    /// a placeholder label. Clicking it raises Clicked(null) so the owner can clear the slot.
    /// </summary>
    public void BindUnequip(Sprite icon = null)
    {
        Item = null;
        Listing = null;
        IsUnequip = true;

        if (nameText != null) nameText.text = "Default";
        if (priceRoot != null) priceRoot.SetActive(false);
        else if (priceText != null) priceText.text = string.Empty;

        if (iconImage != null)
        {
            iconImage.sprite = icon; // null renders as an empty slot; assign an X icon in the editor if desired
            iconImage.color = Color.white;
        }

        SetEquipped(false);
    }

    /// <summary>Show/hide the equipped mark. No-op if the prefab has no indicator assigned.</summary>
    public void SetEquipped(bool equipped)
    {
        if (equippedIndicator != null) equippedIndicator.SetActive(equipped);
    }
}
