using System;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// View for one row instantiated into the Market's Scroll View / Content by
// MarketListController. Mirrors the hierarchy baked into Assets/Prefabs/ItemCell.prefab
// (root "ItemCell" > "GameObject" > ItemName / ItemImg / Price / Price/CoinTxt).
public class ItemCellView : MonoBehaviour
{
    private TMP_Text nameText;
    private Image iconImage;
    private TMP_Text priceText;
    private Button button;

    public CatalogItem Item { get; private set; }

    /// <summary>Raised when this cell is clicked, carrying the bound item.</summary>
    public event Action<CatalogItem> Clicked;

    void Awake()
    {
        nameText = transform.Find("ItemName").GetComponent<TMP_Text>();
        iconImage = transform.Find("ItemImg").GetComponent<Image>();
        priceText = transform.Find("Price/CoinTxt").GetComponent<TMP_Text>();

        // The prefab's Button carries a stale persistent onClick copied from the original
        // static scene item; replace it with the dynamic click below.
        button = GetComponent<Button>();
        button.onClick = new Button.ButtonClickedEvent();
        button.onClick.AddListener(() => Clicked?.Invoke(Item));
    }

    public void Bind(CatalogItem item)
    {
        Item = item;
        nameText.text = item.DisplayName;
        priceText.text = $"{item.Price}";

        if (!string.IsNullOrEmpty(item.IconResourcePath))
            iconImage.sprite = Resources.Load<Sprite>(item.IconResourcePath);

        iconImage.color = !string.IsNullOrEmpty(item.IconColor) && ColorUtility.TryParseHtmlString(item.IconColor, out var color)
            ? color
            : Color.white;
    }
}
