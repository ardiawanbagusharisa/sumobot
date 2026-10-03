using System;
using System.Collections.Generic;
using System.Linq;
using SumoCore;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Turns BotCreator's existing ButtonMarket placeholder into an owned-skins
/// browser. This is intentionally read-only: equipment is changed with the
/// existing wheel/body/accessory arrows in the Robot Creation panel.
/// </summary>
public sealed class BotCreatorInventoryController : MonoBehaviour
{
    private const int MaxVisibleItems = 12;

    private bool initialized;
    private GameObject overlay;
    private RectTransform itemArea;
    private TMP_Text statusText;
    private TMP_FontAsset font;

    public static void EnsureCreated()
    {
        Button marketButton = Resources.FindObjectsOfTypeAll<Button>()
            .FirstOrDefault(candidate =>
                candidate.name == "ButtonMarket" &&
                candidate.gameObject.scene.IsValid() &&
                candidate.gameObject.scene.name == "BotCreator");
        if (marketButton == null)
            return;

        Canvas canvas = marketButton.GetComponentInParent<Canvas>();
        if (canvas == null)
            return;

        BotCreatorInventoryController controller =
            canvas.GetComponent<BotCreatorInventoryController>() ??
            canvas.gameObject.AddComponent<BotCreatorInventoryController>();
        controller.Initialize(marketButton);
    }

    private void Initialize(Button marketButton)
    {
        if (initialized)
            return;

        initialized = true;
        font = Resources.FindObjectsOfTypeAll<TMP_Text>()
            .Where(text => text.gameObject.scene == gameObject.scene)
            .Select(text => text.font)
            .FirstOrDefault(candidate => candidate != null);
        BuildPanel();
        marketButton.onClick.AddListener(Toggle);
        AddButtonCaption(marketButton.transform, "Owned");
    }

    private void Toggle()
    {
        if (overlay == null)
            return;

        bool show = !overlay.activeSelf;
        overlay.SetActive(show);
        if (show)
        {
            overlay.transform.SetAsLastSibling();
            RefreshOwnedItems();
        }
    }

    private void BuildPanel()
    {
        overlay = CreateImage("OwnedSkinsPanel", transform, new Color(0.04f, 0.06f, 0.14f, 0.78f));
        RectTransform overlayRect = overlay.GetComponent<RectTransform>();
        overlayRect.anchorMin = Vector2.zero;
        overlayRect.anchorMax = Vector2.one;
        overlayRect.offsetMin = Vector2.zero;
        overlayRect.offsetMax = Vector2.zero;

        GameObject card = CreateImage("Card", overlay.transform, new Color(0.92f, 0.95f, 1f, 1f));
        RectTransform cardRect = card.GetComponent<RectTransform>();
        cardRect.anchorMin = cardRect.anchorMax = new Vector2(0.5f, 0.5f);
        cardRect.sizeDelta = new Vector2(900f, 590f);

        AddText(card.transform, "OWNED SKINS", new Vector2(0f, 255f), new Vector2(700f, 50f), 34f, FontStyles.Bold);
        statusText = AddText(card.transform, "Use the part arrows to equip an owned skin.", new Vector2(0f, 215f), new Vector2(700f, 32f), 18f);
        Button close = AddButton(card.transform, "Close", new Vector2(402f, 255f), new Vector2(55f, 45f));
        close.GetComponentInChildren<TMP_Text>().SetText("X");
        close.onClick.AddListener(() => overlay.SetActive(false));

        var area = new GameObject("Items", typeof(RectTransform));
        area.transform.SetParent(card.transform, false);
        itemArea = area.GetComponent<RectTransform>();
        itemArea.anchorMin = itemArea.anchorMax = new Vector2(0.5f, 0.5f);
        itemArea.anchoredPosition = new Vector2(0f, -25f);
        itemArea.sizeDelta = new Vector2(820f, 430f);
        overlay.SetActive(false);
    }

    private void RefreshOwnedItems()
    {
        foreach (Transform child in itemArea)
            Destroy(child.gameObject);

        PlayerData data = GameServices.PlayerData?.Current;
        if (data == null || GameServices.Catalog == null)
        {
            statusText.SetText("Sign in before opening owned skins.");
            return;
        }

        List<CatalogItem> items = data.OwnedItemIds
            .Select(GameServices.Catalog.GetById)
            .Where(item => item != null &&
                string.Equals(item.Type, "Skin", StringComparison.OrdinalIgnoreCase) &&
                IsSupportedSlot(item.Slot))
            .Take(MaxVisibleItems)
            .ToList();

        if (items.Count == 0)
        {
            statusText.SetText("No owned skins yet. Buy one in the Market first.");
            return;
        }

        statusText.SetText($"{items.Count} owned skin{(items.Count == 1 ? string.Empty : "s")} — use the part arrows to equip");
        for (int index = 0; index < items.Count; index++)
            CreateItemCard(items[index], index);
    }

    private void CreateItemCard(CatalogItem item, int index)
    {
        int column = index % 4;
        int row = index / 4;
        Vector2 position = new(-300f + column * 200f, 135f - row * 140f);
        GameObject card = CreateImage($"Owned_{item.Id}", itemArea, new Color(0.37f, 0.48f, 0.83f, 1f));
        RectTransform cardRect = card.GetComponent<RectTransform>();
        cardRect.anchorMin = cardRect.anchorMax = new Vector2(0.5f, 0.5f);
        cardRect.anchoredPosition = position;
        cardRect.sizeDelta = new Vector2(180f, 125f);
        TMP_Text label = AddText(card.transform, item.DisplayName, new Vector2(0f, -45f), new Vector2(170f, 32f), 15f, FontStyles.Bold);
        label.color = Color.white;
        RectTransform labelRect = label.rectTransform;
        labelRect.anchoredPosition = new Vector2(0f, -45f);
        labelRect.sizeDelta = new Vector2(170f, 32f);
        label.fontSize = 15f;

        var iconObject = new GameObject("Icon", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        iconObject.transform.SetParent(card.transform, false);
        RectTransform iconRect = iconObject.GetComponent<RectTransform>();
        iconRect.anchorMin = iconRect.anchorMax = new Vector2(0.5f, 0.5f);
        iconRect.anchoredPosition = new Vector2(0f, 14f);
        iconRect.sizeDelta = new Vector2(72f, 72f);
        Image icon = iconObject.GetComponent<Image>();
        icon.preserveAspect = true;
        icon.raycastTarget = false;
        icon.sprite = Resources.Load<Sprite>(item.IconResourcePath);
        icon.color = ParseItemColor(item);
    }

    private static bool IsSupportedSlot(string slot)
    {
        return string.Equals(slot, "Body", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(slot, "Wheel", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(slot, "Eye", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(slot, "Accessory", StringComparison.OrdinalIgnoreCase);
    }

    private static Color ParseItemColor(CatalogItem item)
    {
        return !string.IsNullOrEmpty(item.IconColor) &&
            ColorUtility.TryParseHtmlString(item.IconColor, out Color color)
                ? color
                : Color.white;
    }

    private void AddButtonCaption(Transform button, string caption)
    {
        TMP_Text label = AddText(button, caption, new Vector2(-4f, -48f), new Vector2(110f, 26f), 14f, FontStyles.Bold);
        label.color = Color.white;
    }

    private Button AddButton(Transform parent, string name, Vector2 position, Vector2 size)
    {
        GameObject buttonObject = CreateImage(name, parent, new Color(0.37f, 0.48f, 0.83f, 1f));
        RectTransform rect = buttonObject.GetComponent<RectTransform>();
        rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        Button button = buttonObject.AddComponent<Button>();
        TMP_Text text = AddText(buttonObject.transform, name, Vector2.zero, size, 18f, FontStyles.Bold);
        text.color = Color.white;
        return button;
    }

    private TMP_Text AddText(
        Transform parent,
        string value,
        Vector2 position,
        Vector2 size,
        float fontSize,
        FontStyles style = FontStyles.Normal)
    {
        var textObject = new GameObject("Text", typeof(RectTransform), typeof(CanvasRenderer), typeof(TextMeshProUGUI));
        textObject.transform.SetParent(parent, false);
        RectTransform rect = textObject.GetComponent<RectTransform>();
        rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        TextMeshProUGUI text = textObject.GetComponent<TextMeshProUGUI>();
        text.font = font;
        text.fontSize = fontSize;
        text.fontStyle = style;
        text.alignment = TextAlignmentOptions.Center;
        text.color = new Color(0.12f, 0.14f, 0.22f, 1f);
        text.textWrappingMode = TextWrappingModes.Normal;
        text.richText = false;
        text.SetText(value);
        return text;
    }

    private static GameObject CreateImage(string name, Transform parent, Color color)
    {
        var imageObject = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        imageObject.transform.SetParent(parent, false);
        imageObject.GetComponent<Image>().color = color;
        return imageObject;
    }
}
