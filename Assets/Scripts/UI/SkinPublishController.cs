using System;
using System.Collections.Generic;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Create & Publish panel for the Garage: the player designs a color-variant skin from a
// tintable base sprite + a chosen tint + a name, previews it live, and publishes it as a
// Source=Player, Author=self, owned SkinItem via GameServices.Publish (decision-7). Once
// published it lands in the inventory and can be listed for sale (ItemDetailController's
// Sell -> IMarketService.ListAsync).
//
// Only SkinItem passes the publish gate today (bot scripts are gated on the sandbox), which
// is why the sellable creation is a skin. The base options are the *_Tintable sprites the
// tint system is built for (Body / Accessory); a flat tint on a numbered variant would tint
// the whole colored sprite, so those are intentionally not offered here.
public class SkinPublishController : MonoBehaviour
{
    // One selectable base the player can recolor. Serialized so designers can add bases
    // (new *_Tintable art) without code changes. PartSprite is what the bot renders in
    // battle; IconResourcePath is the cropped market-grid icon (see SkinItem / CatalogItem).
    [Serializable]
    public class SkinBaseOption
    {
        public string displayName;      // e.g. "Round Accessory"
        public string slot;             // "Accessory" or "Body"
        public string partSprite;       // "Sprites/Character/Accessory_2_Tintable"
        public string iconResourcePath; // "Sprites/MarketIcons/Accessory_2_Tintable"
    }

    [Header("Base selection")]
    [SerializeField] private List<SkinBaseOption> baseOptions = new();
    [SerializeField] private TMP_Text baseNameLabel; // shows the current base's displayName

    [Header("Tint (RGB)")]
    [SerializeField] private Slider redSlider;   // 0..1
    [SerializeField] private Slider greenSlider; // 0..1
    [SerializeField] private Slider blueSlider;  // 0..1

    [Header("Naming")]
    [SerializeField] private TMP_InputField nameInput;

    [Header("Preview")]
    [SerializeField] private Image previewImage; // shows the base sprite under the live tint

    [Header("Actions")]
    [SerializeField] private Button publishButton;
    [SerializeField] private TMP_Text statusText; // optional: publish result / validation hints

    // A published creation has no price until it is listed for sale; keep it 0 so the item's
    // own Price never front-runs the price the seller sets at list time (see ItemDetailController).
    private const int PublishedItemPrice = 0;

    private int baseIndex;
    private bool publishing;

    void Awake()
    {
        if (publishButton != null) publishButton.onClick.AddListener(OnPublishClicked);
        if (redSlider != null) redSlider.onValueChanged.AddListener(_ => RefreshPreview());
        if (greenSlider != null) greenSlider.onValueChanged.AddListener(_ => RefreshPreview());
        if (blueSlider != null) blueSlider.onValueChanged.AddListener(_ => RefreshPreview());
    }

    void OnEnable()
    {
        if (statusText != null) statusText.text = string.Empty;
        RefreshBase();
    }

    /// <summary>Wire to ◀ / ▶ buttons (dir = -1 / +1) to step through the base options.</summary>
    public void CycleBase(int dir)
    {
        if (baseOptions.Count == 0) return;
        baseIndex = (baseIndex + dir + baseOptions.Count) % baseOptions.Count;
        RefreshBase();
    }

    private void RefreshBase()
    {
        SkinBaseOption option = CurrentOption();
        if (baseNameLabel != null) baseNameLabel.text = option != null ? option.displayName : "—";
        RefreshPreview();
    }

    private void RefreshPreview()
    {
        if (previewImage == null) return;

        SkinBaseOption option = CurrentOption();
        string path = option != null
            ? (!string.IsNullOrEmpty(option.iconResourcePath) ? option.iconResourcePath : option.partSprite)
            : null;

        previewImage.sprite = string.IsNullOrEmpty(path) ? null : Resources.Load<Sprite>(path);
        previewImage.color = CurrentColor();
    }

    private SkinBaseOption CurrentOption()
        => baseIndex >= 0 && baseIndex < baseOptions.Count ? baseOptions[baseIndex] : null;

    private Color CurrentColor() => new Color(
        redSlider != null ? redSlider.value : 1f,
        greenSlider != null ? greenSlider.value : 1f,
        blueSlider != null ? blueSlider.value : 1f);

    // Buttons need a void handler; guard re-entrancy so a double-tap can't publish twice.
    private async void OnPublishClicked()
    {
        if (publishing) return;

        SkinBaseOption option = CurrentOption();
        if (option == null)
        {
            SetStatus("Pick a base to recolor first.");
            return;
        }

        publishing = true;
        if (publishButton != null) publishButton.interactable = false;

        try
        {
            string chosenName = nameInput != null ? nameInput.text?.Trim() : null;
            var draft = new SkinItem
            {
                // PublishAsync stamps Id / Source / Author / Type — we only supply the design.
                DisplayName = string.IsNullOrEmpty(chosenName) ? option.displayName : chosenName,
                Slot = option.slot,
                PartSprite = option.partSprite,
                IconResourcePath = option.iconResourcePath,
                IconColor = "#" + ColorUtility.ToHtmlStringRGB(CurrentColor()),
                Price = PublishedItemPrice,
            };

            var result = await GameServices.Publish.PublishAsync(draft);
            if (!result.Success)
            {
                Logger.Warning($"[Market] Publish failed: {result.Error}");
                SetStatus($"Publish failed: {result.Error}");
                return;
            }

            // Owned + Author=self now; it shows up in the inventory via the grant event.
            SetStatus($"Published \"{draft.DisplayName}\". It's in your inventory — open it to sell.");
        }
        finally
        {
            publishing = false;
            if (publishButton != null) publishButton.interactable = true;
        }
    }

    private void SetStatus(string message)
    {
        if (statusText != null) statusText.text = message;
    }
}
