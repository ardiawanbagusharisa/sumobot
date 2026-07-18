using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class MarketManager : MonoBehaviour
{
    public GameObject PanelItemDetail;
    public GameObject PanelChat;
    public GameObject PanelInventory;
    public GameObject buttonUnfoldChat;
    public GameObject buttonUnfoldInventory;

    [Header("Item Detail (populated by ShowItemDetail(CatalogItem))")]
    public TMP_Text DetailNameText;
    public TMP_Text DetailPriceText;
    public Image DetailIconImage;

    [Header("Item Detail buttons (wired here, not via Editor OnClick — see HideItemDetail)")]
    public Button DetailExitButton;

	void Awake() {
		if (DetailExitButton != null)
			DetailExitButton.onClick.AddListener(HideItemDetail);
	}

	public void ToggleChat() {
        SFXManager.Instance.Play2D("ui_accept");
		bool isActive = PanelChat.activeSelf;
        PanelChat.SetActive(!isActive);
        buttonUnfoldChat.SetActive(isActive);
	}

    public void ToggleInventory() {
        SFXManager.Instance.Play2D("ui_accept");
		bool isActive = PanelInventory.activeSelf;
		PanelInventory.SetActive(!isActive);
		buttonUnfoldInventory.SetActive(isActive);
	}

	public void ShowItemDetail() {
		SFXManager.Instance.Play2D("ui_accept");
		PanelItemDetail.SetActive(true);
    }

	// Used by the dynamically generated Market list (MarketListController /
	// ItemCellView), which knows which CatalogItem was clicked. Only fills the fields
	// backed by CatalogItem's schema (name/price/icon) — Creator/Win-rate/Description
	// in PopUpDetail are hand-authored flavor text with no per-item data source yet.
	public void ShowItemDetail(CatalogItem item) {
		SFXManager.Instance.Play2D("ui_accept");
		DetailNameText.text = item.DisplayName;
		DetailPriceText.text = $"SG {item.Price}";
		if (!string.IsNullOrEmpty(item.IconResourcePath))
			DetailIconImage.sprite = Resources.Load<Sprite>(item.IconResourcePath);
		PanelItemDetail.SetActive(true);
	}

	public void HideItemDetail() {
		SFXManager.Instance.Play2D("ui_accept");
		PanelItemDetail.SetActive(false);
	}
}
