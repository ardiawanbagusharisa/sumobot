using UnityEngine;
using UnityEngine.UI;

// Owns the Market screen's PANEL VISIBILITY only: the chat/inventory toggles and the
// item-detail panel's show/hide (plus its exit button and SFX). It is still bound to many
// legacy scene onClicks that call the no-arg ShowItemDetail(), so that entry point stays.
//
// The item-detail CONTENT + purchase behaviour lives in ItemDetailController, which calls
// ShowItemDetail() to open the panel. Keeping the two apart stops this class from growing
// into a god object as buy/sell features land.
public class MarketManager : MonoBehaviour
{
    public GameObject PanelItemDetail;
    public GameObject PanelChat;
    public GameObject PanelInventory;
    public GameObject buttonUnfoldChat;
    public GameObject buttonUnfoldInventory;

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

	// Opens the detail panel. Bound to many legacy scene onClicks and reused by
	// ItemDetailController.Show(CatalogItem) after it fills the per-item content.
	public void ShowItemDetail() {
		SFXManager.Instance.Play2D("ui_accept");
		PanelItemDetail.SetActive(true);
    }

	public void HideItemDetail() {
		SFXManager.Instance.Play2D("ui_accept");
		PanelItemDetail.SetActive(false);
	}
}
